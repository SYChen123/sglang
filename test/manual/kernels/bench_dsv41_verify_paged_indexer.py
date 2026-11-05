"""Compare paged FP8 scorer variants with request-major DSpark verify rows."""

import argparse
import importlib.util
import random
import statistics
import sys

import torch

from sglang.kernels.ops.attention.dsv4.fp4_indexer import fp4_index_logits_paged


def make_case(requests, width, ratio, tokens=6):
    device = "cuda"
    page_size = 256 // ratio
    num_pages = (width + page_size - 1) // page_size
    # Random physical page order with sequential slots within each page.
    pages = torch.randperm(requests * num_pages, device=device).reshape(requests, -1)
    slots = pages[:, :, None] * page_size + torch.arange(page_size, device=device)
    slots = slots.flatten(1)[:, :width]
    req_table = slots[:, :, None] * ratio + torch.arange(ratio, device=device)
    req_table = req_table.flatten(1).to(torch.int32)
    keys = torch.randint(
        0, 256, (requests * num_pages, page_size * 68), device=device, dtype=torch.uint8
    )
    keys[:, page_size * 64 :].random_(123, 129)
    q = torch.randint(0, 16, (requests * tokens, 32, 128), device=device)
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device)
    q = (levels[q % 8] * torch.where(q < 8, 1.0, -1.0) * 0.25).to(torch.bfloat16)
    weights = torch.randn(requests * tokens, 32, device=device, dtype=torch.bfloat16)
    req = torch.arange(requests, device=device).repeat_interleave(tokens)
    pos = (
        width * ratio
        - 1
        - torch.arange(requests, device=device)[:, None]
        * min(193, width // (2 * requests))
        - torch.arange(tokens - 1, -1, -1, device=device)
    )
    lens = ((pos + 1) // ratio).flatten().to(torch.int32)
    return q, weights, req, req_table, lens, keys, page_size, width, ratio


def check(results, lengths, width):
    base = results[0]
    for result in results[1:]:
        for n, (a, b) in enumerate(zip(base, result, strict=True)):
            scale = 8 if n else 1
            visible = torch.arange(a.shape[1], device=a.device) < (
                (lengths[:, None] + scale - 1) // scale
            )
            x, y = a[visible], b[visible]
            if not torch.equal(x, y):
                raise AssertionError(
                    f"output={n} width={width}: {int((x != y).sum())} unequal scores, "
                    f"max abs={float((x - y).abs().nan_to_num().max())}"
                )


def run_case(
    requests, width, ratio, source, tokens, baseline=None, correctness_only=False
):
    inputs = make_case(requests, width, ratio, tokens)
    q, _, _, req_table, lengths, keys, page_size, _, _ = inputs
    variants = [
        ("row" if tokens % 2 else "pair", 1 if tokens % 2 else 2),
        ("request", -1),
    ]
    if baseline is not None:
        variants = [("old_" + name, group) for name, group in variants] + variants

    def score(name, group, use_fp8=True):
        if name.startswith("old_"):
            out = baseline(
                *inputs,
                use_fp8=use_fp8,
                return_block_scores=source,
                pair_queries=tokens % 2 == 0 and group != 1,
                request_queries=tokens if group == -1 else 0,
            )
        else:
            out = fp4_index_logits_paged(
                *inputs,
                use_fp8=use_fp8,
                return_block_scores=source,
                k_reuse_group_size=group,
                verify_tokens_per_request=tokens,
            )
        return out if source else (out,)

    outputs = [score(*variant) for variant in variants]
    check(outputs, lengths, width)
    graphs = []
    captured = []
    for variant in variants:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = score(*variant)
        graphs.append(graph)
        captured.append(out)

    if correctness_only:
        # Change graph inputs in place. Exercise C1/C2 tails, zero visibility,
        # unrepresentable Q in one row of a pair, and unsafe K scales.
        boundary_lengths = [0, 1, 63, 64, 127, 129, 255, 256]
        boundary_lengths = (boundary_lengths + [256] * tokens)[:tokens]
        boundary_lengths[-2:] = [127, 129]
        lengths[:tokens] = torch.tensor(boundary_lengths, device="cuda")
        q[tokens - 1, 0, 0] = 2**-12
        # One bad key is visible to only the second query of its pair.
        for position, exponent in [(127, 115), (128, 136)]:
            slot = int(req_table[0, position * ratio]) // ratio
            page, offset = divmod(slot, page_size)
            start = page_size * 64 + offset * 4
            keys[page, start : start + 4] = exponent
    for graph in graphs:
        graph.replay()
    check(captured, lengths, width)
    if correctness_only:
        check([score("row", 1, use_fp8=False), *captured], lengths, width)
        print(f"PASS replay and FP8 repair ratio={ratio} source={source}", flush=True)
        return

    samples = [[] for _ in graphs]
    order = list(range(len(graphs)))
    rng = random.Random(172)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(9):
        rng.shuffle(order)
        for i in order:
            for _ in range(3):
                graphs[i].replay()
            start.record()
            for _ in range(15):
                graphs[i].replay()
            stop.record()
            stop.synchronize()
            samples[i].append(start.elapsed_time(stop) * 1000 / 15)
    timings = " ".join(
        f"{name}={statistics.median(us):.1f}us"
        for (name, _), us in zip(variants, samples, strict=True)
    )
    print(
        f"{'source' if source else 'full'} ratio={ratio} requests={requests} "
        f"rows={requests * tokens} width={width} {timings}",
        flush=True,
    )


def load_scorer(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.fp4_index_logits_paged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--module")
    parser.add_argument("--verify-tokens", type=int, default=6)
    parser.add_argument("--baseline-module")
    args = parser.parse_args()
    if args.verify_tokens < 2:
        parser.error("--verify-tokens must be at least 2")
    if args.module:
        fp4_index_logits_paged = load_scorer(args.module, "fp4_indexer_trial")
    baseline = (
        load_scorer(args.baseline_module, "fp4_indexer_baseline")
        if args.baseline_module
        else None
    )
    torch.cuda.set_device(args.device)
    torch.manual_seed(172)
    props = torch.cuda.get_device_properties(args.device)
    print(
        f"{props.name} {props.multi_processor_count} SMs, torch {torch.__version__}",
        flush=True,
    )
    for ratio, source in [(1, True), (2, False)]:
        run_case(
            6, 512, ratio, source, args.verify_tokens, baseline, correctness_only=True
        )
        for requests in [6, 16] if args.quick else [6, 8, 10, 12, 14, 16]:
            for width in [32768] if args.quick else [16384, 32768, 65536]:
                run_case(requests, width, ratio, source, args.verify_tokens, baseline)
