import hashlib
import math
import os
import pickle
import random
import uuid
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
    compute_random_lens,
    gen_prompt,
    get_available_tokens,
)

GSP_SHARD_CACHE_VERSION = 2
GSP_SHARD_COUNT_ENV = "SGLANG_BENCH_NUM_SHARDS"
GSP_SHARD_INDEX_ENV = "SGLANG_BENCH_SHARD_INDEX"
GSP_EXPECTED_REQUESTS_ENV = "SGLANG_BENCH_EXPECTED_NUM_REQUESTS"


def _get_multiprocess_shard() -> Optional[Tuple[int, int]]:
    shard_count_raw = os.getenv(GSP_SHARD_COUNT_ENV)
    shard_index_raw = os.getenv(GSP_SHARD_INDEX_ENV)
    if shard_count_raw is None and shard_index_raw is None:
        return None
    if shard_count_raw is None or shard_index_raw is None:
        raise ValueError(
            f"{GSP_SHARD_COUNT_ENV} and {GSP_SHARD_INDEX_ENV} must be set together"
        )

    shard_count = int(shard_count_raw)
    shard_index = int(shard_index_raw)
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError(
            f"Invalid GSP benchmark shard: index={shard_index}, count={shard_count}"
        )
    return shard_count, shard_index


def _zipf_group_probs(num_groups: int, alpha: float) -> np.ndarray:
    """Rank-based Zipf probability vector with rank starting at 1.

    weight(rank)      = 1 / rank ** alpha       (rank in 1..num_groups)
    probability(rank) = weight(rank) / sum_over_all_ranks(weight)

    The returned array has length num_groups; element i corresponds to
    group index i (rank i + 1), so group 0 is the hottest.
    """
    if num_groups <= 0:
        raise ValueError(f"num_groups must be > 0, got {num_groups}")
    ranks = np.arange(1, num_groups + 1, dtype=np.float64)
    weights = 1.0 / (ranks**alpha)
    return weights / weights.sum()


@dataclass
class GeneratedSharedPrefixDataset(BaseDataset):
    num_groups: int
    prompts_per_group: int
    system_prompt_len: int
    question_len: int
    output_len: int
    range_ratio: float
    seed: int
    fast_prepare: bool
    send_routing_key: bool
    num_turns: int
    ordered: bool
    group_distribution: str = "uniform"
    zipf_alpha: Optional[float] = None

    @classmethod
    def from_args(cls, args: Namespace) -> "GeneratedSharedPrefixDataset":
        assert not getattr(args, "tokenize_prompt", False)
        group_distribution = getattr(args, "gsp_group_distribution", "uniform")
        zipf_alpha = getattr(args, "gsp_zipf_alpha", None)

        # Defensive validation for in-process callers that construct a
        # Namespace by hand and bypass the argparse boundary in
        # serving.py. The CLI hook enforces the same rules first.
        if group_distribution not in ("uniform", "zipf"):
            raise ValueError(
                f"--gsp-group-distribution must be 'uniform' or 'zipf', "
                f"got {group_distribution!r}"
            )
        if group_distribution == "zipf":
            if zipf_alpha is None:
                raise ValueError(
                    "--gsp-group-distribution=zipf requires --gsp-zipf-alpha "
                    "(a finite float > 0)"
                )
            if not math.isfinite(zipf_alpha) or zipf_alpha <= 0:
                raise ValueError(
                    f"--gsp-zipf-alpha must be a finite float > 0, got {zipf_alpha!r}"
                )
        elif zipf_alpha is not None:
            raise ValueError(
                "--gsp-zipf-alpha is only meaningful with "
                "--gsp-group-distribution=zipf; remove --gsp-zipf-alpha "
                "or set --gsp-group-distribution=zipf"
            )

        return cls(
            num_groups=args.gsp_num_groups,
            prompts_per_group=args.gsp_prompts_per_group,
            system_prompt_len=args.gsp_system_prompt_len,
            question_len=args.gsp_question_len,
            output_len=args.gsp_output_len,
            range_ratio=getattr(args, "gsp_range_ratio", 1.0),
            seed=args.seed,
            fast_prepare=getattr(args, "gsp_fast_prepare", False),
            send_routing_key=getattr(args, "gsp_send_routing_key", False),
            num_turns=getattr(args, "gsp_num_turns", 1),
            ordered=getattr(args, "gsp_ordered", False),
            group_distribution=group_distribution,
            zipf_alpha=zipf_alpha,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        shard = _get_multiprocess_shard()
        if shard is not None:
            expected_count_raw = os.getenv(GSP_EXPECTED_REQUESTS_ENV)
            actual_count = self.num_groups * self.prompts_per_group
            if expected_count_raw is None:
                raise ValueError(
                    f"{GSP_EXPECTED_REQUESTS_ENV} is required for sharded GSP data"
                )
            if actual_count != int(expected_count_raw):
                raise ValueError(
                    "GSP dataset size does not match the multiprocess launcher: "
                    f"configured {actual_count}, expected {expected_count_raw}"
                )
        return sample_generated_shared_prefix_requests(
            num_groups=self.num_groups,
            prompts_per_group=self.prompts_per_group,
            system_prompt_len=self.system_prompt_len,
            question_len=self.question_len,
            output_len=self.output_len,
            range_ratio=self.range_ratio,
            tokenizer=tokenizer,
            seed=self.seed,
            send_routing_key=self.send_routing_key,
            num_turns=self.num_turns,
            fast_prepare=self.fast_prepare,
            ordered=self.ordered,
            group_distribution=self.group_distribution,
            zipf_alpha=self.zipf_alpha,
            shard_count=shard[0] if shard is not None else None,
            shard_index=shard[1] if shard is not None else None,
        )


def get_gen_prefix_cache_path(
    seed: int,
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    tokenizer,
    group_distribution: str = "uniform",
    zipf_alpha: Optional[float] = None,
    shard_count: Optional[int] = None,
    shard_index: Optional[int] = None,
    fast_prepare: bool = False,
    ordered: bool = False,
):
    """Create cache directory under ~/.cache/sglang/benchmark.

    The uniform-mode filename is preserved exactly as before so existing
    on-disk caches remain valid. Non-default sampling modes get an extra
    suffix encoding the parameters that affect the cached payload.
    """
    cache_dir = Path.home() / ".cache" / "sglang" / "benchmark"

    suffix = ""
    if group_distribution != "uniform":
        suffix = f"_{group_distribution}_{zipf_alpha}"
    if (shard_count is None) != (shard_index is None):
        raise ValueError("shard_count and shard_index must be set together")
    if shard_count is not None:
        if shard_count <= 0 or not 0 <= shard_index < shard_count:
            raise ValueError(
                f"Invalid GSP cache shard: index={shard_index}, count={shard_count}"
            )
        suffix += (
            f"_mpv{GSP_SHARD_CACHE_VERSION}"
            f"_shard_{shard_index}_of_{shard_count}"
            f"_fast_{int(fast_prepare)}"
            f"_ordered_{int(ordered)}"
        )

    cache_key = (
        f"gen_shared_prefix_{seed}_{num_groups}_{prompts_per_group}_"
        f"{system_prompt_len}_{question_len}_{output_len}{suffix}_"
        f"{tokenizer.__class__.__name__}.pkl"
    )
    return cache_dir / cache_key


def _derive_seed(seed: int, *parts: object) -> int:
    # Component seeds make every global slot reproducible without generating
    # and discarding the slots owned by other worker processes.
    material = ":".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(
        hashlib.blake2b(material, digest_size=8).digest(), byteorder="big"
    )


def _compute_deterministic_lens(
    full_len: int, range_ratio: float, num: int, seed: int
) -> List[int]:
    if full_len <= 0:
        return [0] * num
    rng = np.random.default_rng(seed)
    return rng.integers(
        max(int(full_len * range_ratio), 1),
        full_len + 1,
        size=num,
    ).tolist()


def _gen_prompt_with_seed(
    tokenizer: PreTrainedTokenizerBase, token_num: int, seed: int
) -> str:
    available_tokens = get_available_tokens(tokenizer)
    selected_tokens = random.Random(seed).choices(available_tokens, k=token_num)
    return tokenizer.decode(selected_tokens)


def _sample_generated_shared_prefix_shard(
    *,
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    seed: int,
    send_routing_key: bool,
    num_turns: int,
    fast_prepare: bool,
    ordered: bool,
    group_distribution: str,
    zipf_alpha: Optional[float],
    shard_count: int,
    shard_index: int,
) -> List[DatasetRow]:
    total_slots = num_groups * prompts_per_group
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError(
            f"Invalid GSP benchmark shard: index={shard_index}, count={shard_count}"
        )

    cache_path = get_gen_prefix_cache_path(
        seed,
        num_groups,
        prompts_per_group,
        system_prompt_len,
        question_len,
        output_len,
        tokenizer,
        group_distribution=group_distribution,
        zipf_alpha=zipf_alpha,
        shard_count=shard_count,
        shard_index=shard_index,
        fast_prepare=fast_prepare,
        ordered=ordered,
    )
    should_cache = range_ratio == 1 and not send_routing_key and num_turns == 1
    if should_cache and cache_path.exists():
        print(f"\nLoading cached generated input shard from {cache_path}")
        with open(cache_path, "rb") as f:
            rows = pickle.load(f)
        expected_shard_size = len(range(shard_index, total_slots, shard_count))
        if len(rows) != expected_shard_size:
            raise ValueError(
                f"Cached GSP shard {shard_index}/{shard_count} has {len(rows)} "
                f"rows, expected {expected_shard_size}: {cache_path}"
            )
        return rows

    if not should_cache:
        print(f"\nCache bypassed ({range_ratio=}, {send_routing_key=}, {num_turns=})")

    # Shuffle globally before striding so interleaving all worker shards
    # reconstructs the exact shard_count=1 request sequence.
    global_slots = list(range(total_slots))
    if not ordered:
        random.Random(_derive_seed(seed, "shuffle")).shuffle(global_slots)
    shard_slots = global_slots[shard_index::shard_count]
    print(
        "\nGenerating new input shard... "
        f"(shard={shard_index}/{shard_count}, shard_size={len(shard_slots)}, "
        f"total_slots={total_slots}, {num_groups=}, {prompts_per_group=}, "
        f"{system_prompt_len=}, {question_len=}, {output_len=}, {range_ratio=}, "
        f"{num_turns=}, {group_distribution=}, {zipf_alpha=})"
    )

    system_prompt_lens = _compute_deterministic_lens(
        system_prompt_len,
        range_ratio,
        num_groups,
        _derive_seed(seed, "system_lens"),
    )
    question_lens = np.array(
        _compute_deterministic_lens(
            question_len,
            range_ratio,
            total_slots * num_turns,
            _derive_seed(seed, "question_lens"),
        )
    ).reshape(total_slots, num_turns)
    output_lens = _compute_deterministic_lens(
        output_len,
        range_ratio,
        total_slots,
        _derive_seed(seed, "output_lens"),
    )

    if group_distribution == "uniform":
        assignment = np.repeat(np.arange(num_groups), prompts_per_group)
    else:
        rng = np.random.default_rng(seed)
        probs = _zipf_group_probs(num_groups, zipf_alpha)
        assignment = rng.choice(num_groups, size=total_slots, replace=True, p=probs)

    # Every worker derives shared prefixes from the same group seed, while only
    # materializing groups and questions referenced by its own slots.
    used_groups = {int(assignment[slot_idx]) for slot_idx in shard_slots}
    system_prompts = {
        group_index: _gen_prompt_with_seed(
            tokenizer,
            system_prompt_lens[group_index],
            _derive_seed(seed, "system_prompt", group_index),
        )
        for group_index in used_groups
    }

    run_random_str = os.getenv("SGLANG_BENCH_GSP_RUN_ID") or uuid.uuid4().hex[:8]
    run_start_timestamp = os.getenv(
        "SGLANG_BENCH_GSP_RUN_TIMESTAMP"
    ) or datetime.now().strftime("%Y%m%d%H%M%S")

    input_requests = []
    total_input_tokens = 0
    total_output_tokens = 0
    for slot_idx in tqdm(
        shard_slots,
        desc=f"Generating shared-prefix shard {shard_index}/{shard_count}",
    ):
        sampled_group = int(assignment[slot_idx])
        turn_questions = [
            _gen_prompt_with_seed(
                tokenizer,
                int(question_lens[slot_idx, turn_index]),
                _derive_seed(seed, "question", slot_idx, turn_index),
            )
            for turn_index in range(num_turns)
        ]
        turn_prompts = [
            f"{system_prompts[sampled_group]}\n\n{turn_questions[0]}"
        ] + turn_questions[1:]
        full_prompt = turn_prompts[0] if num_turns == 1 else turn_prompts
        prompt_len = 1 if fast_prepare else len(tokenizer.encode(turn_prompts[0]))
        output_len_val = int(output_lens[slot_idx])
        routing_key = (
            f"{run_random_str}_{run_start_timestamp}_{sampled_group}"
            if send_routing_key
            else None
        )

        input_requests.append(
            DatasetRow(
                prompt=full_prompt,
                prompt_len=prompt_len,
                output_len=output_len_val,
                routing_key=routing_key,
            )
        )
        total_input_tokens += prompt_len
        total_output_tokens += output_len_val

    print("\nGenerated shared prefix shard statistics:")
    print(f"Shard: {shard_index}/{shard_count}")
    print(f"Shard prompts: {len(input_requests)} of {total_slots}")
    print(f"Number of groups represented: {len(used_groups)}")
    print(f"Number of turns: {num_turns}")
    print(f"Group distribution: {group_distribution}")
    if group_distribution == "zipf":
        print(f"Zipf alpha: {zipf_alpha}")
    if not fast_prepare:
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")

    if should_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching generated input shard to {cache_path}")
        temp_cache_path = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with open(temp_cache_path, "wb") as f:
                pickle.dump(input_requests, f)
            os.replace(temp_cache_path, cache_path)
        finally:
            temp_cache_path.unlink(missing_ok=True)

    return input_requests


def sample_generated_shared_prefix_requests(
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    seed: int,
    send_routing_key: bool = False,
    num_turns: int = 1,
    fast_prepare: bool = False,
    ordered: bool = False,
    group_distribution: str = "uniform",
    zipf_alpha: Optional[float] = None,
    shard_count: Optional[int] = None,
    shard_index: Optional[int] = None,
) -> List[DatasetRow]:
    """Generate benchmark requests with shared system prompts using random tokens and caching.

    When group_distribution is "uniform" (default), each group receives exactly
    prompts_per_group requests; behavior matches the legacy generator.

    When group_distribution is "zipf", each request's group is sampled by rank
    with probability 1/rank**zipf_alpha / sum_k(1/k**zipf_alpha); rank starts at
    1 and group index 0 is the hottest. Sampling uses an isolated
    numpy.random.default_rng(seed) so the shared question/system-prompt pool
    stays byte-identical to uniform mode for the same seed and other args.
    Zipf mode is cached on disk under a distinct key per (group_distribution,
    zipf_alpha) value.

    When shard_count and shard_index are set by the multiprocess launcher, only
    that worker's global slots are generated and cached. Combining all shards in
    stride order is identical to the shard_count=1 path.
    """
    if (shard_count is None) != (shard_index is None):
        raise ValueError("shard_count and shard_index must be set together")
    if shard_count is not None:
        return _sample_generated_shared_prefix_shard(
            num_groups=num_groups,
            prompts_per_group=prompts_per_group,
            system_prompt_len=system_prompt_len,
            question_len=question_len,
            output_len=output_len,
            range_ratio=range_ratio,
            tokenizer=tokenizer,
            seed=seed,
            send_routing_key=send_routing_key,
            num_turns=num_turns,
            fast_prepare=fast_prepare,
            ordered=ordered,
            group_distribution=group_distribution,
            zipf_alpha=zipf_alpha,
            shard_count=shard_count,
            shard_index=shard_index,
        )

    cache_path = get_gen_prefix_cache_path(
        seed,
        num_groups,
        prompts_per_group,
        system_prompt_len,
        question_len,
        output_len,
        tokenizer,
        group_distribution=group_distribution,
        zipf_alpha=zipf_alpha,
    )
    # range_ratio != 1 / num_turns > 1 perturb the payload but are not in the
    # cache key; send_routing_key embeds a per-run uuid + timestamp that is
    # meaningless to cache. Bypass for these pre-existing reasons only.
    should_cache = range_ratio == 1 and not send_routing_key and num_turns == 1

    if should_cache and cache_path.exists():
        print(f"\nLoading cached generated input data from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    if not should_cache:
        print(f"\nCache bypassed ({range_ratio=}, {send_routing_key=}, {num_turns=})")

    print(
        f"\nGenerating new input data... "
        f"({num_groups=}, {prompts_per_group}, {system_prompt_len=}, {question_len=}, {output_len=}, {range_ratio=}, {num_turns=}, {group_distribution=}, {zipf_alpha=})"
    )

    run_random_str = os.getenv("SGLANG_BENCH_GSP_RUN_ID") or uuid.uuid4().hex[:8]
    run_start_timestamp = os.getenv(
        "SGLANG_BENCH_GSP_RUN_TIMESTAMP"
    ) or datetime.now().strftime("%Y%m%d%H%M%S")

    system_prompt_lens = compute_random_lens(
        full_len=system_prompt_len,
        range_ratio=range_ratio,
        num=num_groups,
    )
    question_lens = np.array(
        compute_random_lens(
            full_len=question_len,
            range_ratio=range_ratio,
            num=num_groups * prompts_per_group * num_turns,
        )
    ).reshape(num_groups, prompts_per_group, num_turns)
    output_lens = np.array(
        compute_random_lens(
            full_len=output_len,
            range_ratio=range_ratio,
            num=num_groups * prompts_per_group,
        )
    ).reshape(num_groups, prompts_per_group)
    del system_prompt_len, question_len, output_len

    system_prompts = [
        gen_prompt(tokenizer, system_prompt_lens[i]) for i in range(num_groups)
    ]

    # shape: (num_groups, prompts_per_group, num_turns)
    questions = [
        [
            [
                gen_prompt(tokenizer, int(question_lens[g, p, t]))
                for t in range(num_turns)
            ]
            for p in range(prompts_per_group)
        ]
        for g in range(num_groups)
    ]

    # Per-slot group assignment. Uniform mode is the identity assignment
    # [0,0,...,1,1,...,N-1,N-1]; zipf mode samples from the rank distribution
    # using an isolated RNG so the module-level random / numpy.random state
    # that compute_random_lens / gen_prompt rely on is never perturbed -- this
    # keeps the system-prompt and question pool byte-identical to uniform mode
    # for the same seed and other args.
    total_slots = num_groups * prompts_per_group
    if group_distribution == "uniform":
        assignment = np.repeat(np.arange(num_groups), prompts_per_group)
    else:  # "zipf"
        rng = np.random.default_rng(seed)
        probs = _zipf_group_probs(num_groups, zipf_alpha)
        assignment = rng.choice(num_groups, size=total_slots, replace=True, p=probs)

    input_requests = []
    total_input_tokens = 0
    total_output_tokens = 0
    for slot_idx, sampled_g in enumerate(
        tqdm(assignment, desc="Generating shared-prefix prompts")
    ):
        # src_(g,p) walks the question pool in uniform-enumeration order, so
        # per-slot question text is reproducibly identical across modes.
        src_g, src_p = divmod(slot_idx, prompts_per_group)
        sampled_g = int(sampled_g)

        system_prompt = system_prompts[sampled_g]
        routing_key = (
            f"{run_random_str}_{run_start_timestamp}_{sampled_g}"
            if send_routing_key
            else None
        )
        turn_questions = questions[src_g][src_p]
        turn_prompts = [f"{system_prompt}\n\n{turn_questions[0]}"] + turn_questions[1:]
        full_prompt = turn_prompts[0] if num_turns == 1 else turn_prompts
        prompt_len = 1 if fast_prepare else len(tokenizer.encode(turn_prompts[0]))
        output_len_val = int(output_lens[src_g, src_p])

        input_requests.append(
            DatasetRow(
                prompt=full_prompt,
                prompt_len=prompt_len,
                output_len=output_len_val,
                routing_key=routing_key,
            )
        )
        total_input_tokens += prompt_len
        total_output_tokens += output_len_val

    if not ordered:
        random.shuffle(input_requests)

    print(f"\nGenerated shared prefix dataset statistics:")
    print(f"Number of groups: {num_groups}")
    print(f"Prompts per group: {prompts_per_group}")
    print(f"Number of turns: {num_turns}")
    print(f"Group distribution: {group_distribution}")
    if group_distribution == "zipf":
        print(f"Zipf alpha: {zipf_alpha}")
    print(f"Total prompts: {len(input_requests)}")
    if not fast_prepare:
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(
            f"Average system prompt length: {sum(len(tokenizer.encode(sp)) for sp in system_prompts) / len(system_prompts):.1f} tokens"
        )
        all_questions = [q for group in questions for conv in group for q in conv]
        print(
            f"Average question length: {sum(len(tokenizer.encode(q)) for q in all_questions) / len(all_questions):.1f} tokens\n"
        )

    if should_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching generated input data to {cache_path}")
        temp_cache_path = cache_path.with_name(
            f".{cache_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with open(temp_cache_path, "wb") as f:
                pickle.dump(input_requests, f)
            os.replace(temp_cache_path, cache_path)
        finally:
            temp_cache_path.unlink(missing_ok=True)

    return input_requests
