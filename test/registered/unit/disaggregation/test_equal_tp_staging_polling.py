"""CPU contracts for equal-TP staging registration and completion."""

import ast
import pathlib
import unittest
from enum import IntEnum
from types import SimpleNamespace as NS

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
ROOT = pathlib.Path(__file__).resolve().parents[4] / "python/sglang/srt/disaggregation"


class TestEqualTPStagingPolling(CustomTestCase):
    def test_registered_equal_tp_waits_for_scatter(self):
        tree = ast.parse((ROOT / "utils.py").read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "poll_and_all_reduce_with_staging"
        )
        node.returns = None
        for arg in node.args.args:
            arg.annotation = None

        class Poll(IntEnum):
            Failed = 0
            Transferring = 3
            Success = 4

        calls = []
        state = {"done": False, "failed": False}
        handler = NS(
            is_staging_room=lambda room: room == 7,
            is_done=lambda req: state["done"],
            is_failed=lambda req: state["failed"],
            advance_scatter=lambda req: calls.append(req),
        )
        req = NS(req=NS(bootstrap_room=7), kv_receiver=NS(require_staging=False))
        env = dict(
            KVPoll=Poll,
            _poll_with_failure_injection=lambda receivers: [int(Poll.Success)],
            _all_reduce_polls=lambda polls, group: polls,
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                "actual_polling",
                "exec",
            ),
            env,
        )
        poll = env[node.name]
        self.assertEqual(poll([req], handler, None), [int(Poll.Transferring)])
        self.assertEqual(calls, [req])
        state["done"] = True
        self.assertEqual(poll([req], handler, None), [int(Poll.Success)])
        state["failed"] = True
        self.assertEqual(poll([req], handler, None), [int(Poll.Failed)])

    def test_registration_excludes_host_and_fake_transfers(self):
        tree = ast.parse((ROOT / "decode.py").read_text())
        condition = next(
            n.test
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and "metadata_kwargs.get('destination')" in ast.unparse(n.test)
            and "require_staging" in ast.unparse(n.test)
        )
        util = ast.parse((ROOT / "utils.py").read_text())
        fake = next(
            n
            for n in util.body
            if isinstance(n, ast.FunctionDef) and n.name == "_is_fake_transfer"
        )
        fake.args.args[0].annotation = None
        env = dict(
            FAKE_BOOTSTRAP_HOST="fake",
            get_disagg=lambda: NS(disaggregation_transfer_backend="mooncake"),
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[fake], type_ignores=[])),
                "fake_predicate",
                "exec",
            ),
            env,
        )
        env.update(
            self=NS(is_mla_backend=False, transfer_queue=NS(enable_staging=True)),
            decode_req=NS(
                req=NS(bootstrap_host="real"), kv_receiver=NS(require_staging=False)
            ),
            metadata_kwargs={},
            KVTransferDestination=NS(HOST="host"),
        )
        code = compile(ast.Expression(condition), "actual_registration", "eval")
        self.assertTrue(eval(code, env))
        env["metadata_kwargs"]["destination"] = "host"
        self.assertFalse(eval(code, env))
        env["metadata_kwargs"].clear()
        env["decode_req"].req.bootstrap_host = "fake"
        env["decode_req"].kv_receiver.require_staging = True
        self.assertFalse(eval(code, env))


if __name__ == "__main__":
    unittest.main()
