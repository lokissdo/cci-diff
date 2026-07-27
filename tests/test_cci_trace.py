import json
import tempfile
import unittest
from pathlib import Path


def trace_record(step):
    return {
        "step": step,
        "timestep": 500 - step,
        "progress": step / 10,
        "target": {"activation": 1.0},
        "constraints": {},
        "update": {},
    }


class TestCCITrace(unittest.TestCase):
    def test_writer_truncates_then_appends_deterministic_json_lines(self):
        from cci_diff.cci_trace import JSONLTraceWriter, load_cci_trace

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            path.write_text("stale\n", encoding="utf-8")
            writer = JSONLTraceWriter(path)
            writer.write(trace_record(2))
            writer.write(trace_record(4))
            records = load_cci_trace(path)
            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual([record["step"] for record in records], [2, 4])
        self.assertEqual(
            lines[0],
            json.dumps(records[0], sort_keys=True, separators=(",", ":")),
        )

    def test_validation_rejects_missing_fields_and_non_increasing_steps(self):
        from cci_diff.cci_trace import validate_cci_trace

        with self.assertRaisesRegex(ValueError, "required fields"):
            validate_cci_trace([{"step": 0}])
        record = trace_record(2)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_cci_trace([record, dict(record)])

    def test_writer_rejects_nonfinite_json(self):
        from cci_diff.cci_trace import JSONLTraceWriter

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = JSONLTraceWriter(Path(tmpdir) / "trace.jsonl")
            record = trace_record(2)
            record["progress"] = float("nan")
            with self.assertRaises(ValueError):
                writer.write(record)


if __name__ == "__main__":
    unittest.main()
