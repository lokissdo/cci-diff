import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def records():
    return [
        {
            "step": 2,
            "timestep": 800,
            "progress": 0.2,
            "target": {
                "target_probability": 0.3,
                "required_probability": 0.8,
                "activation": 1.0,
                "gradient_norm": 0.1,
            },
            "constraints": {
                "identity": {
                    "lambda_after": 0.1,
                    "residual": 0.2,
                    "gradient_norm": 0.03,
                },
                "outside_locality": {
                    "lambda_after": 0.0,
                    "residual": -0.4,
                    "gradient_norm": 0.0,
                },
            },
            "update": {
                "eta": 0.1,
                "norm": 0.12,
                "target_constraint_cosine": -0.2,
            },
        },
        {
            "step": 4,
            "timestep": 600,
            "progress": 0.4,
            "target": {
                "target_probability": 0.5,
                "required_probability": 0.8,
                "activation": 0.7,
                "gradient_norm": 0.2,
            },
            "constraints": {
                "identity": {
                    "lambda_after": 0.2,
                    "residual": 0.1,
                    "gradient_norm": 0.04,
                }
            },
            "update": {
                "eta": 0.2,
                "norm": 0.1,
                "target_constraint_cosine": None,
            },
        },
    ]


class TestPlotCCITrace(unittest.TestCase):
    def test_trace_rows_have_stable_constraint_columns(self):
        from scripts.plot_cci_trace import trace_rows

        rows = trace_rows(records())

        self.assertEqual(list(rows[0]), list(rows[1]))
        self.assertIn("target_probability", rows[0])
        self.assertIn("update_norm", rows[0])
        self.assertIn("identity.lambda", rows[0])
        self.assertIn("identity.residual", rows[0])
        self.assertIn("outside_locality.lambda", rows[0])
        self.assertIn("outside_locality.residual", rows[0])
        self.assertIsNone(rows[1]["outside_locality.lambda"])

    def test_csv_export_does_not_import_matplotlib(self):
        from scripts.plot_cci_trace import write_trace_csv

        before = set(sys.modules)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.csv"
            write_trace_csv(records(), path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 2)
        self.assertNotIn("matplotlib.pyplot", set(sys.modules) - before)

    def test_empty_trace_cannot_be_exported(self):
        from scripts.plot_cci_trace import write_trace_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "empty CCI trace"):
                write_trace_csv([], Path(tmpdir) / "trace.csv")

    def test_png_cli_uses_a_headless_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            trace = root / "trace.jsonl"
            trace.write_text(
                "\n".join(json.dumps(record) for record in records()) + "\n",
                encoding="utf-8",
            )
            png = root / "trace.png"
            environment = dict(os.environ)
            environment.pop("MPLBACKEND", None)
            environment["MPLCONFIGDIR"] = str(root / "mpl")
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/plot_cci_trace.py",
                    "--trace",
                    str(trace),
                    "--csv",
                    str(root / "trace.csv"),
                    "--png",
                    str(png),
                ],
                env=environment,
                check=False,
            )
            self.assertEqual(completed.returncode, 0)
            self.assertTrue(png.is_file())


if __name__ == "__main__":
    unittest.main()
