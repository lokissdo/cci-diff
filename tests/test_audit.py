import unittest

from cci_diff.audit import AuditRow, summarize_audit_rows


class TestAuditRows(unittest.TestCase):
    def test_audit_row_is_json_serializable(self):
        row = AuditRow(
            image_id="12.jpg",
            intervention="makeup",
            classifier="smile",
            before_score=0.20,
            after_score=0.55,
            target_success=True,
            preservation_score=0.92,
            leakage=0.15,
            purity=0.80,
        )

        payload = row.to_dict()

        self.assertEqual(payload["image_id"], "12.jpg")
        self.assertEqual(payload["score_delta"], 0.35)
        self.assertTrue(payload["target_success"])

    def test_summarize_audit_rows_returns_matrix_and_means(self):
        rows = [
            AuditRow("1.jpg", "makeup", "smile", 0.10, 0.40, True, 0.90, 0.10, 0.80),
            AuditRow("2.jpg", "makeup", "smile", 0.20, 0.50, True, 0.80, 0.20, 0.70),
        ]

        summary = summarize_audit_rows(rows)

        self.assertAlmostEqual(summary["matrix"]["makeup"]["smile"], 0.30)
        self.assertAlmostEqual(summary["mean_preservation_score"], 0.85)
        self.assertAlmostEqual(summary["mean_leakage"], 0.15)
        self.assertAlmostEqual(summary["target_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
