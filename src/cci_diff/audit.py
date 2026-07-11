"""Audit output helpers for CCI-Diff experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from typing import Iterable

from cci_diff.metrics import bias_audit_matrix


@dataclass(frozen=True)
class AuditRow:
    """Per-image audit result for one concept intervention."""

    image_id: str
    intervention: str
    classifier: str
    before_score: float
    after_score: float
    target_success: bool
    preservation_score: float
    leakage: float
    purity: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable row with score delta included."""

        payload = asdict(self)
        payload["score_delta"] = round(self.after_score - self.before_score, 12)
        return payload


def summarize_audit_rows(rows: Iterable[AuditRow]) -> dict[str, object]:
    """Aggregate audit rows into paper-facing metrics."""

    row_list = list(rows)
    if not row_list:
        return {
            "matrix": {},
            "mean_preservation_score": 0.0,
            "mean_leakage": 0.0,
            "mean_purity": 0.0,
            "target_success_rate": 0.0,
        }

    matrix_records = [
        {
            "intervention": row.intervention,
            "classifier": row.classifier,
            "before": row.before_score,
            "after": row.after_score,
        }
        for row in row_list
    ]

    return {
        "matrix": bias_audit_matrix(matrix_records),
        "mean_preservation_score": mean(row.preservation_score for row in row_list),
        "mean_leakage": mean(row.leakage for row in row_list),
        "mean_purity": mean(row.purity for row in row_list),
        "target_success_rate": mean(1.0 if row.target_success else 0.0 for row in row_list),
    }
