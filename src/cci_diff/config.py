"""JSON configuration loading for CCI-Diff runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cci_diff.spec import ConceptIntervention, GuidanceWeights


@dataclass(frozen=True)
class CCIConfig:
    """Loaded CCI-Diff run configuration."""

    intervention: ConceptIntervention
    weights: GuidanceWeights


def load_cci_config(path: str | Path) -> CCIConfig:
    """Load a CCI config JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    intervention = ConceptIntervention(
        target_concept=payload["target_concept"],
        desired_value=int(payload["desired_value"]),
        preserved_concepts=tuple(payload.get("preserved_concepts", ())),
        candidate_concepts=tuple(payload.get("candidate_concepts", ())),
        target_mask=payload.get("target_mask"),
        preserve_mask=payload.get("preserve_mask"),
    )
    weights = GuidanceWeights(**payload.get("weights", {}))
    return CCIConfig(intervention=intervention, weights=weights)
