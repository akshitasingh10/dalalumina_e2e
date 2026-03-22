"""Audit graph: Anticipation → Observation → Reconciliation + heuristic root-cause mapping."""

from __future__ import annotations

import os
from typing import Literal

import structlog
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from engine.vision import DEFAULT_LUMEN_MODEL, _gemini_generate

log = structlog.get_logger(__name__)

DriftKind = Literal["ok", "side_effect", "broken_logic", "architectural_drift"]
RiskLevel = Literal["low", "medium", "high", "critical"]


class AnticipatedComponent(BaseModel):
    component: str
    expected_change: str


class AnticipationResult(BaseModel):
    """Step 1 — from git diff only."""

    components: list[AnticipatedComponent] = Field(
        description="UI-facing components or regions implied by the diff.",
        default_factory=list,
    )
    reasoning_trace: str = Field(
        default="",
        description="Short chain-of-thought: which files/hunks drive which UI expectations.",
    )


class ObservedRegion(BaseModel):
    label: str
    description: str
    box_2d: list[float] = Field(
        description="Normalized box [ymin, xmin, ymax, xmax] with values 0–1000.",
        min_length=4,
        max_length=4,
    )


class ObservationResult(BaseModel):
    """Step 2 — dual screenshots."""

    regions: list[ObservedRegion] = Field(default_factory=list)
    summary: str = ""


class ReconciliationRow(BaseModel):
    component: str
    expected_change: str
    actual_result: str
    risk_level: RiskLevel
    drift: DriftKind
    notes: str = ""


class ReconciliationResult(BaseModel):
    """Step 3 — compare anticipation vs observation."""

    rows: list[ReconciliationRow] = Field(default_factory=list)
    critical_drifts: list[str] = Field(default_factory=list)


class RootCauseHypothesis(BaseModel):
    """Heuristic evaluator: map a visual anomaly to likely source lines."""

    suggested_path: str = Field(description="Best-guess file path from the diff, or unknown.")
    line_hint: str = Field(
        default="",
        description="Line number or range if inferable from diff hunks (e.g. '42' or '120-135').",
    )
    code_snippet: str = Field(default="", description="Short excerpt from the diff implicated.")
    css_or_selector_hint: str = Field(
        default="",
        description="CSS property, class, or JSX attribute likely responsible.",
    )
    rationale: str = ""


class ImpactRow(BaseModel):
    """CLI impact table row."""

    component: str
    expected_change: str
    actual_result: str
    risk_level: RiskLevel
    drift: DriftKind = "ok"
    root_cause: RootCauseHypothesis | None = None


class AuditReport(BaseModel):
    anticipation: AnticipationResult
    observation: ObservationResult
    reconciliation: ReconciliationResult
    impact_rows: list[ImpactRow] = Field(default_factory=list)
    annotated_current_png: bytes | None = None


class AuditGraph:
    """Three-step reasoning loop plus optional heuristic root-cause attachment."""

    def __init__(self, client: genai.Client, *, model: str | None = None) -> None:
        self._client = client
        self._model = model or os.environ.get("LUMEN_GEMINI_MODEL", DEFAULT_LUMEN_MODEL)

    def run(
        self,
        baseline_screenshot: bytes,
        current_screenshot: bytes,
        git_diff: str,
        *,
        draw_annotated_report: bool = True,
    ) -> AuditReport:
        log.info("audit_step", step="anticipation")
        anticipation = self._anticipation(git_diff)

        log.info("audit_step", step="observation")
        observation = self._observation(baseline_screenshot, current_screenshot)

        log.info("audit_step", step="reconciliation")
        reconciliation = self._reconciliation(anticipation, observation, git_diff)

        impact_rows: list[ImpactRow] = []
        for row in reconciliation.rows:
            rc: RootCauseHypothesis | None = None
            if row.drift in ("side_effect", "broken_logic", "architectural_drift"):
                log.info("heuristic_evaluator", component=row.component)
                rc = self._heuristic_evaluator(
                    git_diff=git_diff,
                    row=row,
                    observation=observation,
                )
            impact_rows.append(
                ImpactRow(
                    component=row.component,
                    expected_change=row.expected_change,
                    actual_result=row.actual_result,
                    risk_level=row.risk_level,
                    drift=row.drift,
                    root_cause=rc,
                )
            )

        annotated: bytes | None = None
        if draw_annotated_report and observation.regions:
            from utils.image_processing import draw_boxes_on_image

            boxes: list[tuple[list[float], str]] = [
                (r.box_2d, r.label) for r in observation.regions
            ]
            annotated = draw_boxes_on_image(current_screenshot, boxes)

        return AuditReport(
            anticipation=anticipation,
            observation=observation,
            reconciliation=reconciliation,
            impact_rows=impact_rows,
            annotated_current_png=annotated,
        )

    def _anticipation(self, git_diff: str) -> AnticipationResult:
        prompt = (
            "You only see a git diff. Step ANTICIPATION: predict which UI components or "
            "screens should change. Do not assume you saw the app. Output structured components "
            "with expected_change each."
            f"\n\n```diff\n{git_diff[:120_000]}\n```"
        )
        out = _gemini_generate(
            self._client,
            model=self._model,
            parts=[types.Part.from_text(text=prompt)],
            response_schema=AnticipationResult,
            system_instruction=(
                "You are Lumen's anticipation engine. Reason about JSX/CSS/routing/config changes "
                "and name user-visible components."
            ),
        )
        assert isinstance(out, AnticipationResult)
        return out

    def _observation(self, baseline_png: bytes, current_png: bytes) -> ObservationResult:
        prompt = (
            "Image 1 = BASELINE UI. Image 2 = CURRENT UI.\n"
            "Step OBSERVATION: locate visible differences. For each distinct change, output "
            "label, description, and box_2d as normalized [ymin, xmin, ymax, xmax] with "
            "coordinates from 0 to 1000 (inclusive), matching the visible bounding box of that "
            "change on the CURRENT image."
        )
        parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=baseline_png, mime_type="image/png"),
            types.Part.from_bytes(data=current_png, mime_type="image/png"),
        ]
        out = _gemini_generate(
            self._client,
            model=self._model,
            parts=parts,
            response_schema=ObservationResult,
            system_instruction="You are Lumen's vision module. Be spatially precise.",
        )
        assert isinstance(out, ObservationResult)
        return out

    def _reconciliation(
        self,
        anticipation: AnticipationResult,
        observation: ObservationResult,
        git_diff: str,
    ) -> ReconciliationResult:
        prompt = (
            "Step RECONCILIATION.\n"
            "Compare ANTICIPATION (from diff-only reasoning) to OBSERVATION (from screenshots).\n"
            "- If the diff predicted a change that does not appear in the UI: drift=broken_logic.\n"
            "- If the UI changed in a way not explained by the diff: drift=side_effect.\n"
            "- If both mismatch in a systemic way (many components): drift=architectural_drift.\n"
            "- Aligned outcomes: drift=ok.\n"
            f"ANTICIPATION JSON:\n{anticipation.model_dump_json()}\n\n"
            f"OBSERVATION JSON:\n{observation.model_dump_json()}\n\n"
            "Use the diff only as extra grounding for ambiguity:\n"
            f"```diff\n{git_diff[:40_000]}\n```"
        )
        out = _gemini_generate(
            self._client,
            model=self._model,
            parts=[types.Part.from_text(text=prompt)],
            response_schema=ReconciliationResult,
            system_instruction=(
                "You are Lumen's reconciliation engine. Populate one table row per anticipated "
                "component where possible; add rows for major unanticipated visual changes."
            ),
        )
        assert isinstance(out, ReconciliationResult)
        return out

    def _heuristic_evaluator(
        self,
        *,
        git_diff: str,
        row: ReconciliationRow,
        observation: ObservationResult,
    ) -> RootCauseHypothesis:
        obs_excerpt = observation.model_dump_json()[:8_000]
        prompt = (
            "A visual audit flagged an anomaly. Map it to the most likely source in the diff "
            "(CSS, JSX, style objects, layout components).\n"
            f"Impact row:\n{row.model_dump_json()}\n\n"
            f"Observation context (truncated):\n{obs_excerpt}\n\n"
            f"Git diff (truncated):\n{git_diff[:80_000]}"
        )
        out = _gemini_generate(
            self._client,
            model=self._model,
            parts=[types.Part.from_text(text=prompt)],
            response_schema=RootCauseHypothesis,
            system_instruction=(
                "You are Lumen's Heuristic Evaluator. Prefer concrete file paths and line hints "
                "present in the diff hunks. If uncertain, say so in rationale."
            ),
        )
        assert isinstance(out, RootCauseHypothesis)
        return out
