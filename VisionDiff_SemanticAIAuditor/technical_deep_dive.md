# Technical deep dive: AI / ML in Lumen

This project does **not** train custom models. The “AI” is **prompted inference** on a **multimodal large language model (VLM)**—by default **Gemini** via the `google-genai` client—with **structured JSON outputs** validated against **Pydantic** schemas. Understanding it is mostly about **inputs, schemas, and multi-step reasoning**, not about gradients or datasets.

## Stack at a glance

| Piece | Technology | Role |
|-------|------------|------|
| Model API | `google.genai` (`genai.Client`) | Text + image in, JSON (or text) out. |
| Default model | `gemini-2.5-flash` (overridable with `LUMEN_GEMINI_MODEL`) | Fast multimodal model suited to vision + structured extraction. |
| Structured output | `GenerateContentConfig` with `response_mime_type="application/json"` and `response_schema=...` | Constrains the model to return instances of your Pydantic models (`resp.parsed`). |
| Vision + browser | Playwright (Chromium) | Deterministic pixels fed to the model as PNG bytes. |
| Post-processing | Pillow | Maps model box coordinates to pixel rectangles on the screenshot. |
| Resilience | `tenacity` on `_gemini_generate` | Retries transient API failures (exponential backoff). |

So “ML” here means: **a pretrained VLM doing conditional generation** under schema and system-instruction constraints.

## How `_gemini_generate` works

In `engine/vision.py`, `_gemini_generate` builds a `GenerateContentConfig` when you pass a `response_schema`:

- **`system_instruction`** — Steady “persona” and task framing (anticipation vs vision vs reconciliation).
- **`response_mime_type="application/json"`** — Asks the API for JSON.
- **`response_schema`** — Supplies a Pydantic model so the API knows the target shape; the client returns `resp.parsed` as that type when successful.

If parsing fails, the code raises with a hint—there is no local fallback parser beyond what the client provides.

## AuditGraph: a three-node “agent graph”

`engine/auditor.py` implements a fixed pipeline (not a learned policy):

### 1. Anticipation (text-only)

- **Input:** Truncated git diff (up to ~120k chars in code).
- **Output:** `AnticipationResult` — list of `{ component, expected_change }` plus a short `reasoning_trace`.
- **Idea:** The model must **not** pretend it saw the app; it only infers UI impact from code changes.

This is **chain-of-thought–style reasoning** in natural language, but the **contract** the rest of the system cares about is the structured `components` list.

### 2. Observation (multimodal)

- **Input:** Two images (baseline PNG, current PNG) + instructions.
- **Output:** `ObservationResult` — `regions[]` each with `label`, `description`, and **`box_2d`**.

**Spatial grounding:** Boxes use Gemini-style **normalized coordinates** `[ymin, xmin, ymax, xmax]` on a **0–1000** scale (inclusive), aligned to the **current** image. The model is asked to be “spatially precise”; quality depends on the VLM’s localization ability, not on a separate detector.

### 3. Reconciliation (text + structured prior)

- **Input:** JSON serializations of anticipation and observation, plus a shorter diff excerpt for grounding.
- **Output:** `ReconciliationResult` — table rows with `risk_level`, `drift` kind (`ok` | `side_effect` | `broken_logic` | `architectural_drift`), and narrative fields.

The prompt encodes **explicit rules** mapping mismatch patterns to drift labels—for example, expected UI change missing from screenshots → `broken_logic`; unexpected visual change not explained by diff → `side_effect`.

### 4. Heuristic evaluator (conditional, text-only)

For rows where drift is not `ok`, another call produces `RootCauseHypothesis`: guessed path, line hint, snippet, CSS/selector hint, rationale. This is **RAG-like** in spirit (grounding on the diff) but implemented as **long-context prompting**, not vector search.

### 5. Annotation (non-ML)

`utils/image_processing.py` converts `box_2d` to pixel rectangles and draws overlays—**deterministic** given the model’s box outputs.

## Visual Chain-of-Thought (V-CoT)

In `AutonomousObserver.visual_chain_of_thought`:

1. **Pass 1 (one multimodal call):** Returns structured `scan` + `hypothesis` — inventory the UI, then state what *should* be true given `task_context` and where bugs might hide.
2. **Pass 2 (second multimodal call):** Returns `verify` — compare screenshot to the hypothesis; set `matches_mental_model`, explain discrepancies, optional `corrected_assessment`, and a **self-reported** `confidence_0_to_1`.
3. **Pass 3 (optional):** If there was a mismatch but no corrected assessment, a short nudge asks for a concrete correction.

This is **explicit multi-step reasoning** with **two (or three) forward passes** over the same image. It can reduce incoherence versus a single blob of text, at the cost of latency and tokens. It is **not** guaranteed calibration: `confidence_0_to_1` is model-generated text constrained to a float, not a calibrated probability.

## What can go wrong (model behavior)

- **Hallucinated regions or boxes** — The VLM may mis-localize or invent UI elements; overlays will reflect those errors faithfully.
- **Diff/screenshot skew** — If screenshots don’t match the diff (wrong branch, stale build, different viewport), reconciliation will still run and may mislabel drift.
- **Truncation** — Large diffs are truncated; anticipation and reconciliation may miss relevant hunks.
- **Schema pressure** — Strong JSON schemas help shape outputs but don’t eliminate reasoning errors inside the fields.

## Summary

Lumen is a **semantic visual diff auditor**: it orchestrates **Gemini** for (1) code→UI expectation, (2) pixel-level change description + boxes, and (3) consistency checking, with optional **root-cause narration** and a separate **V-CoT** path for single screenshots. The “learning” is entirely in the **foundation model**; this repo supplies **data plumbing, prompts, schemas, and deterministic rendering**.
