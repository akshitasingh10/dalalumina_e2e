# VisionDiff / Semantic AI Auditor (Lumen)

This folder contains **Lumen**, a small Python CLI that audits how **code changes** line up with **what users actually see** in the browser. It combines a **git diff**, **before/after screenshots**, and **Google Gemini** (vision + text) to produce a structured “impact table” and optional annotated images.

## What it does

1. **Anticipation** — Reads only the diff and predicts which UI-facing areas should change.
2. **Observation** — Compares baseline and current screenshots and asks the model to name differences and place each on the **current** image using normalized bounding boxes.
3. **Reconciliation** — Aligns “what we expected from the diff” with “what we see,” labeling drift (for example OK vs side effect vs broken logic vs architectural drift) and risk.

For rows that show drift, an extra **heuristic evaluator** step suggests likely files, line hints, and CSS/selector clues grounded in the diff.

Separately, **observe** can open a URL with Playwright (with simple “try common dev ports” URL resolution), capture a PNG, and optionally run a **Visual Chain-of-Thought** pass on that screenshot.

## Requirements

- Python 3.11+ (see `environment.yml` for a Conda example).
- Dependencies in `requirements.txt` (notably `google-genai`, `playwright`, `pydantic`, `Pillow`, `rich`).
- After installing Playwright, install browsers: `playwright install chromium` (or your usual Playwright setup).
- A **Google AI API key** in `GOOGLE_API_KEY` or a `.env` file.

Optional: set `LUMEN_GEMINI_MODEL` to override the default Gemini model (see `engine/vision.py`).

## Quick start

```bash
cd VisionDiff_SemanticAIAuditor
pip install -r requirements.txt
playwright install chromium
export GOOGLE_API_KEY=your_key_here
```

### Audit (diff + two PNGs)

```bash
python cli.py audit \
  --baseline path/to/before.png \
  --current path/to/after.png \
  --diff path/to/changes.diff
```

Writes an annotated overlay to `lumen_annotated_current.png` by default (change with `--annotate-out`, or skip with `--no-annotate`).

### Observe (screenshot + optional V-CoT)

```bash
python cli.py observe --url http://localhost:3000 --out capture.png
python cli.py observe --url localhost --out capture.png --vcot --task "Check that the login form renders correctly."
```

Use `--headful` for a visible browser window while debugging.

## Project layout

| Path | Role |
|------|------|
| `cli.py` | Entry point: `audit` and `observe` subcommands, Rich tables/panels. |
| `engine/auditor.py` | `AuditGraph`: anticipation → observation → reconciliation + root-cause helper. |
| `engine/vision.py` | `AutonomousObserver`, Gemini calls, URL probing, V-CoT. |
| `utils/image_processing.py` | Draws model-provided boxes on PNGs (Pillow). |

## Further reading

For how the Gemini calls, schemas, and multimodal steps fit together, see [`technical_deep_dive.md`](technical_deep_dive.md).
