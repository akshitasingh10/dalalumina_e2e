"""Autonomous browser observation: URL probing, resilient screenshots, Visual Chain-of-Thought."""

from __future__ import annotations

import os
import ssl
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

import structlog
from google import genai
from google.genai import types
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = structlog.get_logger(__name__)

DEFAULT_LUMEN_MODEL = os.environ.get("LUMEN_GEMINI_MODEL", "gemini-2.5-flash")
COMMON_PORTS: tuple[int, ...] = (3000, 5173, 8080)


class ScanPhase(BaseModel):
    """What is visibly present (neutral inventory)."""

    visible_structure: str = Field(description="Layout regions, key UI chrome, obvious components.")
    salient_elements: list[str] = Field(
        description="Short labels for salient widgets or content blocks.",
        default_factory=list,
    )


class HypothesisPhase(BaseModel):
    """Predicted behavior or rendering intent."""

    predicted_rendering: str = Field(description="What the UI should show given the task context.")
    risk_areas: list[str] = Field(
        description="Where visual bugs would most likely appear.",
        default_factory=list,
    )


class VerifyPhase(BaseModel):
    """Multi-turn self-correction: does observation match the mental model?"""

    matches_mental_model: bool
    discrepancy_summary: str = Field(default="")
    corrected_assessment: str = Field(
        default="",
        description="If mismatch, revised description after re-examining the image.",
    )
    confidence_0_to_1: float = Field(ge=0.0, le=1.0, default=0.8)


class VisualCoTResult(BaseModel):
    """Full V-CoT trace: Scan → Hypothesize → Verify (with optional correction)."""

    scan: ScanPhase
    hypothesis: HypothesisPhase
    verify: VerifyPhase


def _url_reachable(url: str, timeout: float = 4.0) -> bool:
    ctx = ssl.create_default_context()
    headers = {"User-Agent": "LumenAutonomousObserver/1.0"}
    for method in ("HEAD", "GET"):
        try:
            req = Request(url, method=method, headers=headers)
            with urlopen(req, timeout=timeout, context=ctx) as resp:  # noqa: S310
                return 200 <= resp.status < 400
        except HTTPError as exc:
            if exc.code in (405, 501) and method == "HEAD":
                continue
            return False
        except (URLError, TimeoutError, OSError):
            if method == "HEAD":
                continue
            return False
    return False


def _build_port_candidates(url: str) -> list[str]:
    seed = url.strip()
    if "://" not in seed:
        seed = f"http://{seed}"
    parsed = urlparse(seed)
    if not parsed.scheme:
        parsed = parsed._replace(scheme="http")
    host = parsed.hostname
    if not host:
        return [urlunparse(parsed)]
    netloc_base = host
    if "@" in parsed.netloc:
        netloc_base = parsed.netloc.split("@")[-1]
    path = parsed.path or "/"
    query = parsed.query
    fragment = parsed.fragment

    seen: set[str] = set()
    out: list[str] = []

    def push(netloc: str) -> None:
        u = urlunparse((parsed.scheme, netloc, path, "", query, fragment))
        if u not in seen:
            seen.add(u)
            out.append(u)

    if parsed.port:
        push(parsed.netloc)
        for p in COMMON_PORTS:
            if p != parsed.port:
                push(f"{host}:{p}")
    else:
        push(host)
        for p in COMMON_PORTS:
            push(f"{host}:{p}")
    return out


def resolve_reachable_url(seed_url: str) -> str:
    """Try the given URL, then common dev ports, before failing."""
    candidates = _build_port_candidates(seed_url)
    log.info("url_probe_candidates", count=len(candidates))
    for u in candidates:
        if _url_reachable(u):
            log.info("url_resolved", url=u)
            return u
    raise ConnectionError(
        f"Could not reach any candidate URL for {seed_url!r} (tried {len(candidates)} variants)."
    )


def _gemini_generate(
    client: genai.Client,
    *,
    model: str,
    parts: list[types.Part],
    response_schema: type[BaseModel] | None = None,
    system_instruction: str | None = None,
) -> BaseModel | str:
    config_kwargs: dict = {}
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=45),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call() -> types.GenerateContentResponse:
        return client.models.generate_content(model=model, contents=parts, config=config)

    resp = _call()
    if response_schema is not None:
        if resp.parsed is not None:
            return resp.parsed  # type: ignore[return-value]
        raise RuntimeError("Structured response missing parsed payload; raw: " + (resp.text or ""))
    return (resp.text or "").strip()


class AutonomousObserver:
    """Playwright capture with self-healing selectors + Gemini V-CoT."""

    def __init__(
        self,
        client: genai.Client,
        *,
        model: str | None = None,
        headless: bool = True,
    ) -> None:
        self._client = client
        self._model = model or DEFAULT_LUMEN_MODEL
        self._headless = headless

    def capture_screenshot(
        self,
        url: str,
        *,
        selector: str | None = None,
        full_page: bool = True,
        viewport: tuple[int, int] = (1280, 720),
        goto_timeout_ms: float = 30_000,
    ) -> tuple[bytes, str]:
        """Return (png_bytes, resolved_url). Falls back from selector → body → full page."""
        resolved = resolve_reachable_url(url)
        log.info("playwright_goto", url=resolved, selector=selector)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self._headless)
            try:
                page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
                page.goto(resolved, wait_until="networkidle", timeout=goto_timeout_ms)

                shot: bytes | None = None
                if selector:
                    try:
                        loc = page.locator(selector).first
                        loc.wait_for(state="visible", timeout=8_000)
                        shot = loc.screenshot(type="png")
                        log.info("screenshot_mode", mode="selector", selector=selector)
                    except PlaywrightTimeoutError:
                        log.warning("selector_timeout_falling_back", selector=selector)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("selector_failed_falling_back", error=str(exc))

                if shot is None:
                    try:
                        body = page.locator("body")
                        body.wait_for(state="visible", timeout=5_000)
                        shot = body.screenshot(type="png")
                        log.info("screenshot_mode", mode="body")
                    except Exception as exc:  # noqa: BLE001
                        log.warning("body_screenshot_failed", error=str(exc))
                        shot = page.screenshot(full_page=full_page, type="png")
                        log.info("screenshot_mode", mode="full_page")

                return shot, resolved
            finally:
                browser.close()

    def visual_chain_of_thought(
        self,
        screenshot_png: bytes,
        *,
        task_context: str,
        system_instruction: str | None = None,
    ) -> VisualCoTResult:
        """Scan → Hypothesize → Verify with a second corrective pass when needed."""
        base_si = system_instruction or (
            "You are Lumen, an autonomous UI architect. Use Visual Chain-of-Thought: "
            "first inventory what you see, then state what should be true given the task, "
            "then verify against the image. Be precise and concise."
        )
        img = types.Part.from_bytes(data=screenshot_png, mime_type="image/png")

        # Pass 1: structured Scan + Hypothesis (same call for latency).
        class _Pass1(BaseModel):
            scan: ScanPhase
            hypothesis: HypothesisPhase

        p1_prompt = (
            f"Task context:\n{task_context}\n\n"
            "Phase SCAN: Describe visible structure and salient elements.\n"
            "Phase HYPOTHESIZE: Given the task, what should the UI show? List risk areas."
        )
        p1 = _gemini_generate(
            self._client,
            model=self._model,
            parts=[types.Part.from_text(text=p1_prompt), img],
            response_schema=_Pass1,
            system_instruction=base_si,
        )
        assert isinstance(p1, _Pass1)

        p2_prompt = (
            f"Task context:\n{task_context}\n\n"
            f"Prior SCAN:\n{p1.scan.model_dump_json()}\n\n"
            f"Prior HYPOTHESIS:\n{p1.hypothesis.model_dump_json()}\n\n"
            "Phase VERIFY: Compare the screenshot to the hypothesis. "
            "If anything conflicts, set matches_mental_model=false and explain, "
            "then give corrected_assessment after re-checking the image."
        )
        verify = _gemini_generate(
            self._client,
            model=self._model,
            parts=[types.Part.from_text(text=p2_prompt), img],
            response_schema=VerifyPhase,
            system_instruction=base_si,
        )
        assert isinstance(verify, VerifyPhase)

        if not verify.matches_mental_model and not verify.corrected_assessment:
            # Light-weight self-correction nudge (multi-turn).
            p3_prompt = (
                "Your verification reported a mismatch but corrected_assessment was empty. "
                "Re-examine the image and produce corrected_assessment (concrete UI description)."
            )
            verify2 = _gemini_generate(
                self._client,
                model=self._model,
                parts=[types.Part.from_text(text=p3_prompt), img],
                response_schema=VerifyPhase,
                system_instruction=base_si,
            )
            assert isinstance(verify2, VerifyPhase)
            verify = verify2

        return VisualCoTResult(scan=p1.scan, hypothesis=p1.hypothesis, verify=verify)
