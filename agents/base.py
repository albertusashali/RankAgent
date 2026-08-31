"""Shared machinery for the specialised agents.

Three things live here because every role needs them and none should reimplement
them:

  * **One LLM client**, resolved once per run, supporting Anthropic or OpenAI.
  * **Token accounting attributed per role**, so the report can say what the
    Product Manager cost versus the Engineer. Feasibility is graded on total
    spend, and per-role numbers are what make that number actionable.
  * **A strict JSON contract**. Every agent asks for a JSON object and validates
    it against a Pydantic model. A response that does not parse or does not
    validate is a *failed call*, and the agent falls back to its deterministic
    behaviour rather than passing malformed data downstream.

That last rule is the one that matters. The earlier single-agent loop accepted a
half-usable LLM response and substituted a fallback command while keeping the
LLM's prose, which produced a run log claiming one experiment and running another.
Here a proposal is accepted whole or replaced whole.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from orchestrator.schemas import TokenUsage

T = TypeVar("T", bound=BaseModel)

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
DEFAULT_OPENAI_MODEL = "gpt-4o"


class LLMUnavailable(RuntimeError):
    """Raised when no usable client is configured; callers fall back."""


class LLMClient:
    """Thin provider-agnostic wrapper that meters every call."""

    def __init__(self, meter: TokenUsage):
        self.meter = meter
        self.per_agent: Dict[str, Dict[str, int]] = {}
        self._client = None
        self._kind = None
        self._resolve()

    def _resolve(self):
        key = os.environ.get("ANTHROPIC_API_KEY")
        if key and len(key) > 10 and not key.lower().startswith("your"):
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=key)
                self._kind = "anthropic"
                self.model = os.environ.get("RANKAGENT_MODEL", DEFAULT_ANTHROPIC_MODEL)
                return
            except Exception:
                pass
        key = os.environ.get("OPENAI_API_KEY")
        if key and len(key) > 10 and not key.lower().startswith("your"):
            try:
                import openai
                self._client = openai.OpenAI(api_key=key)
                self._kind = "openai"
                self.model = os.environ.get("RANKAGENT_MODEL", DEFAULT_OPENAI_MODEL)
                return
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return self._client is not None

    def _record(self, agent: str, prompt_tokens: int, completion_tokens: int):
        self.meter.add(prompt_tokens, completion_tokens)
        slot = self.per_agent.setdefault(agent, {"prompt": 0, "completion": 0, "calls": 0})
        slot["prompt"] += int(prompt_tokens or 0)
        slot["completion"] += int(completion_tokens or 0)
        slot["calls"] += 1

    def complete(self, agent: str, system: str, user: str, max_tokens: int = 1500,
                 json_mode: bool = True) -> str:
        """One completion. Set ``json_mode=False`` for free-form output.

        Most agents return a JSON object, so JSON mode is the default. The
        Engineer's code path must not use it: it returns SEARCH/REPLACE blocks,
        and OpenAI rejects the request outright when JSON mode is on and the
        prompt does not ask for JSON.
        """
        if not self.available:
            raise LLMUnavailable("no API key configured")
        if self._kind == "anthropic":
            resp = self._client.messages.create(
                model=self.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in resp.content if b.type == "text")
            self._record(agent, resp.usage.input_tokens, resp.usage.output_tokens)
            return text
        kwargs: Dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(
            model=self.model, temperature=0.3,
            # max_tokens was previously not forwarded at all, so a patch large
            # enough to matter was truncated by the provider default.
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **kwargs)
        text = resp.choices[0].message.content or ""
        self._record(agent, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return text


def extract_json(text: str) -> Any:
    """Pull the first JSON object or array out of a model response.

    Models wrap JSON in prose or fences often enough that being strict here would
    burn iterations on formatting rather than research.
    """
    if not text:
        raise ValueError("empty response")
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1)
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"no JSON found in response: {text[:200]!r}")
    start = min(starts)
    closer = "}" if text[start] == "{" else "]"
    end = text.rfind(closer)
    if end <= start:
        raise ValueError("unterminated JSON in response")
    return json.loads(text[start:end + 1])


class Agent:
    """Base class: a named role with a system prompt and a fallback.

    Subclasses implement ``_build_prompt``, ``_parse``, and ``fallback``.
    ``run`` handles the call, the JSON extraction, the validation and the
    fallback, so no subclass has to get that sequence right on its own.
    """

    name: str = "agent"
    system_prompt: str = ""
    max_tokens: int = 1500

    def __init__(self, llm: Optional[LLMClient] = None, verbose: bool = True):
        self.llm = llm
        self.verbose = verbose
        self.last_error: Optional[str] = None

    # -- subclass hooks ---------------------------------------------------

    def _build_prompt(self, ctx, **kwargs) -> str:
        raise NotImplementedError

    def _parse(self, payload: Any, ctx, **kwargs):
        raise NotImplementedError

    def fallback(self, ctx, **kwargs):
        raise NotImplementedError

    # -- driver -----------------------------------------------------------

    def run(self, ctx, **kwargs):
        """Consult the LLM; on any problem, fall back deterministically.

        A failure here is never fatal and never partial: the agent either returns
        a fully validated object from the model, or a fully formed fallback.
        """
        self.last_error = None
        if self.llm is None or not self.llm.available:
            return self.fallback(ctx, **kwargs)
        try:
            text = self.llm.complete(self.name, self.system_prompt,
                                     self._build_prompt(ctx, **kwargs),
                                     max_tokens=self.max_tokens)
            return self._parse(extract_json(text), ctx, **kwargs)
        except (ValidationError, ValueError, KeyError, TypeError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:                       # transport, rate limit, etc.
            self.last_error = f"{type(exc).__name__}: {exc}"
        if self.verbose:
            print(f"    [{self.name}] LLM output unusable ({self.last_error}); "
                  f"using deterministic fallback")
        return self.fallback(ctx, **kwargs)


def validated(model: Type[T], payload: Any) -> T:
    """Validate a parsed payload, raising ValidationError on mismatch."""
    return model.model_validate(payload)
