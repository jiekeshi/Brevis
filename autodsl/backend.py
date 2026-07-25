"""LLM backends for the proposer.

Deliberately dependency-free: the cluster installs from a local wheel index and
`numpy` is the only thing `requirements.txt` asks for, so this speaks HTTP
directly rather than pulling in `openai` or `anthropic`. Both protocols are a
single JSON POST.

The proposer is the only place a model appears in this system. It never
compresses, never decompresses, and never decides what is kept — it writes
candidate grammar, and the deterministic gates in `verify.py` and
`evaluate.py` decide the candidate's fate.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import time
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent
CREDENTIAL = REPO / "llm.credential"

OPENAI_BASE = "https://api.openai-proxy.org/v1"
ANTHROPIC_BASE = "https://api.openai-proxy.org/anthropic"

# Reachable through the same proxy with the same key; protocol differs.
KNOWN_MODELS = {
    "qwen3.7-max": "openai",
    "qwen3.6-plus": "openai",
    "claude-opus-5": "anthropic",
}
DEFAULT_MODEL = "claude-opus-5"


class BackendError(RuntimeError):
    pass


def load_api_key(path: pathlib.Path = CREDENTIAL) -> str:
    """Read the key from `llm.credential`, `$BREVIS_LLM_API_KEY`, or fail.

    Accepts a bare key, `NAME=value` lines, or a JSON object with an
    `api_key`/`key` field, because the file is written by hand. The value is
    never logged: callers get the string and nothing else records it.
    """
    from_env = os.environ.get("BREVIS_LLM_API_KEY")
    if from_env:
        return from_env.strip()
    if not path.exists():
        raise BackendError(
            f"no API key: set $BREVIS_LLM_API_KEY or create {path} "
            "(it is git-ignored)"
        )
    text = path.read_text().strip()
    if text.startswith("{"):
        blob = json.loads(text)
        for field in ("api_key", "key", "apiKey"):
            if blob.get(field):
                return str(blob[field]).strip()
        raise BackendError(f"{path}: JSON has no api_key field")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line and not line.startswith("sk-"):
            return line.split("=", 1)[1].strip().strip("'\"")
        return line
    raise BackendError(f"{path}: empty")


@dataclasses.dataclass
class Backend:
    """One model behind one protocol. `complete` is the whole interface."""

    model: str = DEFAULT_MODEL
    protocol: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    # Reasoning models spend this budget on thinking before any text is
    # emitted; too small a cap returns a response with no text at all.
    max_tokens: int = 16000
    # Omitted from the request when None. Newer reasoning models reject it
    # outright, so a default of None keeps one class usable across every
    # model the proxy exposes.
    temperature: float | None = None
    timeout_s: float = 240.0
    retries: int = 3

    def __post_init__(self) -> None:
        if self.protocol is None:
            self.protocol = KNOWN_MODELS.get(self.model, "openai")
        if self.protocol not in ("openai", "anthropic"):
            raise BackendError(f"unknown protocol {self.protocol!r}")
        if self.base_url is None:
            self.base_url = (
                ANTHROPIC_BASE if self.protocol == "anthropic" else OPENAI_BASE
            )
        if self.api_key is None:
            self.api_key = load_api_key()

    def describe(self) -> dict:
        """Everything about this backend except the key."""
        return {
            "model": self.model,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def complete(self, system: str, user: str) -> str:
        if self.protocol == "anthropic":
            url = f"{self.base_url}/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            body = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            }
            extract = _extract_anthropic
        else:
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "content-type": "application/json",
            }
            body = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            extract = _extract_openai
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return extract(self._post(url, headers, body))

    def _post(self, url: str, headers: dict, body: dict) -> dict:
        payload = json.dumps(body).encode()
        last: Exception | None = None
        for attempt in range(self.retries):
            request = urllib.request.Request(
                url, data=payload, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:2000]
                last = BackendError(f"HTTP {exc.code} from {url}: {detail}")
                if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                    raise last from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = BackendError(f"{type(exc).__name__} from {url}: {exc}")
            if attempt + 1 < self.retries:
                time.sleep(2.0 * (2**attempt))
        raise last if last else BackendError("request failed with no error recorded")


def _extract_anthropic(response: dict) -> str:
    parts = [
        block.get("text", "")
        for block in response.get("content", [])
        if block.get("type") == "text"
    ]
    if not any(parts):
        if response.get("stop_reason") == "max_tokens":
            raise BackendError(
                "response hit max_tokens before emitting any text — the model "
                "spent the whole budget thinking; raise max_tokens"
            )
        raise BackendError(f"no text in response: {json.dumps(response)[:600]}")
    return "".join(parts)


def _extract_openai(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise BackendError(f"no choices in response: {json.dumps(response)[:800]}")
    content = choices[0].get("message", {}).get("content")
    if not content:
        raise BackendError(f"empty content: {json.dumps(response)[:800]}")
    return content


@dataclasses.dataclass
class ScriptedBackend:
    """A backend that replays fixed replies. Used by the tests, and by
    `--dry-run`, so the loop can be exercised without a network or a key."""

    replies: list[str]
    calls: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    model: str = "scripted"
    protocol: str = "none"

    def describe(self) -> dict:
        return {"model": self.model, "protocol": self.protocol}

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.replies:
            raise BackendError("ScriptedBackend ran out of replies")
        return self.replies.pop(0)
