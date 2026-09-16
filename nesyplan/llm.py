"""OpenAI-compatible chat client with a PER-REQUEST reasoning toggle (stdlib only).

Forked from agent/llm.py (which fixes reasoning at construction and discards usage)
and extended for what the orchestrator needs:

  - per-request reasoning switch: chat(..., reasoning=True|False|None). True enables
    thinking at on_effort -- and on_effort defaults to None, which OMITS the effort field
    entirely so the model uses its OWN default level (some models expose low/medium/high;
    we don't force one). Pass an explicit on_effort (low/medium/high) only to pin a level.
    False sends effort "none", the chain-of-thought-off switch -- a no-op on models that
    can't disable thinking (e.g. kimi/command; "minimal"/"low" are no-ops on Phoenix).
    None falls back to the client's default (env AGENT_REASONING_EFFORT or omit). This
    wraps the proposal's provider-agnostic `reasoning: bool`; the wire spelling
    (`reasoning_effort` on Phoenix, `reasoning: {effort}` on OpenRouter) is
    nesyplan/providers.py's job.
  - trace capture: get_trace(message) returns the returned reasoning trace.
  - usage capture: chat returns a ChatResponse carrying the API `usage` object so
    metrics can split reasoning vs completion vs prompt tokens.
  - multi-provider: the endpoint, credential and reasoning dialect follow from the MODEL
    (nesyplan/providers.py), not from a global env var -- so one process can drive a
    Phoenix model and an OpenRouter one, and the summarizer/judge keep their own provider.

Auth is unchanged for Phoenix (Pharia) models -- the repo-root .env (BEARER_TOKEN,
PHOENIX_*), so the SessionStart token hook keeps working. OpenRouter models read
OPENROUTER_API_KEY from the same .env.
"""

import base64
import http.client
import json
import os
import socket
import time
import urllib.error
import urllib.request

from nesyplan.model_aliases import resolve_model
from nesyplan.providers import endpoint_for, reasoning_payload, routing_payload

# HTTP statuses worth retrying (transient server-side / rate-limit); 4xx are NOT.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4   # 1 try + 3 retries, exponential backoff (1/2/4 s)

# ONE shared sampling default for EVERY model, so cross-model AND cross-config runs are
# maximally comparable. The default is GREEDY (temp 0, no top_p).
#
# CORRECTION (measured 2026-07-30, phoenix, identical prompt x3 at temp 0): greedy on this
# endpoint is NOT deterministic. Two of three draws matched, the third differed in content
# and trace (6731 vs 7154 trace chars, 1750 vs 1842 completion tokens) -- consistent with
# MoE routing / batching nondeterminism on the serving stack. So temp 0 does NOT remove the
# need for repetitions; it only hides the variance, leaving one unquantifiable draw. Any
# claim of the form "temp 0, so n=1 is enough" is unsupported here.
#
# Second, independent trade-off: greedy is the degenerate regime for Qwen3-based thinking
# models like phoenix -- it inflates reasoning (~3.7x runaway) and biases effect *magnitudes*
# (16x vs ~3x; see docs/FINDINGS.md C2). Config *ranking* is largely preserved, absolute
# magnitudes are not.
#
# Hence: keep temp 0 for the live demo (predictable on stage), but run measurement campaigns
# at `--temperature 0.6 --top-p 0.95` with `--reps > 1`. 0.6/0.95 is also Qwen3's own
# recommended thinking-mode sampling, so for a Qwen3 family sweep the shared default and
# each model's own default coincide.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = None


def resolve_sampling(temperature=None, top_p=None):
    """Return (temperature, top_p): an explicit value wins, else the shared default.

    The SAME default (DEFAULT_TEMPERATURE / DEFAULT_TOP_P) applies to every model --
    it is deliberately not model-tuned, so cross-model runs use identical sampling.
    """
    t = temperature if temperature is not None else DEFAULT_TEMPERATURE
    p = top_p if top_p is not None else DEFAULT_TOP_P
    return t, p


def _env_model():
    # Raw env value (alias or full id); resolve_model() applies DEFAULT_MODEL when empty.
    return os.environ.get('AGENT_MODEL')


def _env_reasoning_effort():
    return os.environ.get('AGENT_REASONING_EFFORT') or None


def get_trace(message):
    """The reasoning trace the provider returned with an assistant message, or None.

    Field name is provider-specific; we accept the common variants. The trace is
    returned but NOT automatically re-fed by the API -- the ReasoningCache is what
    re-injects it (see nesyplan/cache.py).
    """
    if not isinstance(message, dict):
        return None
    for key in ('reasoning', 'reasoning_content'):
        val = message.get(key)
        if val:
            return val
    return None


class ChatResponse:
    """Wraps one completion: the assistant message + the API usage object.

    latency_ms is normally None (callers time the call themselves); nesyplan/replay.py sets it
    to the RECORDED duration, so a replayed turn can be shown with the thinking time it
    actually took instead of how long the replay took to hand it over.
    """

    def __init__(self, message, usage, latency_ms=None):
        self.message = message or {}
        self.usage = usage or {}
        self.latency_ms = latency_ms

    @property
    def content(self):
        return self.message.get('content')

    @property
    def tool_calls(self):
        return self.message.get('tool_calls') or []

    @property
    def trace(self):
        return get_trace(self.message)


class LLMClient:
    def __init__(self, base_url=None, api_key=None, model=None, temperature=None, top_p=None,
                 timeout=120, on_effort=None, default_reasoning_effort=None):
        # Accept a short alias (phoenix/kimi/command/merlin/qwen3-32b) or a full id; store
        # the resolved FULL id so callers, the console line and the log agree.
        self.model = resolve_model(model or _env_model())
        # The MODEL picks the endpoint: Phoenix (Pharia) or OpenRouter, each with its own
        # base URL, key, headers and reasoning dialect (nesyplan/providers.py).
        endpoint = endpoint_for(self.model)
        self.provider = endpoint.provider
        self.base_url = (base_url or endpoint.base_url).rstrip('/')
        self.api_key = api_key if api_key is not None else endpoint.api_key
        self._headers = endpoint.headers
        self._key_env = endpoint.key_env
        # Remembered so with_model() can re-resolve the provider for the new model while
        # still honoring an explicit override (an override is for ONE endpoint, not all).
        self._base_url_override = base_url
        self._api_key_override = api_key
        # temperature / top_p: an explicit value wins; None -> the shared default
        # (nesyplan.llm.DEFAULT_*), identical for every model so runs stay comparable.
        self.temperature, self.top_p = resolve_sampling(temperature, top_p)
        self.timeout = timeout
        # What a reasoning=True turn sends. None (default) OMITS reasoning_effort so the
        # model thinks at its OWN default level; set it (low/medium/high) to pin a level.
        self.on_effort = on_effort
        # Used only when a call passes reasoning=None (no explicit per-turn decision).
        self.default_reasoning_effort = (default_reasoning_effort if default_reasoning_effort is not None
                                         else _env_reasoning_effort())

    def with_model(self, model):
        """A sibling client with the SAME sampling but a different model -- re-resolving the
        endpoint, because the new model may live at a different provider.

        Used for the cache summarizer, which is pinned to one model regardless of the driver
        (see nesyplan.cache.SUMMARIZER_MODEL): with an OpenRouter driver, the phoenix
        summarizer must still go to Phoenix, so the endpoint follows `model`, not `self`.
        """
        return LLMClient(base_url=self._base_url_override, api_key=self._api_key_override,
                         model=model, temperature=self.temperature, top_p=self.top_p,
                         timeout=self.timeout, on_effort=self.on_effort,
                         default_reasoning_effort=self.default_reasoning_effort)

    def _effort_for(self, reasoning):
        if reasoning is True:
            return self.on_effort            # None -> omit -> model's own default level
        if reasoning is False:
            return 'none'                    # off where supported; no-op elsewhere
        return self.default_reasoning_effort  # None -> omit

    def _auth_hint(self, status):
        """For an auth/billing rejection: which provider, which key, and is it expired?

        A bare provider 401/402 body leaves the two obvious questions open -- did the
        request even go where I think, and is my token stale? Both are answerable locally,
        so answer them in the message. Note an UNEXPIRED token can still be rejected (the
        endpoint may drop a signing key it no longer knows), hence printing exp either way.
        """
        if status not in (401, 402, 403):
            return ''
        hint = f'\n  provider={self.provider} base_url={self.base_url} model={self.model}' \
               f'\n  key from ${self._key_env}'
        try:   # JWT exp, unverified -- Phoenix tokens are short-lived (~12 h)
            payload = self.api_key.split('.')[1]
            exp = int(json.loads(base64.urlsafe_b64decode(
                payload + '=' * (-len(payload) % 4)))['exp'])
            left = (exp - time.time()) / 60
            hint += (f' (expired {-left:.0f} min ago -- mint a fresh one)' if left <= 0
                     else f' (valid another {left:.0f} min, so NOT expired: an "unknown key" '
                          f'401 is the endpoint rejecting the signing key itself)')
        except Exception:   # noqa: BLE001 -- not a JWT (e.g. an OpenRouter key): nothing to add
            pass
        return hint

    def chat(self, messages, tools=None, tool_choice=None, reasoning=None):
        """One chat completion. Returns a ChatResponse (message + usage).

        reasoning: True (deliberate, effort=on_effort) | False (effort="none") |
        None (client default). The reasoning trace, if returned, is on the message
        and reachable via response.trace / get_trace(response.message).
        """
        if not self.api_key:
            raise RuntimeError(f'no API key for provider {self.provider!r} (model {self.model}): '
                               f'set {self._key_env} in the repo-root .env')
        if not self.base_url:
            # Only the phoenix route can land here: it ships no default endpoint (see
            # nesyplan/providers.py). Say which knob is missing rather than fail on a URL
            # that is the empty string.
            raise RuntimeError(
                f'no endpoint for provider {self.provider!r} (model {self.model}): set '
                f'PHOENIX_BASE_URL in the repo-root .env, or pick a model served by a '
                f'provider that has one (e.g. --model qwen3-32b).')
        payload = {'model': self.model, 'temperature': self.temperature, 'messages': messages}
        if self.top_p is not None:
            payload['top_p'] = self.top_p
        effort = self._effort_for(reasoning)
        # How the effort is spelled is provider-specific (reasoning_effort vs. a `reasoning`
        # object); a falsy effort omits it entirely -> the model's own default level.
        payload.update(reasoning_payload(self.provider, effort))
        payload.update(routing_payload(self.provider, bool(effort)))
        if tools:
            payload['tools'] = tools
            if tool_choice is not None:
                payload['tool_choice'] = tool_choice

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}
        headers.update(self._headers)
        data = json.dumps(payload).encode('utf-8')

        # Retry transient network/proxy failures: the reverse proxy occasionally drops
        # connections mid-run (RemoteDisconnected / connection refused) and returns
        # 429/5xx under load. Real client errors (4xx) and bad payloads are NOT retried.
        for attempt in range(_MAX_ATTEMPTS):
            req = urllib.request.Request(self.base_url + '/chat/completions',
                                         data=data, headers=headers, method='POST')
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.load(resp)
                return ChatResponse(body['choices'][0]['message'], body.get('usage'))
            except urllib.error.HTTPError as exc:
                if exc.code in _RETRY_STATUS and attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
                    continue
                detail = exc.read().decode('utf-8', 'replace')[:800]
                raise RuntimeError(
                    f'LLM request failed: HTTP {exc.code} {exc.reason} -- {detail}'
                    f'{self._auth_hint(exc.code)}') from exc
            except (urllib.error.URLError, http.client.HTTPException, ConnectionError,
                    TimeoutError, socket.timeout) as exc:
                if attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(
                    f'LLM request failed after {attempt + 1} attempts: {exc!r}') from exc
