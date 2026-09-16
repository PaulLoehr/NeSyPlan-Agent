"""Inference providers: which endpoint, credential and reasoning dialect a model id uses.

Until now the whole harness talked to ONE OpenAI-compatible endpoint (Aleph Alpha's
Phoenix / Pharia reverse proxy), so a model was just a `model` string. Adding OpenRouter
(which is also OpenAI-compatible, so the request/response shape is unchanged) means a
model id now also decides *where* the request goes, *which* key signs it, and *how* the
reasoning switch is spelled. That routing lives here, so nesyplan/llm.py stays a plain
OpenAI-compatible client and nesyplan/model_aliases.py stays a flat name table.

Two providers:

  phoenix     -- Aleph Alpha's reverse proxy. Key: BEARER_TOKEN from pharia-iam-cli
                 (short-lived, refreshed by the SessionStart hook). Needs the
                 `x-ai-backend` header. Reasoning switch: `reasoning_effort: <level>`.
  openrouter  -- openrouter.ai/api/v1. Key: OPENROUTER_API_KEY (long-lived, from the
                 OpenRouter dashboard). Reasoning switch: `reasoning: {effort: <level>}`
                 (OpenRouter's own field; it does NOT read `reasoning_effort`).

Routing is by model id, not by env var, so ONE process can mix providers -- the demo can
switch from phoenix to qwen mid-session, and the eval judge / cache summarizer stay on
their own provider regardless of the driver model.

The `effort` vocabulary is deliberately shared across both providers (none / low /
medium / high, empty = omit = the model's own default level), so EpisodeConfig
.reasoning_on_effort and the reasoning=True/False switch mean the same thing everywhere.
Whether a model actually HONORS "none" is a property of the serving backend, not of the
API -- verify with `python3 -m nesyplan.probe_reasoning` before trusting a reasoning-OFF
leg on a new model.
"""

import os

PHOENIX = 'phoenix'
OPENROUTER = 'openrouter'

# No default for the phoenix route on purpose: it is whatever OpenAI-compatible endpoint
# you point it at, and hardcoding one private deployment's hostname into a public
# repository serves nobody. Set PHOENIX_BASE_URL in .env to use this route; without it,
# a phoenix-routed model fails with a clear message instead of dialling somewhere.
PHOENIX_DEFAULT_BASE_URL = ''
OPENROUTER_DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'

# Full model ids whose provider is not obvious from the id alone. Phoenix is the default,
# so only exceptions belong here (`moonshotai/kimi-k2.6` is served BY PHOENIX, even though
# OpenRouter also carries moonshotai models -- hence an explicit table, not slash-sniffing).
MODEL_PROVIDERS = {}

# Fallback for ids not in the table: an id starting with one of these prefixes routes to
# that provider. Keeps `--model qwen/qwen3-235b-a22b` working without a new alias entry.
_PREFIX_ROUTES = (
    ('qwen/', OPENROUTER),
)


class Endpoint:
    """Where and how to call one provider (resolved once per LLMClient)."""

    def __init__(self, provider, base_url, api_key, headers, key_env):
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key
        self.headers = headers
        self.key_env = key_env   # named in the error message when the key is missing


def provider_for(model_id):
    """The provider serving `model_id` (default: phoenix). Expects a FULL id, not an alias."""
    model_id = (model_id or '').strip()
    if model_id in MODEL_PROVIDERS:
        return MODEL_PROVIDERS[model_id]
    low = model_id.lower()
    for prefix, provider in _PREFIX_ROUTES:
        if low.startswith(prefix):
            return provider
    return PHOENIX


def _phoenix_endpoint():
    # The AGENT_*/OPENAI_* overrides are scoped to phoenix on purpose: they exist to point
    # at a customer / c-prod Pharia endpoint instead of the platform proxy, and must not
    # hijack an OpenRouter model in the same process.
    base = (os.environ.get('AGENT_BASE_URL') or os.environ.get('PHOENIX_BASE_URL')
            or os.environ.get('OPENAI_BASE_URL') or PHOENIX_DEFAULT_BASE_URL)
    key = (os.environ.get('AGENT_API_KEY') or os.environ.get('OPENAI_API_KEY')
           or os.environ.get('BEARER_TOKEN') or '')
    # x-ai-backend: what the platform reverse proxy needs; empty for a customer token.
    backend = os.environ.get('PHOENIX_BACKEND', 'd.inference')
    headers = {'x-ai-backend': backend} if backend else {}
    return Endpoint(PHOENIX, base, key, headers, 'BEARER_TOKEN')


def _openrouter_endpoint():
    base = os.environ.get('OPENROUTER_BASE_URL') or OPENROUTER_DEFAULT_BASE_URL
    key = os.environ.get('OPENROUTER_API_KEY') or ''
    # Attribution headers are optional; OpenRouter uses them for its app rankings only.
    headers = {'X-OpenRouter-Title': os.environ.get('OPENROUTER_TITLE', 'NeSyPlan research harness')}
    referer = os.environ.get('OPENROUTER_SITE_URL')
    if referer:
        headers['HTTP-Referer'] = referer
    return Endpoint(OPENROUTER, base, key, headers, 'OPENROUTER_API_KEY')


_BUILDERS = {PHOENIX: _phoenix_endpoint, OPENROUTER: _openrouter_endpoint}


def endpoint_for(model_id):
    """Resolve the Endpoint for `model_id`. Reads env each call, so a token refreshed
    mid-session (or a .env loaded after import) is picked up by the next client."""
    return _BUILDERS[provider_for(model_id)]()


def reasoning_payload(provider, effort):
    """The request field(s) expressing reasoning `effort` for this provider.

    `effort` is the shared vocabulary ('none' | 'low' | 'medium' | 'high'); a falsy value
    returns {} so the field is OMITTED entirely and the model thinks at its own default
    level. OpenRouter ignores `reasoning_effort`, hence the different spelling.
    """
    if not effort:
        return {}
    if provider == OPENROUTER:
        return {'reasoning': {'effort': effort}}
    return {'reasoning_effort': effort}


def routing_payload(provider, has_reasoning):
    """Provider-specific routing preferences (empty for phoenix, one upstream).

    OpenRouter fans one model id out over several upstream providers, and by default a
    provider that does not support a parameter may simply IGNORE it. For a reasoning
    experiment that is silent data corruption (a reasoning-OFF leg that actually reasoned),
    so when we send a `reasoning` object we also require the chosen upstream to support the
    parameters in the request. Escape hatch if routing ever comes up empty ("No endpoints
    available matching..."): OPENROUTER_REQUIRE_PARAMETERS=0.
    """
    if provider != OPENROUTER or not has_reasoning:
        return {}
    if (os.environ.get('OPENROUTER_REQUIRE_PARAMETERS') or '1').strip().lower() in ('0', 'false', 'no'):
        return {}
    return {'provider': {'require_parameters': True}}
