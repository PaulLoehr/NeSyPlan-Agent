"""Short model aliases -> full OpenAI-compatible model ids.

Every model this harness can drive is reached over an OpenAI-compatible endpoint, so
choosing a different model is only a matter of sending a different `model` id. This maps a
short, memorable alias to the exact production id -- so you set AGENT_MODEL=qwen3-32b (or
pass --model qwen3-32b) while the request, the console line and the JSON transcript all
carry the full `qwen/qwen3-32b`.

Which ENDPOINT an id goes to is not decided here: nesyplan/providers.py routes each full id
to its provider (OpenRouter for the `qwen/...` ids, Phoenix otherwise) and supplies that
provider's base URL, key and reasoning dialect. Adding a model from a new provider means
one line here plus, if it is a new provider, one route there.

resolve_model() is deliberately lenient: an already-full id -- or any value that is not a
known alias -- is returned unchanged, so you can always pass the exact model string too.
An empty / missing value resolves to DEFAULT_MODEL.

Two provider families are wired up:

  OpenRouter (OPENROUTER_API_KEY) -- the qwen ids. This is the path anyone can reproduce:
      a normal account and some credit is all it takes. See README.md.
  "phoenix" route (BEARER_TOKEN + PHOENIX_BASE_URL) -- a generic OpenAI-compatible
      endpoint with NO default host: point it wherever you like. The four non-qwen aliases
      below name models on Aleph Alpha's internal deployment, kept because the experiment
      in docs/EXPERIMENT.md was developed against them; without access to it (and without
      PHOENIX_BASE_URL set) they will not resolve. Use the qwen aliases instead.

A bad id fails at request time (404 from the provider), so sanity-check a new entry with
`python3 -m nesyplan.probe_reasoning --models <alias>`.
"""
MODEL_ALIASES = {
    # --- OpenRouter (OPENROUTER_API_KEY) ---------------------------------------------
    # One model family, one tokenizer, one thinking-mode switch at every size. That turns
    # "model" from a confound into an AXIS: a pool that mixes 30B with 1T models, and mixes
    # models whose reasoning can be disabled with models whose cannot, yields no
    # interpretable cross-model number.
    #
    # VERIFIED live (2026-08-31, POST /chat/completions with a tool definition): all three
    # answer, call tools, and return a reasoning trace; at effort "none" the trace is
    # genuinely 0 chars, so a reasoning-OFF leg on them is real and not silently reasoning.
    # Upstream was DeepInfra for all three at the time of the probe.
    'qwen3-14b':     'qwen/qwen3-14b',
    'qwen3-30b-a3b': 'qwen/qwen3-30b-a3b',
    'qwen3-32b':     'qwen/qwen3-32b',
    # In OpenRouter's catalog, but it has exactly ONE upstream (Alibaba) and that upstream
    # is blocked unless your account's data policy allows it -- otherwise every request is
    # HTTP 404 "No endpoints available matching your guardrail restrictions and data
    # policy". Fix it once at https://openrouter.ai/settings/privacy, or leave this alias
    # unused. (qwen/qwen3-4b is NOT in the catalog at all -- do not add it back.)
    'qwen3-8b':      'qwen/qwen3-8b',

    # --- Phoenix / Pharia (BEARER_TOKEN) ---------------------------------------------
    # Aleph Alpha's internal endpoint. Unreachable without access to that deployment; kept
    # because docs/EXPERIMENT.md's development runs used them. `phoenix` is Qwen3-derived,
    # which is what makes qwen3-30b-a3b its closest public sibling.
    'phoenix':   'Aleph-Alpha-Research/phoenix-1-chat-v2',
    'kimi':      'moonshotai/kimi-k2.6',
    'command':   'CohereLabs/command-a-plus-05-2026-w4a4',
    'merlin':    'Aleph-Alpha-Research/merlin_arthur_chat',
}

# The size ladder in order, for a scaling sweep. qwen3-8b is omitted: it needs the account
# data-policy change described above, so a sweep would fail on that rung for most users.
#   --model "$(python3 -c 'from nesyplan.model_aliases import QWEN_LADDER; print(",".join(QWEN_LADDER))')"
QWEN_LADDER = ['qwen3-14b', 'qwen3-30b-a3b', 'qwen3-32b']

# Default when no model is specified anywhere (env / CLI both empty). A publicly reachable
# model on purpose: the repository must be runnable by someone who only has an OpenRouter key.
# DEFAULT_ALIAS is the short name (what an interactive picker should preselect and echo);
# DEFAULT_MODEL is the full id that goes on the wire. Both must name the SAME model, so they
# are defined together here rather than spelled out again at each entry point.
DEFAULT_ALIAS = 'qwen3-32b'
DEFAULT_MODEL = MODEL_ALIASES[DEFAULT_ALIAS]


def resolve_model(name):
    """Map a short alias to its full model id.

    Case-insensitive on the alias; surrounding whitespace is trimmed. A value
    that is not a known alias (e.g. an already-full id like
    'Aleph-Alpha-Research/phoenix-1-chat-v2') is returned unchanged. Empty / None
    yields DEFAULT_MODEL.
    """
    if not name or not name.strip():
        return DEFAULT_MODEL
    key = name.strip()
    return MODEL_ALIASES.get(key.lower(), key)
