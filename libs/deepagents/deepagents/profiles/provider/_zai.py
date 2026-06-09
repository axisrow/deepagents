"""Built-in z.ai (GLM Coding Plan) provider profile.

z.ai exposes GLM models (`glm-5.1`, `glm-4.7`, `glm-4.6`, `glm-4.5-air`) over an
OpenAI-compatible Chat Completions endpoint. LangChain has no native `zai`
provider — `init_chat_model("zai:...")` cannot infer one — so this profile maps
the `zai:` prefix onto the OpenAI client (`langchain_openai.ChatOpenAI`) pointed
at z.ai's Coding Plan base URL.

Registered directly by `_ensure_builtin_profiles_loaded` during the first
profile-registry access. Not exposed as an `importlib.metadata` entry point —
built-ins ship with the SDK and should not depend on install-time metadata to
activate.

The `model_provider="openai"` kwarg in the returned profile is what tells
`resolve_model` to strip the `zai:` prefix before calling `init_chat_model`, so
the model name reaches z.ai bare (`glm-4.6`, not `zai:glm-4.6`).
"""

from __future__ import annotations

import os
from typing import Any

from deepagents.profiles.provider.provider_profiles import (
    ProviderProfile,
    _register_provider_profile_impl,
)

ZAI_CODING_BASE_URL = "https://api.z.ai/api/coding/paas/v4"
"""Default base URL for the z.ai GLM Coding Plan OpenAI-compatible endpoint.

z.ai also offers an Anthropic-protocol endpoint (`/api/anthropic`) for Claude
Code and Goose; the SDK uses the OpenAI Chat Completions endpoint, which is the
one z.ai recommends for general (non-Claude-Code) tools.
"""

ZAI_API_KEY_ENV = "ZAI_API_KEY"
"""Environment variable holding the z.ai API key (the GLM Coding Plan key)."""

ZAI_BASE_URL_ENV = "ZAI_BASE_URL"
"""Optional override for `ZAI_CODING_BASE_URL` (e.g. a regional or proxy host)."""


def _zai_init_kwargs() -> dict[str, Any]:
    """Build `init_chat_model` kwargs for z.ai, reading credentials lazily.

    Env vars are read at resolution time (not registration time) so a missing
    `ZAI_API_KEY` only surfaces when a `zai:*` model is actually constructed,
    mirroring the OpenRouter profile's lazy-env behavior.

    `use_responses_api=False` is mandatory: the built-in OpenAI profile enables
    the Responses API, but z.ai's Chat Completions endpoint does not implement
    it, so the call would fail without this override.

    Returns:
        Dictionary of kwargs to spread into `init_chat_model`.
    """
    return {
        "model_provider": "openai",
        "base_url": os.environ.get(ZAI_BASE_URL_ENV) or ZAI_CODING_BASE_URL,
        "api_key": os.environ.get(ZAI_API_KEY_ENV),
        "use_responses_api": False,
    }


def register() -> None:
    """Register the built-in z.ai provider profile."""
    _register_provider_profile_impl(
        "zai",
        ProviderProfile(init_kwargs_factory=_zai_init_kwargs),
    )
