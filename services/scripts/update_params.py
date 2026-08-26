#!/usr/bin/env python3
"""
Template-based update_services.py for Groq.

Yields model dictionaries that are rendered using Jinja2 templates.

Usage: python scripts/update_services.py
"""

import os
import sys
from pathlib import Path
from typing import Iterator

import httpx

from unitysvc_sellers.model_data import ModelDataFetcher, ModelDataLookup
from unitysvc_sellers.params_render import write_params_from_iterator

# Provider Configuration
PROVIDER_NAME = "groq"
PROVIDER_DISPLAY_NAME = "Groq"
API_BASE_URL = "https://api.groq.com/openai/v1"
ENV_API_KEY_NAME = "GROQ_API_KEY"

SCRIPT_DIR = Path(__file__).parent


class ModelSource:
    """Fetches models and yields template dictionaries."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.data_fetcher = ModelDataFetcher()
        self.litellm_data = None

    def iter_models(self) -> Iterator[dict]:
        """Yield model dictionaries for template rendering."""
        # Fetch LiteLLM data once
        self.litellm_data = self.data_fetcher.fetch_litellm_model_data()

        print(f"Fetching models from {PROVIDER_DISPLAY_NAME} API...")
        try:
            r = httpx.get(
                f"{API_BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=30.0,
            )
            r.raise_for_status()
            models = r.json().get("data", [])
            print(f"Found {len(models)} models\n")
        except Exception as e:
            print(f"Error listing models: {e}")
            return

        for i, model_info in enumerate(models, 1):
            model_id = model_info.get("id", "")
            print(f"[{i}/{len(models)}] {model_id}")

            # Build template variables
            template_vars = self._build_template_vars(model_id, model_info)
            if template_vars:
                yield template_vars
                print("  OK")

    def _build_template_vars(self, model_id: str, model_info: dict) -> dict:
        """Build template variables for a model."""
        service_type = self._determine_service_type(model_id)
        modality = self._modality(model_id, model_info)
        display_name = model_id.replace("-", " ").replace("_", " ").title()

        # Build details from LiteLLM data and model info
        details = {}
        model_data = ModelDataLookup.lookup_model_details(
            model_id, self.litellm_data or {})

        if model_data:
            for field in [
                    "max_tokens", "max_input_tokens", "max_output_tokens",
                    "mode"
            ]:
                if field in model_data:
                    details[field] = model_data[field]
            if "litellm_provider" in model_data:
                details["litellm_provider"] = model_data["litellm_provider"]

        if "owned_by" in model_info:
            details["owned_by"] = model_info["owned_by"]
        if "object" in model_info:
            details["object"] = model_info["object"]

        # Canonical (snake_case) metadata required by the platform validator
        # for LLM offerings.  Both keys must be present; null asserts
        # "unknown".  Closed-source models will report parameter_count as
        # null per the canonical helper.  metadata_sources records
        # provenance so reviewers can triage stale-value reports.
        canonical = ModelDataLookup.get_canonical_metadata(
            model_id,
            fetcher=self.data_fetcher,
        )
        details["context_length"] = canonical["context_length"]
        details["parameter_count"] = canonical["parameter_count"]
        if canonical["sources"]:
            details["metadata_sources"] = canonical["sources"]

        # Extract upstream pricing for description, but set prices to 0 for BYOK.
        #
        # `pricing_note` is the bare rate card — no "Service provider charges"
        # prefix, because the copy that consumes it already names the biller.
        # It is a param in its own right so the templates can place it: the
        # listing cell puts it behind the `|` of the price-description grammar
        # (unitysvc/unitysvc#1886) and the offering description states it in
        # prose. Do NOT fold it back into `pricing["description"]` — that dict
        # feeds `payout_price` too, which is seller-facing and stays as it is.
        pricing = None
        pricing_note = None
        if model_data:
            if "input_cost_per_token" in model_data and "output_cost_per_token" in model_data:
                input_price = round(float(
                    model_data["input_cost_per_token"]) * 1_000_000, 4)
                output_price = round(float(
                    model_data["output_cost_per_token"]) * 1_000_000, 4)
                pricing_note = (
                    f"${self._format_price(input_price)} / "
                    f"${self._format_price(output_price)} "
                    f"per 1M input/output tokens"
                )
                pricing = {
                    "type": "one_million_tokens",
                    "input": "0",
                    "output": "0",
                    "description": f"Service provider charges {pricing_note}",
                }
                # Include cached_input if available
                if "cache_read_input_token_cost" in model_data:
                    cached_price = round(float(
                        model_data["cache_read_input_token_cost"]) * 1_000_000, 4)
                    pricing["cached_input"] = "0"
                    pricing_note = (
                        f"${self._format_price(input_price)} / "
                        f"${self._format_price(output_price)} / "
                        f"${self._format_price(cached_price)} "
                        f"per 1M input/output/cached tokens"
                    )
                    pricing["description"] = f"Service provider charges {pricing_note}"

        return {
            # Folder path under specs/ == listing.name == "<provider>/<model_id>"
            # (flat layout, #1263). populate_from_iterator preserves the slash.
            "name": f"{PROVIDER_NAME}/{model_id}",
            # Offering name is the bare upstream model_id
            "offering_name": model_id,
            # Offering fields
            "display_name": display_name,
            "description": f"{display_name} language model",
            "service_type": service_type,
            # Chat-only docs (the OpenAI/Anthropic examples, function calling)
            # are wrong for a transcription/TTS/classification model, and the
            # gateway + preset coverage for those modalities is incomplete
            # (no connectivity preset for tts/classification, /v1/audio/speech
            # unproven). Keep them out of the published catalog as drafts until
            # that lands — see unitysvc/unitysvc#1781.
            "status": "draft" if modality else "ready",
            "modality": modality,
            # Only some Groq chat models advertise tool use; the function-calling
            # example 400s on the ones that do not.
            "supports_tools": "tools" in (model_info.get("supported_features") or []),
            "details": details,
            "payout_price": pricing,
            # Bare upstream rate card, placed by the templates (listing price
            # cell's hover note + the offering description).
            "pricing_note": pricing_note,
            # Listing fields
            "list_price": pricing,
            # Provider config (for templates)
            "provider_name": PROVIDER_NAME,
            "provider_display_name": PROVIDER_DISPLAY_NAME,
            "api_base_url": API_BASE_URL,
            "env_api_key_name": ENV_API_KEY_NAME,
        }

    def _modality(self, model_id: str, model_info: dict) -> str | None:
        """Return the non-chat modality of a model, or None when it is a chat model."""
        outputs = set(model_info.get("output_modalities") or [])
        inputs = set(model_info.get("input_modalities") or [])
        if "transcription" in outputs or "audio" in inputs:
            return "transcription"
        if "speech" in outputs:
            return "tts"
        # Classification models carry no distinguishing metadata field — they
        # simply reject a normal chat request. They are marked via per-model
        # <name>.override.json companions ({"parameters": {"modality":
        # "classification", "status": "draft"}}), merged at render time, so
        # this script never changes for one.
        return None

    def _determine_service_type(self, model_id: str) -> str:
        model_lower = model_id.lower()
        if any(kw in model_lower for kw in ["embed", "embedding"]):
            return "embedding"
        if any(kw in model_lower for kw in ["rerank"]):
            return "rerank"
        if any(kw in model_lower for kw in ["vision"]):
            return "vision_language_model"
        return "llm"

    def _format_price(self, price: float) -> str:
        """Format price without trailing .0 for whole numbers."""
        if price == int(price):
            return str(int(price))
        return str(price)


def main():
    api_key = os.environ.get(ENV_API_KEY_NAME)
    if not api_key:
        print(f"Error: {ENV_API_KEY_NAME} not set")
        sys.exit(1)

    source = ModelSource(api_key)
    write_params_from_iterator(
        iterator=source.iter_models(),
        output_dir=SCRIPT_DIR.parent / "specs",
    )


if __name__ == "__main__":
    main()
