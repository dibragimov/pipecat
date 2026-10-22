#
# Copyright (c) 2024-2026, Beeline Uzbekistan
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Text filter that makes TTS read Uzbek numbers, dates, codes and abbreviations correctly.

Each sentence is checked first: Uzbek sentences are normalized (digits, symbols and
abbreviations become spoken words); Russian and English sentences pass through unchanged,
so the TTS provider's own handling of those languages is kept.

    UzbekTextNormalizerFilter()  ->  "Balansingiz 25 000 so'm."  ->  "Balansingiz yigirma besh ming so'm."

Use it in a TTS service's ``text_filters``. Note: Pipecat builds the spoken-text frames from the
filtered text, so call transcripts and the assistant's turns in the LLM context also show the
normalized words ("yigirma besh ming so'm" rather than "25 000 so'm"). Verified in
tests/test_uzbek_text_filter.py.
"""

from typing import Any, Mapping

from loguru import logger

from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.text.uzbek_normalizer import VERSION, detect_language, normalize


class UzbekTextNormalizerFilter(BaseTextFilter):
    """Normalize Uzbek sentences for speech; leave other languages untouched."""

    def __init__(self, *, enabled: bool = True):
        self._enabled = enabled

    @property
    def version(self) -> str:
        return VERSION

    async def update_settings(self, settings: Mapping[str, Any]):
        if "enabled" in settings:
            self._enabled = bool(settings["enabled"])

    async def filter(self, text: str) -> str:
        if not self._enabled or not text.strip() or detect_language(text) != "uz":
            return text
        try:
            core = normalize(text)
        except Exception as exc:  # never break speech because of a normalization rule
            logger.warning(f"Uzbek normalization failed, speaking the original text: {type(exc).__name__}: {exc}")
            return text
        # Keep surrounding whitespace: sentences are joined by the TTS aggregator.
        lead = text[: len(text) - len(text.lstrip())]
        trail = text[len(text.rstrip()):]
        return f"{lead}{core}{trail}"

    async def handle_interruption(self):
        pass

    async def reset_interruption(self):
        pass
