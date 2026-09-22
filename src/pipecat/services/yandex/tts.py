#
# Copyright (c) 2024-2026, Beeline Uzbekistan
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Yandex SpeechKit v1 text-to-speech service implementation.

Uses the v1 REST synthesize endpoint requesting raw LPCM output directly,
rather than the v3 gRPC streaming API - this avoids needing an Opus decoder
in the pipeline, and REST proved simpler to validate(the v3 TTS gRPC
stubs require generating protobuf bindings from Yandex's proto sources;
REST does not).

uz-UZ synthesis is not in Yandex's official documented voice list,
but was confirmed working end to end during Phase 0 testing with
default voice.

Per-utterance language detection (see _detect_utterance_language) picks a
matching voice for each response text independently, rather than locking
the whole call to one language/voice - a Russian-language voice reading
an English sentence phonetically sounds as wrong as the reverse, so this
runs on every run_tts() call rather than once at construction time.
"""

import re
from typing import AsyncGenerator, AsyncIterator, Optional

import aiohttp
from loguru import logger

from pipecat.frames.frames import CancelFrame, EndFrame, ErrorFrame, Frame, StartFrame
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language

try:
    from langdetect import DetectorFactory
    from langdetect import detect as _langdetect_detect

    DetectorFactory.seed = 0  # deterministic results across calls
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False
    logger.warning(
        "langdetect not installed - per-utterance TTS language/voice "
        "switching will fall back to the configured default language for "
        "every utterance. Add `langdetect` to the yandex pyproject.toml "
        "extra and the Dockerfile's install list to enable it."
    )

_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def _detect_utterance_language(text: str, default: str) -> str:
    """Best-effort per-utterance language detection for picking a matching
    Yandex voice. Falls back to `default` when the text is too short to
    detect reliably, or when langdetect isn't installed.

    Cyrillic is checked first and unconditionally - it's an unambiguous
    signal for Russian in our language set, and Uzbek is overwhelmingly
    written in Latin script today, so no separate library call is needed
    for that case. Latin-script text goes through langdetect to
    distinguish Uzbek from English, which script alone can't do.
    """
    stripped = text.strip()
    if len(stripped) < 4:
        return default

    if _CYRILLIC_RE.search(stripped):
        return "ru-RU"

    if not _HAS_LANGDETECT:
        return default

    try:
        code = _langdetect_detect(stripped)
    except Exception:
        return default

    return {"uz": "uz-UZ", "en": "en-US", "ru": "ru-RU"}.get(code, default)


LANGUAGE_MAP = {
    Language.UZ: "uz-UZ",
    Language.RU: "ru-RU",
    Language.EN: "en-US",
}

# nigora is the one uz-UZ voice confirmed compatible with the v1 REST API
# this service uses (Yandex's current AI Studio docs, cross-checked
# 2026-09-09: https://aistudio.yandex.ru/en/docs/speechkit/tts/voices).
# Two other real uz-UZ voices exist - zamira and yulduz, both with role
# variants (strict/friendly/whisper) - but both are v3-API-only. Using them
# would mean implementing the v3 gRPC synthesis path instead of this REST
# endpoint - a real follow-up, not something to bolt on here.
DEFAULT_VOICE = {
    "ru-RU": "alena",
    "en-US": "john",
    "uz-UZ": "nigora",
}


class YandexTTSService(TTSService):
    """Text-to-speech using Yandex SpeechKit's v1 synthesize REST API.

    Args:
        api_key: Yandex Cloud API key (Authorization: Api-Key <key>).
        folder_id: Yandex Cloud folder ID.
        language: Synthesis language. One of Language.UZ, Language.RU,
            Language.EN - extend LANGUAGE_MAP for others.
        voice: Explicit voice name. If None, uses DEFAULT_VOICE for the
            resolved language (and Yandex's own default for uz-UZ, since
            no named voice is documented).
        sample_rate: Output sample rate in Hz, requested directly from
            Yandex via sampleRateHertz (no resampling needed downstream).
        endpoint: Yandex TTS REST endpoint.
        **kwargs: Additional arguments passed to TTSService.
    """

    def __init__(
        self,
        *,
        api_key: str,
        folder_id: str,
        language: str = "ru-RU",
        voice: Optional[str] = None,
        sample_rate: int = 16000,
        endpoint: str = "https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize",
        **kwargs,
    ):
        super().__init__(
            sample_rate=sample_rate,
            settings=TTSSettings(model="general", voice=voice, language=language),
            **kwargs,
        )
        self._api_key = api_key
        self._folder_id = folder_id
        self._language = language
        self._voice = voice
        self._endpoint = endpoint
        self._session: Optional[aiohttp.ClientSession] = None

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return LANGUAGE_MAP.get(language)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._session = aiohttp.ClientSession()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        if self._session:
            await self._session.close()
            self._session = None

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        if self._session:
            await self._session.close()
            self._session = None

    async def run_tts(
        self, text: str, context_id: Optional[str] = None
    ) -> AsyncGenerator[Frame, None]:
        """Modeled directly on Piper's confirmed non-websocket TTS pattern:
        start_tts_usage_metrics/stop_ttfb_metrics are called explicitly, but
        TTSStartedFrame/TTSStoppedFrame are NOT yielded here - the base
        class handles those automatically around this call.
        """
        configured_lang = self._language or "ru-RU"
        lang_code = _detect_utterance_language(text, configured_lang)

        # An explicit voice override only applies to the configured "home"
        # language - otherwise a forced uz-UZ voice would speak detected
        # English text phonetically, the exact problem this is fixing.
        if lang_code == configured_lang and self._voice:
            voice = self._voice
        else:
            voice = DEFAULT_VOICE.get(lang_code)

        data = {
            "text": text,
            "lang": lang_code,
            "folderId": self._folder_id,
            "format": "lpcm",
            "sampleRateHertz": str(self.sample_rate),
        }
        if voice:
            data["voice"] = voice
        headers = {"Authorization": f"Api-Key {self._api_key}"}

        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            await self.start_tts_usage_metrics(text)

            async def audio_iterator() -> AsyncIterator[bytes]:
                async with self._session.post(
                    self._endpoint, headers=headers, data=data
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise RuntimeError(
                            f"Yandex TTS error {resp.status}: {error_text}"
                        )
                    async for chunk in resp.content.iter_chunked(4096):
                        if chunk:
                            yield chunk

            async for frame in self._stream_audio_frames_from_iterator(
                audio_iterator(), in_sample_rate=self.sample_rate, context_id=context_id
            ):
                await self.stop_ttfb_metrics()
                yield frame
        except Exception as e:
            logger.error(f"{self} exception: {e}")
            yield ErrorFrame(error=f"Yandex TTS error: {e}")
        finally:
            logger.debug(f"{self}: Finished TTS [{text}]")
            await self.stop_ttfb_metrics()
