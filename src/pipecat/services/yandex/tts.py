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

uz-UZ synthesis uses the nigora voice (see DEFAULT_VOICE), confirmed
working end to end in live Web Calls.

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
    from langdetect import detect_langs as _langdetect_detect_langs

    DetectorFactory.seed = 0  # deterministic results across calls
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False
    logger.warning(
        "langdetect not installed - English text with no common English "
        "words will be voiced as Uzbek. Add `langdetect` to the yandex "
        "pyproject.toml extra and the Dockerfile's install list."
    )

HTTP_TIMEOUT = aiohttp.ClientTimeout(total=15, sock_connect=5)

_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
# Letters of the Uzbek Cyrillic alphabet that Russian does not use.
_UZ_CYRILLIC_RE = re.compile(r"[ЎўҚқҒғҲҳ]")
# Every apostrophe-like character used for Uzbek oʻ/gʻ and the tutuq belgisi.
_APOSTROPHES_RE = re.compile(r"[ʻʼ‘’`´]")
_LATIN_WORD_RE = re.compile(r"[a-z']+")

# Function words and stock call-center phrases. Words that are also real
# Uzbek words are deliberately left out of the English list ("men", "on",
# "it", "had", "or"), so a hit here is a genuine English signal.
_EN_WORDS = frozenset("""
    the an and is are was were be been to of in for with at by from this that
    these those you your we our i me my he she they them their what which who
    how why when where can could will would should do does did have has not
    please thank thanks hello hi yes sorry help there here about if but just
    okay ok hold call calling today
    one moment good morning evening afternoon wait sure let look
""".split())
_UZ_WORDS = frozenset("""
    va bu shu u men siz sizga sizni sizning biz bilan uchun emas bor yo'q ha
    kerak mumkin iltimos rahmat assalomu alaykum salom qanday nima qaysi
    qachon yana ham lekin agar bo'ladi bo'lsa edi endi hozir xizmat raqam
    so'm mening ular o'z deb juda yaxshi keyin faqat hech narsa kechirasiz
    tushundim xayr xush kelibsiz albatta tushunarli kuting bemalol
""".split())


def _is_uzbek_latin_word(word: str) -> bool:
    """Spelling cues that English practically never produces: oʻ/gʻ followed
    by a letter (but not the English possessive "dog's"), q not followed by
    u, and the 2nd-person-polite suffix -ingiz."""
    if word in _UZ_WORDS:
        return True
    if re.search(r"[og]'[a-z]", word) and not word.endswith("'s"):
        return True
    return bool(re.search(r"q(?!u)", word) or "ngiz" in word or word.startswith("x"))


def _detect_utterance_language(text: str, default: str) -> str:
    """Best-effort per-utterance language detection for picking a matching
    Yandex voice, among uz-UZ / ru-RU / en-US.

    langdetect has no Uzbek model (it labels Uzbek as so/id/lt/it...), so it
    cannot tell Uzbek from English on its own. Instead:
      - Cyrillic with Uzbek-only letters (ў қ ғ ҳ) -> uz-UZ.
      - Other text containing Cyrillic -> ru-RU, unless its Latin words are
        clearly Uzbek (an Uzbek cue, or Latin letters > 2x Cyrillic):
        "Sizga «Супер» tarifi mos keladi." -> uz-UZ, while
        "Ваш тариф Beeline Smart." -> ru-RU. Cyrillic Uzbek without
        ў қ ғ ҳ still goes to ru-RU, so the agent prompt should require Latin Uzbek.
      - Latin -> en-US only with at least two English function words that
        outnumber the Uzbek cues (or a one-word English reply), so
        "OK, tushunarli." stays Uzbek. Domain words that are also loanwords in
        Uzbek (balance, account, support...) are deliberately not English cues.
        langdetect is consulted only for Latin text with no cues either way,
        and only to confirm English.
    Too-short text keeps `default`.
    """
    stripped = text.strip()
    if len(stripped) < 4:
        return default

    if _CYRILLIC_RE.search(stripped):
        if _UZ_CYRILLIC_RE.search(stripped):
            return "uz-UZ"                      # Uzbek Cyrillic letters
        latin_words = _LATIN_WORD_RE.findall(_APOSTROPHES_RE.sub("'", stripped.lower()))
        cyr = len(_CYRILLIC_RE.findall(stripped))
        lat = sum(len(w) for w in latin_words)
        # Mixed script: Uzbek only if the Latin part is clearly Uzbek, e.g.
        # "Sizga «Супер» tarifi mos keladi." -> uz; "Ваш тариф Beeline Smart." -> ru
        if not latin_words or not (any(_is_uzbek_latin_word(w) for w in latin_words) or lat > 2 * cyr):
            return "ru-RU"

    # Latin script: the choice is Uzbek or English.
    latin_default = default if default in ("uz-UZ", "en-US") else "uz-UZ"
    words = _LATIN_WORD_RE.findall(_APOSTROPHES_RE.sub("'", stripped.lower()))
    if not words:
        return latin_default

    en_hits = sum(w in _EN_WORDS for w in words)
    uz_hits = sum(_is_uzbek_latin_word(w) for w in words)
    if en_hits > uz_hits and (en_hits >= 2 or len(words) == 1):
        return "en-US"
    if uz_hits:
        return "uz-UZ"

    if _HAS_LANGDETECT and len(stripped) >= 10:
        try:
            best = _langdetect_detect_langs(stripped)[0]
            if best.lang == "en" and best.prob >= 0.95:
                return "en-US"
        except Exception:
            pass
    return latin_default


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
        voice: Explicit voice name for the configured language. If None,
            or when an utterance is detected in another language, uses
            DEFAULT_VOICE for that language.
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
        # aiohttp's default total timeout is 5 minutes: a hung Yandex call would stall the turn.
        self._session = aiohttp.ClientSession(timeout=HTTP_TIMEOUT)

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

        # Never log the text itself: it can contain customer data.
        logger.debug(f"{self}: Generating TTS ({lang_code}, {voice}, {len(text)} chars)")

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
            logger.debug(f"{self}: Finished TTS ({lang_code}, {len(text)} chars)")
            await self.stop_ttfb_metrics()
