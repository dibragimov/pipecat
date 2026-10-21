#
# Copyright (c) 2024-2026, Beeline Uzbekistan
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Yandex SpeechKit v3 streaming speech-to-text service implementation.

Yandex's streaming Recognizer API is bidirectional gRPC, not a WebSocket -
every existing cloud STT provider in Pipecat (Deepgram, Soniox, AWS
Transcribe) is WebSocket-based, so this subclasses STTService directly
rather than WebsocketSTTService. That means this service is responsible for
its own connection lifecycle (opening the gRPC stream on StartFrame, tearing
it down on EndFrame/CancelFrame) rather than getting that for free from a
websocket mixin.

Protocol details (partial/final/eou_update field shapes, language codes,
audio format) were confirmed against the live API during standalone testing
before this integration was written.
"""

import asyncio
from typing import AsyncGenerator, Optional

import grpc
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
    import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc
except ModuleNotFoundError as e:
    logger.error(
        "In order to use Yandex STT, you need `pip install yandexcloud`. "
        "If that doesn't expose yandex.cloud.ai.stt.v3.stt_pb2 / "
        "stt_service_pb2_grpc, generate the stubs manually from "
        "https://github.com/yandex-cloud/cloudapi with grpc_tools.protoc "
    )
    raise Exception(f"Missing module: {e}")


# Extend as needed - these three are what Phase 0 validated end to end.
LANGUAGE_MAP = {
    Language.UZ: "uz-UZ",
    Language.RU: "ru-RU",
    Language.EN: "en-US",
}

# The languages we want to auto-detect, per-utterance,
# instead of locking the whole call to one.
AUTO_DETECT_LANGUAGES = ["uz-UZ", "ru-RU", "en-US"]

# Alternative languages[] entries use ISO 639-1.
_ISO639_TO_LOCALE = {"uz": "uz-UZ", "ru": "ru-RU", "en": "en-US"}


def _detected_language(alternative) -> Optional[str]:
    """Best guess at which language this specific utterance was actually
    spoken in, from Yandex's own per-utterance language distribution
    (Alternative.languages[], confirmed field path). Returns None if the
    field is empty or maps to a language outside AUTO_DETECT_LANGUAGES.
    """
    try:
        best = max(alternative.languages, key=lambda entry: entry.probability)
    except ValueError:
        return None
    return _ISO639_TO_LOCALE.get(best.language_code)


class YandexSTTService(STTService):
    """Real-time speech-to-text using Yandex SpeechKit's v3 streaming API.

    Args:
        api_key: Yandex Cloud API key (Authorization: Api-Key <key>).
        folder_id: Yandex Cloud folder ID (sent as x-folder-id metadata).
        language: Recognition language. One of Language.UZ, Language.RU,
            Language.EN - extend LANGUAGE_MAP for others.
        sample_rate: Audio sample rate in Hz. If None, uses the pipeline's
            configured rate.
        endpoint: Yandex STT gRPC endpoint. Defaults to the public one.
        **kwargs: Additional arguments passed to STTService.
    """

    def __init__(
        self,
        *,
        api_key: str,
        folder_id: str,
        language: str = "ru-RU",
        sample_rate: Optional[int] = None,
        endpoint: str = "stt.api.cloud.yandex.net:443",
        **kwargs,
    ):
        # Populate settings explicitly - the base class logs a "NOT_GIVEN"
        # warning (confirmed for the TTS side in production logs) if a
        # subclass never passes these through.
        super().__init__(
            sample_rate=sample_rate,
            settings=STTSettings(model="general", language=language),
            **kwargs,
        )
        self._api_key = api_key
        self._folder_id = folder_id
        self._language = language
        self._endpoint = endpoint

        self._channel: Optional[grpc.aio.Channel] = None
        self._call = None
        self._send_queue: Optional[asyncio.Queue] = None
        self._recv_queue: Optional[asyncio.Queue] = None
        self._receive_task: Optional[asyncio.Task] = None

    def language_to_service_language(self, language: Language) -> Optional[str]:
        """Map a Pipecat Language to Yandex's language code. Returns None
        (unsupported) for anything not in LANGUAGE_MAP.
        """
        return LANGUAGE_MAP.get(language)

    async def start(self, frame: StartFrame):
        """Open the gRPC stream when the pipeline starts. Mirrors the
        start/stop/cancel hook pattern STTService already uses for its own
        lifecycle accounting (confirmed via its stop()/cancel() overrides) -
        the idiomatic extension point, rather than intercepting StartFrame
        by hand inside process_frame.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        logger.debug(f"{self}: connecting to Yandex STT ({self._endpoint})")

        creds = grpc.ssl_channel_credentials()
        self._channel = grpc.aio.secure_channel(self._endpoint, creds)
        stub = stt_service_pb2_grpc.RecognizerStub(self._channel)

        self._send_queue = asyncio.Queue()
        self._recv_queue = asyncio.Queue()

        session_options = stt_pb2.StreamingRequest(
            session_options=stt_pb2.StreamingOptions(
                recognition_model=stt_pb2.RecognitionModelOptions(
                    audio_format=stt_pb2.AudioFormatOptions(
                        raw_audio=stt_pb2.RawAudio(
                            audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                            sample_rate_hertz=self.sample_rate,
                            audio_channel_count=1,
                        )
                    ),
                    language_restriction=stt_pb2.LanguageRestrictionOptions(
                        restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                        language_code=AUTO_DETECT_LANGUAGES,
                    ),
                    audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
                )
            )
        )
        await self._send_queue.put(session_options)

        metadata = (
            ("authorization", f"Api-Key {self._api_key}"),
            ("x-folder-id", self._folder_id),
        )
        self._call = stub.RecognizeStreaming(self._request_iterator(), metadata=metadata)
        self._receive_task = self.create_task(self._receive_loop(), name="yandex_stt_receive")

        await self._call_event_handler("on_connected")

    async def _request_iterator(self):
        """Feeds the outgoing side of the gRPC bidi stream from the send
        queue. A None sentinel signals a clean shutdown.
        """
        while True:
            item = await self._send_queue.get()
            if item is None:
                return
            yield item

    async def _receive_loop(self):
        """Runs for the lifetime of the connection, converting Yandex
        responses into queued Pipecat frames as they arrive - independently
        of when run_stt() next gets called to pick them up.
        """
        try:
            async for response in self._call:
                await self._handle_response(response)
        except grpc.aio.AioRpcError as e:
            logger.error(f"{self}: Yandex STT stream error: {e.code()} {e.details()}")
            await self._call_event_handler("on_connection_error", str(e))
            # push_error takes the message string (it builds the ErrorFrame itself)
            await self.push_error(f"Yandex STT error: {e.details()}")
        except asyncio.CancelledError:
            pass

    async def _handle_response(self, response):
        if response.HasField("partial"):
            try:
                alt = response.partial.alternatives[0]
                text = alt.text
                if text:
                    lang = _detected_language(alt) or self._language
                    await self._recv_queue.put(
                        InterimTranscriptionFrame(
                            text, self._user_id, time_now_iso8601(), lang
                        )
                    )
            except (IndexError, AttributeError):
                pass
        elif response.HasField("final"):
            try:
                alt = response.final.alternatives[0]
                text = alt.text
                if text:
                    lang = _detected_language(alt) or self._language
                    self.request_finalize()
                    self.confirm_finalize()
                    await self._recv_queue.put(
                        TranscriptionFrame(
                            text, self._user_id, time_now_iso8601(), lang
                        )
                    )
            except (IndexError, AttributeError):
                pass

    async def _disconnect(self):
        if self._send_queue is not None:
            await self._send_queue.put(None)
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._channel:
            await self._channel.close()
            self._channel = None
        await self._call_event_handler("on_disconnected")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Called once per incoming audio chunk by the base class. Sends the
        chunk into the open gRPC stream, then drains and yields whatever
        transcripts have arrived since the last call - these are not 1:1
        with the chunk just sent, since Yandex's responses arrive out of
        band on their own schedule.
        """
        if self._send_queue is None:
            # StartFrame hasn't connected us yet - drop rather than crash.
            return

        await self._send_queue.put(
            stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=audio))
        )

        while not self._recv_queue.empty():
            yield self._recv_queue.get_nowait()
