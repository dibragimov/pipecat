#
# Copyright (c) 2024-2026, Beeline Uzbekistan
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""UzbekTextNormalizerFilter: unit behaviour, and inside a real TTS pipeline."""

import unittest

from pipecat.utils.text import uzbek_normalizer
from pipecat.utils.text.uzbek_normalizer import detect_language, normalize
from pipecat.utils.text.uzbek_text_filter import UzbekTextNormalizerFilter


class TestNormalizer(unittest.TestCase):
    def test_telecom_and_banking(self):
        cases = {
            "Sizning balansingiz 25 000 so'm.": "Sizning balansingiz yigirma besh ming so'm.",
            "Aloqa raqamingiz: +998 90 123 45 67.":
                "Aloqa raqamingiz: plyus to'qqiz yuz to'qson sakkiz, to'qson, bir yuz yigirma uch, qirq besh, oltmish yetti.",
            "Tasdiqlash kodi: 482915.": "Tasdiqlash kodi: to'rt, sakkiz, ikki, to'qqiz, bir, besh.",
            "Paket 15-noyabrgacha amal qiladi.": "Paket o'n beshinchi noyabrgacha amal qiladi.",
            "Omonat bo'yicha daromad yillik 22,5% ni tashkil etadi.":
                "Omonat bo'yicha daromad yillik yigirma ikki butun o'ndan besh foizni tashkil etadi.",
            "Balansni tekshirish uchun *102# raqamini tering.":
                "Balansni tekshirish uchun yulduzcha, bir, nol, ikki, panjara raqamini tering.",
            "Abonent to'lovi 79 000 so'm/oy.": "Abonent to'lovi oyiga yetmish to'qqiz ming so'm.",
        }
        for raw, want in cases.items():
            self.assertEqual(normalize(raw), want)

    def test_language_guard(self):
        self.assertEqual(detect_language("Ваш баланс 25 000 сумов."), "ru")
        self.assertEqual(detect_language("Your balance is 25,000 sum. Thank you!"), "en")
        self.assertEqual(detect_language("Sizga yordam bera olaman. OK, tushunarli."), "uz")
        self.assertEqual(detect_language("Raqamingiz 901234567."), "uz")

    def test_idempotent(self):
        once = normalize("To'lov 14.10.2026, soat 14:35 da, 50 000 so'm.")
        self.assertEqual(normalize(once), once)
        self.assertTrue(uzbek_normalizer.VERSION.startswith("uznorm-"))


class TestFilter(unittest.IsolatedAsyncioTestCase):
    async def test_filter_rules_and_settings(self):
        f = UzbekTextNormalizerFilter()
        self.assertEqual(await f.filter(" Kod: 4821. "), " Kod: to'rt, sakkiz, ikki, bir. ")  # spaces kept
        self.assertEqual(await f.filter("Your balance is 25,000 sum. Thank you!"),
                         "Your balance is 25,000 sum. Thank you!")
        self.assertEqual(await f.filter("Ваш баланс 25 000 сумов."), "Ваш баланс 25 000 сумов.")
        self.assertEqual(await f.filter("   "), "   ")
        await f.update_settings({"enabled": False})
        self.assertEqual(await f.filter("Balans 25 000 so'm."), "Balans 25 000 so'm.")
        await f.update_settings({"enabled": True})
        self.assertEqual(await f.filter("Balans 25 000 so'm."), "Balans yigirma besh ming so'm.")
        await f.handle_interruption()
        await f.reset_interruption()

    async def test_rule_failure_falls_back_to_original(self):
        import pipecat.utils.text.uzbek_text_filter as mod
        original = mod.normalize
        mod.normalize = lambda text: (_ for _ in ()).throw(ValueError("bad rule"))
        try:
            self.assertEqual(await UzbekTextNormalizerFilter().filter("Balans 25 000 so'm."), "Balans 25 000 so'm.")
        finally:
            mod.normalize = original


class TestInPipeline(unittest.IsolatedAsyncioTestCase):
    """A real TTS service with the filter, through Pipecat's test runner."""

    async def test_yandex_tts_receives_words_and_transcript_shows_them(self):
        from aiohttp import web

        from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame, TTSTextFrame
        from pipecat.services.yandex.tts import YandexTTSService
        from pipecat.tests.utils import run_test

        received = []

        async def synth(request):
            received.append((await request.post())["text"])
            resp = web.StreamResponse()
            await resp.prepare(request)
            await resp.write(b"\x00\x10" * 3200)
            await resp.write_eof()
            return resp

        app = web.Application()
        app.router.add_post("/tts", synth)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            tts = YandexTTSService(api_key="k", folder_id="f", language="uz-UZ", voice="nigora",
                                   endpoint=f"http://127.0.0.1:{port}/tts",
                                   text_filters=[UzbekTextNormalizerFilter()])
            down, _ = await run_test(tts, frames_to_send=[TTSSpeakFrame(text="Balansingiz 25 000 so'm.")])
        finally:
            await runner.cleanup()
        self.assertEqual(received, ["Balansingiz yigirma besh ming so'm."])           # what Yandex speaks
        self.assertTrue(any(isinstance(f, TTSAudioRawFrame) for f in down))
        spoken_text = " ".join(f.text for f in down if isinstance(f, TTSTextFrame))
        # Pipecat reports the filtered text as spoken (transcript and LLM context). If a future
        # Pipecat keeps the original text instead, this assertion flags it.
        self.assertIn("yigirma besh ming", spoken_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
