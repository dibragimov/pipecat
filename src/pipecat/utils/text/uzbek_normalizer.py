#
# Copyright (c) 2024-2026, Beeline Uzbekistan
#
# SPDX-License-Identifier: BSD 2-Clause License
#
"""Uzbek text normalization for TTS: digits, symbols and abbreviations -> spoken words.

Self-contained (standard library only). The same code (same VERSION) is used by the
tts-eval suite, where it was measured, and by UzbekTextNormalizerFilter in this package. Rules run in a fixed order;
the most specific patterns (e-mail, USSD, phone, dates, times) come before plain numbers.

    normalize("Balansingiz 25 000 so'm.")  ->  "Balansingiz yigirma besh ming so'm."

Russian sentences are returned unchanged (only markdown/emoji cleanup), so a provider's
own Russian handling is kept. Latin is the output script; Uzbek Cyrillic is transliterated.
"""
from __future__ import annotations

import re

VERSION = "uznorm-1.0"

ONES = ["nol", "bir", "ikki", "uch", "to'rt", "besh", "olti", "yetti", "sakkiz", "to'qqiz"]
TENS = {1: "o'n", 2: "yigirma", 3: "o'ttiz", 4: "qirq", 5: "ellik", 6: "oltmish", 7: "yetmish",
        8: "sakson", 9: "to'qson"}
SCALES = ["", "ming", "million", "milliard", "trillion"]
MONTHS = ["yanvar", "fevral", "mart", "aprel", "may", "iyun", "iyul", "avgust", "sentabr", "oktabr",
          "noyabr", "dekabr"]
MONTH_RE = r"(?:yanvar|fevral|mart|aprel|may|iyun|iyul|avgust|sent[ay]abr|okt[ay]abr|noyabr|dekabr)"
# Suffixes that may be written apart from a number ("9:00 dan", "22,5% ni").
SUFFIX = r"(?:[ -]?(?P<suf>gacha|dagi|dan|ning|ini|ga|da|ni|si|i)\b)?"

# Domain lexicon: written form -> how it should be read. Matched as whole words
# (case-sensitive unless marked), also as the first part of hyphen compounds ("SMS-kodni").
LEXICON = [
    ("JShShIR", "je-she-she-er"), ("STIR", "stir"), ("SMS", "es-em-es"), ("CVV", "si-vi-vi"),
    ("PUK", "puk"), ("PIN", "pin"), ("OTP", "o-ti-pi"), ("LTE", "el-ti-i"), ("QR", "kyu-ar"),
    ("eSIM", "e-sim"), ("SIM", "sim"), ("P2P", "pi-tu-pi"), ("MyID", "may-ay-di"), ("USSD", "yu-es-es-di"),
    ("Wi-Fi", "vay-fay"), ("WiFi", "vay-fay"), ("Beeline", "Bilayn"), ("beeline", "bilayn"),
    ("Uzcard", "Uzkard"), ("Visa", "Viza"), ("Click", "Klik"), ("Payme", "Peymi"),
    ("App Store", "Epp Stor"), ("Google Play", "Gugl Pley"), ("online", "onlayn"), ("Online", "Onlayn"),
    ("offline", "oflayn"), ("WhatsApp", "Vatsap"), ("Telegram", "Telegram"),
    ("4G", "to'rt ji"), ("5G", "besh ji"), ("3G", "uch ji"), ("2G", "ikki ji"),
]
KEEP_UPPER = {"AQSH", "BMT", "MDH", "SMS", "CVV", "PUK", "PIN", "OTP", "LTE", "QR", "SIM", "STIR", "USSD", "ID"}
UNITS = {"GB": "gigabayt", "MB": "megabayt", "KB": "kilobayt", "TB": "terabayt", "Gb": "gigabayt",
         "Mb": "megabayt", "mln": "million", "mlrd": "milliard", "km": "kilometr", "kg": "kilogramm"}
SPEED_UNITS = {"Mbit/s": "megabit", "Mbps": "megabit", "Mbit/sek": "megabit", "Gbit/s": "gigabit",
               "Kbit/s": "kilobit"}
CURRENCY_SYMBOLS = {"$": "dollar", "€": "yevro", "₽": "rubl", "£": "funt"}
PER_UNITS = {"oy": "oyiga", "kun": "kuniga", "yil": "yiliga", "hafta": "haftasiga", "soat": "soatiga",
             "daq": "daqiqasiga", "daqiqa": "daqiqasiga"}

# Uzbek Cyrillic (and Russian letters inside Uzbek text) -> Latin
CYR = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "j", "з": "z", "и": "i",
       "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
       "у": "u", "ф": "f", "х": "x", "ц": "s", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "'", "ы": "i", "ь": "",
       "э": "e", "ю": "yu", "я": "ya", "ў": "o'", "қ": "q", "ғ": "g'", "ҳ": "h"}
_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_UZ_CYRILLIC = re.compile(r"[ЎўҚқҒғҲҳ]")
_LATIN = re.compile(r"[A-Za-z]")
_APOS = re.compile(r"(?<=\w)[ʻʼ‘’`´](?=\w)")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\u2B00-\u2BFF\uFE0E\uFE0F\u200D\U000E0020-\U000E007F]+")


# ---------------------------------------------------------------- numbers
def _below_thousand(n: int, scale_index: int) -> list[str]:
    words = []
    h, rest = divmod(n, 100)
    if h:
        # "yuz" alone for an exact hundred (yuz, yuz ming); "bir yuz ..." otherwise.
        words.append("yuz" if (h == 1 and rest == 0) else f"{ONES[h]} yuz")
    t, o = divmod(rest, 10)
    if t:
        words.append(TENS[t])
    if o:
        words.append(ONES[o])
    return words


def cardinal(n: int) -> str:
    if n == 0:
        return "nol"
    if n < 0:
        return "minus " + cardinal(-n)
    groups, words = [], []
    while n:
        n, g = divmod(n, 1000)
        groups.append(g)
    for idx in range(len(groups) - 1, -1, -1):
        g = groups[idx]
        if not g:
            continue
        words += _below_thousand(g, idx)
        if SCALES[idx]:
            words.append(SCALES[idx])
    return " ".join(words)


def ordinal_word(word: str) -> str:
    return word + ("nchi" if word[-1] in "aeiou" else "inchi")


def ordinal(n: int) -> str:
    words = cardinal(n).split()
    words[-1] = ordinal_word(words[-1])
    return " ".join(words)


def digits(s: str, sep: str = ", ") -> str:
    return sep.join(ONES[int(c)] for c in s if c.isdigit())


def group_words(g: str) -> str:
    """A digit group in a phone/account number: leading zeros are spoken ('05' -> 'nol besh')."""
    zeros = len(g) - len(g.lstrip("0"))
    rest = g.lstrip("0")
    parts = ["nol"] * zeros + ([cardinal(int(rest))] if rest else [])
    return " ".join(parts) if parts else "nol"


def decimal(int_part: str, frac: str) -> str:
    """'2','5' -> 'ikki butun o'ndan besh'; '1','25' -> 'bir butun yuzdan yigirma besh'."""
    whole = cardinal(int(int_part or "0"))
    frac = frac.rstrip("0") or "0"
    denom = {1: "o'ndan", 2: "yuzdan", 3: "mingdan"}.get(len(frac))
    if denom is None:
        return f"{whole} butun {digits(frac, ' ')}"
    return f"{whole} butun {denom} {cardinal(int(frac))}"


def attach(word_phrase: str, suffix: str | None) -> str:
    """Join a suffix to the last word with Uzbek consonant harmony (ellik+ga -> ellikka)."""
    if not suffix:
        return word_phrase
    head, _, last = word_phrase.rpartition(" ")
    if suffix.startswith("g") and last.endswith("k"):
        suffix = "k" + suffix[1:]
    elif suffix.startswith("g") and last.endswith("q"):
        suffix = "q" + suffix[1:]
    elif suffix in ("i", "ini") and last[-1] in "aeiou":
        suffix = "si"
    elif suffix == "si" and last[-1] not in "aeiou":
        suffix = "i"
    return (head + " " if head else "") + last + suffix


def _num(s: str) -> int:
    return int(re.sub(r"[ \u00a0\u202f.]", "", s))


NUM = r"\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+"          # 25 000 / 3 456 789 / 250


# ---------------------------------------------------------------- language & script
def looks_russian(text: str) -> bool:
    cyr = len(_CYRILLIC.findall(text))
    return cyr > 0 and not _UZ_CYRILLIC.search(text) and cyr > len(_LATIN.findall(text))


def transliterate(text: str) -> str:
    out = []
    for i, ch in enumerate(text):
        low = ch.lower()
        if low in CYR:
            lat = CYR[low]
            if low == "е" and (i == 0 or not text[i - 1].isalpha() or text[i - 1].lower() in "аеёиоуэюяўaeiou"):
                lat = "ye"
            if ch.isupper() and lat:
                lat = lat[0].upper() + lat[1:]
            out.append(lat)
        else:
            out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------- cleanup
def cleanup(text: str) -> str:
    t = re.sub(r"\[([^\]\n]+)\]\((?:[^)\s]+)\)", r"\1", text)            # [label](url)
    t = re.sub(r"(\*\*|__|~~)(?=\S)([^*#_~\n]+?)(?<=\S)\1", r"\2", t)      # paired emphasis around text
    t = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]+", "", t)
    t = re.sub(r"(?m)^[ \t]*(?:[-*•▪◦]|\d{1,2}[.)])[ \t]+", "", t)
    t = _EMOJI.sub(" ", t)
    t = re.sub(r"[«»“”„\"]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _caps(t: str) -> str:
    """ALL-CAPS shouting -> normal case (keeps known acronyms)."""
    letters = [c for c in t if c.isalpha()]
    if len(letters) < 8 or sum(c.isupper() for c in letters) / len(letters) < 0.7:
        return t
    words = [w if w.strip(".,!?:;'") in KEEP_UPPER else w.lower() for w in t.split(" ")]
    out = " ".join(words)
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)


# ---------------------------------------------------------------- entity rules
def _lexicon(t: str) -> str:
    for written, spoken in LEXICON:
        t = re.sub(rf"(?<![\w-]){re.escape(written)}(?=$|[^\w]|-)", spoken, t)
    t = re.sub(r"\b(kyu-ar|es-em-es)-(?=kod)", r"\1 ", t)                   # QR-kodni -> kyu-ar kodni
    return t


def _email_url(t: str) -> str:
    def say_domain(d: str) -> str:
        return " nuqta ".join(d.split("."))

    t = re.sub(r"\b([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)\b",
               lambda m: f"{m.group(1).replace('.', ' nuqta ')} kuchukcha {say_domain(m.group(2))}", t)
    t = re.sub(r"\bhttps?://", "", t)
    t = re.sub(r"\b(?:www\.)?([a-z][\w-]*(?:\.[\w-]+)*\.(?:uz|com|ru|net|org|io))\b",
               lambda m: say_domain(m.group(1)), t)
    return t


def _ussd(t: str) -> str:
    def speak(m):
        out = []
        for tok in re.findall(r"\*|#|\d", m.group(0)):
            out.append({"*": "yulduzcha", "#": "panjara"}.get(tok) or ONES[int(tok)])
        return ", ".join(out)
    return re.sub(r"(?<![\w*])\*[\d*]*\d[\d*]*#(?!\w)", speak, t)


def _cards_accounts(t: str) -> str:
    t = re.sub(r"\b(\d{4})(?:\s+\*{4}){1,3}\s+(\d{4})\b",
               lambda m: f"{digits(m.group(1))} ... {digits(m.group(2))}", t)       # 8600 **** **** 4821
    t = re.sub(r"\b\d{4}(?:[ \u00a0]\d{4}){3,4}\b",
               lambda m: ", ".join(group_words(g) for g in m.group(0).split()), t)  # account / card number
    return t


def _phones(t: str) -> str:
    def phone(prefix, groups):
        words = ", ".join(group_words(g) for g in groups)
        return (f"{prefix}, {words}" if prefix else words)

    t = re.sub(r"(?<![\w+])(\+)?998[\s-]*\(?(\d{2})\)?[\s-]*(\d{3})[\s-]*(\d{2})[\s-]*(\d{2})(?!\d)",
               lambda m: phone(("plyus " if m.group(1) else "") + "to'qqiz yuz to'qson sakkiz",
                               m.groups()[1:]), t)
    t = re.sub(r"\((\d{2})\)\s*(\d{3})[\s-]?(\d{2})[\s-]?(\d{2})(?!\d)",
               lambda m: phone("", m.groups()), t)                                   # (71) 200-12-34
    t = re.sub(r"(?<![\d,.])(\d{2})[\s-](\d{3})[\s-](\d{2})[\s-](\d{2})(?![\d,.])",
               lambda m: phone("", m.groups()), t)                                   # 90 123 45 67
    t = re.sub(r"(?<![\d,.])(\d{2})(\d{3})(\d{2})(\d{2})(?![\d,.])(?!\s*(?:so'm|dollar|\$|%))",
               lambda m: phone("", m.groups()), t)                                   # 901234567
    return t


def _codes(t: str) -> str:
    # Digit-by-digit after a code/number keyword: "kodi: 482915", "oxirgi to'rt raqami: 4821"
    # Up to two words may sit between keyword and code: "Kodni kiriting: 482915".
    return re.sub(r"(?i)\b(kod\w*|parol\w*|pin(?:-kod)?\w*|raqami|raqamlari)((?:\s+[a-z']+){0,2})(\s*[:\-]?\s*)(\d{4,8})\b",
                  lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{digits(m.group(4))}", t)


def _dates(t: str) -> str:
    def full(m):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            return m.group(0)
        return f"{ordinal(d)} {MONTHS[mo - 1]}, {ordinal(y)} yil"
    t = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", full, t)                  # 01.12.2026
    t = re.sub(rf"\b(\d{{1,2}})-({MONTH_RE})", lambda m: f"{ordinal(int(m.group(1)))} {m.group(2)}", t)
    t = re.sub(r"\b(\d{4})-(yil\w*)", lambda m: f"{ordinal(int(m.group(1)))} {m.group(2)}", t)

    def expiry(m):                                                              # card expiry 09/28 da
        mo, yy = int(m.group(1)), m.group(2)
        if not 1 <= mo <= 12:
            return m.group(0)
        year = int(yy) + (2000 if len(yy) == 2 else 0)
        return f"{ordinal(year)} yil {attach(MONTHS[mo - 1], m.group('suf'))}"
    return re.sub(r"(?<![\d/])(0[1-9]|1[0-2])/(\d{2}|\d{4})(?![\d/])" + SUFFIX, expiry, t)


def _time_words(h: str, mi: str) -> str:
    words = cardinal(int(h))
    return words if int(mi) == 0 else f"{words} {cardinal(int(mi))}"


def _times(t: str) -> str:
    T = r"([01]?\d|2[0-3]):([0-5]\d)"

    def soat(prefix_text: str) -> str:
        return "" if re.search(r"\bsoat\s*$", prefix_text, re.I) else "soat "

    def rng(m):                                                                 # 9:00 dan 18:00 gacha / 9:00–18:00
        a = _time_words(m.group(1), m.group(2))
        b = _time_words(m.group(3), m.group(4))
        return f"{soat(t[:m.start()])}{attach(a, 'dan')} {attach(b, m.group('suf2') or 'gacha')}"
    t = re.sub(T + r"\s*(?:dan\s+|[–—-]\s*)" + T + r"(?:\s?(?P<suf2>gacha|ga)\b)?", rng, t)

    def single(m):
        return soat(t[:m.start()]) + attach(_time_words(m.group(1), m.group(2)), m.group("suf"))
    return re.sub(r"(?<![\d:])" + T + r"(?![\d:])" + SUFFIX, single, t)


def _money_units(t: str) -> str:
    # 79 000 so'm/oy -> oyiga yetmish to'qqiz ming so'm
    t = re.sub(rf"({NUM})\s*so'm\s*/\s*(oy|kun|yil|hafta|soat)\b",
               lambda m: f"{PER_UNITS[m.group(2)]} {cardinal(_num(m.group(1)))} so'm", t)
    # 1 250,50 so'm -> ... so'm ellik tiyin
    t = re.sub(rf"({NUM}),(\d{{2}})\s*so'm(\w*)",
               lambda m: f"{cardinal(_num(m.group(1)))} so'm{m.group(3)}"
                         + ("" if int(m.group(2)) == 0 else f" {cardinal(int(m.group(2)))} tiyin"), t)
    # $250 / 250$ / €100
    for sym, name in CURRENCY_SYMBOLS.items():
        s = re.escape(sym)
        t = re.sub(rf"{s}\s?({NUM})(?:,(\d+))?", lambda m: f"{_amount(m.group(1), m.group(2))} {name}", t)
        t = re.sub(rf"({NUM})(?:,(\d+))?\s?{s}", lambda m: f"{_amount(m.group(1), m.group(2))} {name}", t)
    # 100 Mbit/s gacha -> sekundiga yuz megabitgacha
    for unit, name in SPEED_UNITS.items():
        t = re.sub(rf"({NUM})(?:,(\d+))?\s?{re.escape(unit)}" + SUFFIX,
                   lambda m: f"sekundiga {attach(_amount(m.group(1), m.group(2)) + ' ' + name, m.group('suf'))}", t)
    # 2,5 GB / 750 MB / 100 mln
    for unit, name in UNITS.items():
        t = re.sub(rf"({NUM})(?:,(\d+))?\s?{re.escape(unit)}\b" + SUFFIX,
                   lambda m: attach(f"{_amount(m.group(1), m.group(2))} {name}", m.group("suf")), t)
    # 22,5% ni / 25% ini / 2%
    t = re.sub(rf"({NUM})(?:,(\d+))?\s?%" + SUFFIX,
               lambda m: attach(f"{_amount(m.group(1), m.group(2))} foiz", m.group("suf")), t)
    return t


def _amount(int_part: str, frac: str | None) -> str:
    return decimal(re.sub(r"\D", "", int_part), frac) if frac else cardinal(_num(int_part))


def _symbols(t: str) -> str:
    t = t.replace("№", "")
    # A = B so'm  ->  A B so'mga teng
    t = re.sub(r"\s*=\s*([^.,;!?=]+?)(?=\s*(?:[.,;!?]|$))", lambda m: " " + attach(m.group(1), "ga") + " teng", t)
    # ranges 1–3 -> birdan uch
    t = re.sub(r"(?<![\d,])(\d{1,3})\s?[–—]\s?(\d{1,3})(?![\d,])",
               lambda m: f"{attach(cardinal(int(m.group(1))), 'dan')} {cardinal(int(m.group(2)))}", t)
    t = re.sub(r"(?<!\d)(\d{1,3})-(\d{1,3})(?=\s+(?:ish\s+)?(?:kun|oy|yil|hafta|soat|daqiqa))",
               lambda m: f"{attach(cardinal(int(m.group(1))), 'dan')} {cardinal(int(m.group(2)))}", t)
    return t


def _numbers(t: str) -> str:
    # ordinal before a word: 3-qavat -> uchinchi qavat
    t = re.sub(r"\b(\d{1,4})-(?=[a-zA-Z])", lambda m: ordinal(int(m.group(1))) + " ", t)
    # negative amounts: -3 400 -> minus uch ming to'rt yuz
    t = re.sub(rf"(?:(?<=\s)|(?<=^)|(?<=:))-\s?({NUM})(?:,(\d+))?",
               lambda m: "minus " + _amount(m.group(1), m.group(2)), t)
    # decimals and plain/grouped numbers, with an optional detached suffix
    t = re.sub(rf"(?<![\w,])({NUM}),(\d+)(?![\w])" + SUFFIX,
               lambda m: attach(decimal(re.sub(r"\D", "", m.group(1)), m.group(2)), m.group("suf")), t)
    t = re.sub(rf"(?<![\w])({NUM})(?![\d])" + SUFFIX,
               lambda m: attach(cardinal(_num(m.group(1))), m.group("suf")), t)
    return t


def _short_numbers(t: str) -> str:
    # Service numbers with a leading zero are read digit by digit: 0611 -> nol, olti, bir, bir
    # ...but never a 3-digit group inside an amount ("25 000").
    return re.sub(r"(?<![\w,.:/])(?<!\d[ \u00a0\u202f])0\d{2,4}(?![\w,:/])", lambda m: digits(m.group(0)), t)


def normalize(text: str) -> str:
    t = cleanup(text)
    if looks_russian(t):
        return t
    if _CYRILLIC.search(t):
        t = transliterate(t)
    t = _APOS.sub("'", t)
    t = _caps(t)
    t = _email_url(t)
    t = _ussd(t)
    t = _lexicon(t)
    t = _cards_accounts(t)
    t = _dates(t)
    t = _times(t)
    t = _phones(t)
    t = _codes(t)
    t = _short_numbers(t)
    t = _money_units(t)
    t = _symbols(t)
    t = _numbers(t)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t[:1].upper() + t[1:] if t and t[0].islower() and text[:1].isupper() else t


# ---------------------------------------------------------------- language guard
_EN_WORDS = frozenset("""
    the an and is are was were be been to of in for with at by from this that these those you your we our
    i me my he she they them their what which who how why when where can could will would should do does
    did have has not please thank thanks hello hi yes sorry help there here about if but just okay ok hold
    call calling today one moment good morning evening afternoon wait sure let look
""".split())
_UZ_CUE = re.compile(r"[og]'[a-z]|q(?!u)|ngiz|\b(?:va|bu|siz|sizga|sizning|uchun|bilan|emas|bor|yo'q|ha|"
                     r"kerak|mumkin|iltimos|rahmat|qanday|nima|so'm|raqam|balans|albatta|tushunarli)\b")


def detect_language(text: str) -> str:
    """'ru' (Russian Cyrillic), 'en' (clearly English), otherwise 'uz'."""
    if looks_russian(text):
        return "ru"
    words = re.findall(r"[a-z']+", _APOS.sub("'", text.lower()))
    en = sum(w in _EN_WORDS for w in words)
    uz = sum(bool(_UZ_CUE.search(w)) for w in words)
    return "en" if en >= 2 and en > uz else "uz"
