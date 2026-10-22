from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger("Vieneu.PhonemeOverride")

_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_PHONE_TOKEN_RE = re.compile(r"\S+")
_PUNCT_TAIL_RE = re.compile(r"([.,!?;:]+)$")

# Onset produced by sea-g2p for both ch and tr.
_G2P_CH_TR_ONSET = "tʃ"

# Normalize non-standard tone placements / rare spellings that cause sea-g2p
# to split single Vietnamese syllables into multiple phoneme tokens.
_TEXT_SPELLING_FIXES = (
    (re.compile(r"\b([hH])ọe\b"), r"\g<1>oẹ"),
    (re.compile(r"\b([cC])hõa\b"), r"\g<1>hoã"),
    (re.compile(r"\b([nN])ghoẹo\b"), r"\g<1>goẹo"),
)

# Normalize Bible references: e.g. "Ma-thi-ơ 2:5-10" -> "Ma thi ơ chương 2 từ câu 5 đến câu 10"
# "Ma-thi-ơ 2:5" -> "Ma thi ơ chương 2 câu 5"
_BIBLE_BOOKS = (
    # Cựu Ước
    r"Sáng[- ]+Thế(?:[- ]+Ký)?|Xuất[- ]+Ê[- ]+díp[- ]+tô(?:[- ]+Ký)?|Xuất[- ]+Hành|Lê[- ]+vi(?:[- ]+Ký)?|"
    r"Dân[- ]+Số(?:[- ]+Ký)?|Phục[- ]+Truyền(?:[- ]+Luật[- ]+Lệ)?(?:[- ]+Ký)?|Giô[- ]+suê|Các[- ]+Quan[- ]+Xét|Ru[- ]+tơ|"
    r"(?:[12]|I{1,2})[- ]+Sa[- ]+mu[- ]+ên|(?:[12]|I{1,2})[- ]+Các[- ]+Vua|(?:[12]|I{1,2})[- ]+Sử[- ]+Ký|"
    r"E[- ]+xơ[- ]+ra|Nê[- ]+hê[- ]+mi|Ê[- ]+xơ[- ]+tê|Gióp|Thi[- ]+Thiên|Thánh[- ]+Vịnh|Châm[- ]+Ngôn|"
    r"Truyền[- ]+Đạo|Nhã[- ]+Ca|Ê[- ]+sai|Giê[- ]+rê[- ]+mi|Ca[- ]+Thương|Ê[- ]xê[- ]+chi[- ]+ên|Đa[- ]+ni[- ]+ên|"
    r"Ô[- ]+sê|Giô[- ]+ên|A[- ]+mốt|Áp[- ]+đia|Giô[- ]+na|Mi[- ]+chê|Na[- ]+hum|Ha[- ]+ba[- ]+cúc|"
    r"Xô[- ]+phô[- ]+ni|A[- ]+ghê|Xa[- ]+cha[- ]+ri|Ma[- ]+la[- ]+chi|"
    # Tân Ước
    r"Ma[- ]+thi[- ]+ơ|Mát[- ]+thêu|Mác[- ]+cô|Mác|Lu[- ]+ca|Giăng|Gio[- ]+an|"
    r"Công[- ]+Vụ(?:[- ]+Các[- ]+Sứ[- ]+Đồ)?|Rô[- ]+ma|"
    r"(?:[1-3]|I{1,3})[- ]+Cô[- ]+rinh[- ]+tô|Ga[- ]+la[- ]+ti|Ê[- ]+phê[- ]+sô|Phi[- ]+líp|Cô[- ]+lô[- ]+se|"
    r"(?:[12]|I{1,2})[- ]+Tê[- ]+sa[- ]+lô[- ]+ni[- ]+ca|(?:[12]|I{1,2})[- ]+Ti[- ]+mô[- ]+thê|"
    r"Tít|Phi[- ]+lê[- ]+môn|Hê[- ]bơ[- ]+rơ|Gia[- ]+cơ|(?:[12]|I{1,2})[- ]+Phi[- ]+e[- ]+rơ|"
    r"(?:[1-3]|I{1,3})[- ]+Giăng|(?:[1-3]|I{1,3})[- ]+Gio[- ]+an|Giu[- ]+đe|Khải[- ]+Huyền|"
    # Viết tắt thông dụng
    r"Mt|Mc|Lc|Ga|Stk|Tv|Rm|Cv|Kh"
)

# Quy ước câu Kinh Thánh:
# - Viết sát dấu phẩy (không có dấu cách, vd 3:2,4 hoặc 1:1-3,5) -> tính là câu 2 và câu 4
# - Có khoảng trắng sau dấu phẩy (vd 3:2, 4...) -> dấu phẩy là ngắt câu, số 4 thuộc mệnh đề sau
_BIBLE_REF_RE = re.compile(
    rf"(?<!\w)({_BIBLE_BOOKS})[,.]?\s+(\d+):(\d+(?:\s*[-–—]\s*\d+)?(?:,\d+(?:\s*[-–—]\s*\d+)?)*)(?!\w)",
    re.IGNORECASE,
)


_NUMBERED_BIBLE_BOOKS = (
    r"Cô[- ]+rinh[- ]+tô|Sa[- ]+mu[- ]+ên|Các[- ]+Vua|Sử[- ]+Ký|"
    r"Tê[- ]+sa[- ]+lô[- ]+ni[- ]+ca|Ti[- ]+mô[- ]+thê|Phi[- ]+e[- ]+rơ|Giăng|Gio[- ]+an"
)

_STANDALONE_NUMBERED_BOOK_RE = re.compile(
    rf"(?<!\w)(?:(1|I{1})|(2|I{2})|(3|I{3}))\s+({_NUMBERED_BIBLE_BOOKS})(?!\w)",
    re.IGNORECASE,
)


def _clean_bible_book(raw_book: str) -> str:
    b = raw_book.replace("-", " ")
    b = re.sub(r"\s+", " ", b).strip()
    m = re.match(r"^(?:(1|I{1})\s+|(2|I{2})\s+|(3|I{3})\s+)(.+)$", b, flags=re.IGNORECASE)
    if m:
        suffix = "nhất" if m.group(1) else ("nhì" if m.group(2) else "ba")
        main_name = m.group(4).strip()
        return f"{main_name} {suffix}"
    return b


def _format_single_verse_or_range(v: str) -> str:
    v = v.strip()
    parts = re.split(r"\s*[-–—]\s*", v)
    if len(parts) == 2 and parts[0] and parts[1]:
        return f"từ câu {parts[0]} đến câu {parts[1]}"
    return f"câu {v}"


def _format_verses(verses_raw: str) -> str:
    if "," in verses_raw or ";" in verses_raw:
        chunks = [c.strip() for c in re.split(r"[,;]\s*", verses_raw) if c.strip()]
        formatted = [_format_single_verse_or_range(c) for c in chunks]
        if len(formatted) == 1:
            return formatted[0]
        elif len(formatted) == 2:
            return f"{formatted[0]} và {formatted[1]}"
        else:
            return ", ".join(formatted[:-1]) + f" và {formatted[-1]}"
    return _format_single_verse_or_range(verses_raw)


def _normalize_bible_refs(text: str) -> str:
    def _repl(m: re.Match) -> str:
        book = _clean_bible_book(m.group(1))
        chap = m.group(2)
        verses_raw = m.group(3).strip()
        verse_str = _format_verses(verses_raw)
        return f"{book.rstrip(',')}, chương {chap} {verse_str}"

    text = _BIBLE_REF_RE.sub(_repl, text)

    # Đổi các sách có số 1/2/3 đứng riêng lẻ không kèm số chương:câu
    # ví dụ: "1 Cô-rinh-tô", "2 cô-rinh-tô" -> "Cô rinh tô nhất", "cô rinh tô nhì"
    def _repl_standalone(m: re.Match) -> str:
        suffix = "nhất" if m.group(1) else ("nhì" if m.group(2) else "ba")
        book_name = m.group(4).replace("-", " ")
        book_name = re.sub(r"\s+", " ", book_name).strip()
        return f"{book_name} {suffix}"

    text = _STANDALONE_NUMBERED_BOOK_RE.sub(_repl_standalone, text)

    # Thêm dấu phẩy ngắt nghỉ nếu sau "nhất/nhì/ba" đi liền với "chương" / "đoạn"
    # Ví dụ: "Cô rinh tô nhất chương 13" -> "Cô rinh tô nhất, chương 13"
    text = re.sub(
        r"\b(nhất|nhì|ba)\s+(chương|đoạn)\b",
        r"\1, \2",
        text,
        flags=re.IGNORECASE,
    )
    return text


def _normalize_problematic_words(text: str) -> str:
    text = _normalize_bible_refs(text)
    for pat, repl in _TEXT_SPELLING_FIXES:
        text = pat.sub(repl, text)
    return text


# Applied for every voice, northern and southern.
_UNIVERSAL_WORD_PHONEME_OVERRIDES: dict[tuple[str, ...], str] = {
    ("christ",): "kɹˈɪst",
}
# Southern-only. Merged with the universal table for southern voices.
_SOUTHERN_WORD_PHONEME_OVERRIDES: dict[tuple[str, ...], str] = {
    ("nghìn",): "ŋˈaː2n",
}
_WORD_PHONEME_OVERRIDES: dict[tuple[str, ...], str] = {
    **_UNIVERSAL_WORD_PHONEME_OVERRIDES,
    **_SOUTHERN_WORD_PHONEME_OVERRIDES,
}


def southern_phoneme_enabled() -> bool:
    """True unless VIENEU_SOUTHERN_PHONEME is explicitly disabled."""
    v = os.getenv("VIENEU_SOUTHERN_PHONEME", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def apply_southern_onset_to_word(word: str, phones: str) -> str:
    """Map one syllable's phone string using its grapheme onset:
    - ch vs tr: ch -> c, tr -> ʈ
    - s vs x:   s -> ʂ (uốn lưỡi / retroflex), x -> s (thẳng lưỡi / alveolar)
    """
    w = word.lower().strip("'_")
    m = _PUNCT_TAIL_RE.search(phones)
    tail = m.group(1) if m else ""
    core = phones[: -len(tail)] if tail else phones

    if core.startswith(_G2P_CH_TR_ONSET):
        if w.startswith("tr"):
            core = "ʈ" + core[len(_G2P_CH_TR_ONSET) :]
        elif w.startswith("ch"):
            core = "c" + core[len(_G2P_CH_TR_ONSET) :]
    elif core.startswith("s"):
        if w.startswith("s"):
            core = "ʂ" + core[1:]
        elif w.startswith("x"):
            core = "s" + core[1:]

    return core + tail


def _phonemize_fragment_base(text: str, *, punc_norm_end: bool) -> str:
    from vieneu_utils.phonemize_text import _phonemize_cached, phonemize_text_with_emotions

    fragment = text.strip()
    if not fragment:
        return ""
    if punc_norm_end:
        return phonemize_text_with_emotions(fragment)
    return _phonemize_cached(fragment, punc_norm=False)


def _align_words(text: str) -> list[str]:
    from vieneu_utils.phonemize_text import _get_normalizer

    norm = _get_normalizer().normalize(text, punc_norm=False)
    return _WORD_RE.findall(norm.lower())


_word_token_count_cache: dict[str, int] = {}


def _token_counts_for_words(words: list[str]) -> list[int]:
    """How many phoneme tokens G2P emits per word.

    Batches all cache misses into a single ``phonemize_batch`` call (instead of
    one Rust/FFI call per word) and memoizes results across calls, since common
    function words (là, và, của, không, ...) recur constantly across chunks.
    """
    missing = [w for w in words if w not in _word_token_count_cache]
    if missing:
        from vieneu_utils.phonemize_text import _get_g2p

        unique_missing = list(dict.fromkeys(missing))
        phones_list = _get_g2p().phonemize_batch(unique_missing, punc_norm=False)
        for w, phones in zip(unique_missing, phones_list):
            _word_token_count_cache[w] = len(_PHONE_TOKEN_RE.findall(phones))

    return [_word_token_count_cache[w] for w in words]


def _find_override(
    words: list[str],
    i: int,
    overrides: dict[tuple[str, ...], str],
) -> tuple[str | None, int]:
    """Longest match in ``overrides`` starting at ``words[i]``.

    Returns ``(override_phones, span)``, or ``(None, 1)`` if no override
    applies to ``words[i]``.
    """
    if overrides:
        max_span = min(max(len(k) for k in overrides), len(words) - i)
        for span in range(max_span, 0, -1):
            override = overrides.get(tuple(words[i : i + span]))
            if override is not None:
                return override, span
    return None, 1


def _apply_word_transforms(
    text: str,
    phones: str,
    *,
    apply_onset: bool,
    overrides: dict[tuple[str, ...], str] | None = None,
) -> str:
    """Apply word-phoneme overrides and, if ``apply_onset``, southern ch/tr
    and s/x onset patches to a full-fragment phone string, using G2P word/token
    alignment.
    """
    if overrides is None:
        overrides = _WORD_PHONEME_OVERRIDES
    words = _align_words(text)
    tokens = _PHONE_TOKEN_RE.findall(phones)
    if not words or not tokens:
        return phones

    token_counts = _token_counts_for_words(words)

    patched: list[str] = []
    ti = 0
    i = 0
    while i < len(words):
        override, span = _find_override(words, i, overrides)
        n = sum(token_counts[i : i + span])
        if ti + n > len(tokens):
            logger.warning(
                "Southern phoneme align mismatch (%d words, %d tokens consumed, "
                "%d tokens total); skipping onset/override for this fragment.",
                len(words),
                ti,
                len(tokens),
            )
            return phones

        if override is not None:
            # Preserve any trailing punctuation attached to the last token in
            # the matched span (e.g. "christ." at the end of a sentence) since
            # the override phones don't carry it.
            last_token = tokens[ti + n - 1]
            m = _PUNCT_TAIL_RE.search(last_token)
            override_tokens = override.split()
            if m:
                override_tokens[-1] += m.group(1)
            patched.extend(override_tokens)
        else:
            chunk = tokens[ti : ti + n]
            if apply_onset:
                chunk = [apply_southern_onset_to_word(words[i], tok) for tok in chunk]
            patched.extend(chunk)

        ti += n
        i += span

    if ti != len(tokens):
        logger.warning(
            "Southern phoneme align mismatch (%d words vs %d tokens); "
            "skipping onset/override for this fragment.",
            len(words),
            len(tokens),
        )
        return phones

    return " ".join(patched)


def _apply_southern_to_phone_stream(text: str, phones: str) -> str:
    """Patch ch/tr and s/x onsets on a full-sentence phone string."""
    return _apply_word_transforms(text, phones, apply_onset=True)


def phonemize_words_southern(text: str, *, punc_norm_end: bool = True) -> str:
    text = _normalize_problematic_words(text)
    phones = _phonemize_fragment_base(text, punc_norm_end=punc_norm_end)
    if not phones:
        return ""
    return _apply_word_transforms(text, phones, apply_onset=southern_phoneme_enabled())


def phonemize_text_with_emotions_southern(
    text: str,
    *,
    apply_onset: bool = True,
    overrides: dict[tuple[str, ...], str] | None = None,
) -> str:
    """Like phonemize_text_with_emotions but applies word-phoneme overrides
    and, if ``apply_onset``, ch/tr and s/x onset patches for the southern accent.

    ``overrides`` defaults to the full southern table (universal + southern-only).
    Northern voices pass ``_UNIVERSAL_WORD_PHONEME_OVERRIDES`` with
    ``apply_onset=False``.

    Preserves inline emotion tokens (<|emotion_k|>, [cười], ...).
    """
    from vieneu_utils.phonemize_text import phonemize_text_with_emotions

    if overrides is None:
        overrides = _WORD_PHONEME_OVERRIDES

    text = _normalize_problematic_words(text)
    if not overrides and not apply_onset:
        return phonemize_text_with_emotions(text)

    from sea_g2p import punc_norm as _punc_norm

    from vieneu_utils.phonemize_text import (
        _ATTACHING_PUNCT,
        _EMOTION_SPLIT_RE,
        _emotion_tag_token,
    )

    if "[" not in text and "<|emotion_" not in text:
        norm_text = _normalize_problematic_words(text)
        phones = _phonemize_fragment_base(norm_text, punc_norm_end=True)
        if not phones:
            return ""
        return _apply_word_transforms(
            norm_text, phones, apply_onset=apply_onset, overrides=overrides
        )

    out = ""
    for i, part in enumerate(_EMOTION_SPLIT_RE.split(text)):
        token = _emotion_tag_token(part) if i % 2 == 1 else None
        if token is not None:
            out = (out + " " + token) if out else token
            continue
        if not part or not part.strip():
            continue
        norm_part = _normalize_problematic_words(part)
        base = _phonemize_fragment_base(norm_part, punc_norm_end=False)
        ph = (
            _apply_word_transforms(
                norm_part, base, apply_onset=apply_onset, overrides=overrides
            )
            if base
            else ""
        )
        if not ph:
            continue
        if not out:
            out = ph
        elif ph[0] in _ATTACHING_PUNCT:
            out += ph
        else:
            out += " " + ph

    return _punc_norm(out)


def phonemize_for_southern_voice(text: str) -> str:
    """Entry point for finetune train/infer for southern voices.

    Word-phoneme overrides (``_WORD_PHONEME_OVERRIDES``) always apply; ch/tr
    and s/x onset patching for the southern accent additionally applies unless
    VIENEU_SOUTHERN_PHONEME is explicitly disabled.
    """
    return phonemize_text_with_emotions_southern(
        text, apply_onset=southern_phoneme_enabled()
    )


def phonemize_for_northern_voice(text: str) -> str:
    """Entry point for northern voices (giữ nguyên phoneme gốc từ sea-g2p).

    Preserves inline emotion tokens without modifying onsets or applying
    southern-specific word overrides. Accent-independent overrides such as
    "christ" still apply.
    """
    return phonemize_text_with_emotions_southern(
        text,
        apply_onset=False,
        overrides=_UNIVERSAL_WORD_PHONEME_OVERRIDES,
    )


def is_southern_accent(
    accent: str | None = None,
    description: str | None = None,
    voice_name: str | None = None,
    model_name: str | None = None,
) -> bool:
    """Determine if a voice or model is Southern (miền Nam) or Northern (miền Bắc).

    Returns True for Southern accent, False for Northern accent (default/gốc).
    """
    if accent is not None and str(accent).strip():
        acc = str(accent).strip().lower()
        if any(s in acc for s in ("south", "nam", "mien_nam", "miền nam", "giong_nam", "giọng nam")):
            return True
        if any(s in acc for s in ("north", "bac", "mien_bac", "miền bắc", "giong_bac", "giọng bắc")):
            return False

    if description is not None and str(description).strip():
        desc = str(description).strip().lower()
        if any(s in desc for s in ("miền nam", "mien nam", "nam bộ", "nam bo", "southern")):
            return True
        if any(s in desc for s in ("miền bắc", "mien bac", "bắc bộ", "bac bo", "hà nội", "ha noi", "northern")):
            return False

    for s in (voice_name, model_name):
        if s is not None and str(s).strip():
            name = str(s).strip().lower()
            if any(k in name for k in ("mydung", "my_dung", "han_voice", "vgm-han", "southern", "mien_nam", "giong_nam")):
                return True
            if any(k in name for k in ("hoan_voice", "vgm-hoan", "northern", "mien_bac", "giong_bac")):
                return False

    return False


def phonemize_for_voice(text: str, is_southern: bool = False) -> str:
    """Phonemize text based on voice accent.

    Southern voices get ch/tr and s/x onset patches plus southern word
    overrides. Every voice, including northern, still gets accent-independent
    overrides such as "christ".
    """
    if is_southern:
        return phonemize_for_southern_voice(text)
    return phonemize_for_northern_voice(text)
