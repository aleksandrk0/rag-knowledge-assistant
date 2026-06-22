"""Сигнатуры атак на LLM (RU + EN). Эвристики, а не панацея: детектор поднимает
recall на известных классах инъекций, но не заменяет защиту в глубину.
"""
from __future__ import annotations

import re

# Непрямая промпт-инъекция (OWASP LLM01). Каждый паттерн = высокая опасность.
INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "instruction_override",
        re.compile(
            r"(?i)\b(ignore|disregard|forget|override)\b[^.\n]{0,40}"
            r"\b(all|previous|above|prior|earlier|any)\b[^.\n]{0,40}"
            r"\b(instruction|prompt|rule|message|context|guideline)s?"
        ),
    ),
    (
        "instruction_override_ru",
        re.compile(
            r"(?i)(игнорир\w*|забуд\w*|отмен\w*|не\s+обращай\s+внимани\w*)"
            r"[^.\n]{0,40}(предыдущ\w*|вышеуказан\w*|прежн\w*|инструкц\w*|правил\w*)"
        ),
    ),
    (
        "role_override",
        re.compile(
            r"(?i)(you\s+are\s+now|act\s+as|pretend\s+to\s+be|new\s+(system\s+)?"
            r"instructions?|теперь\s+ты\b|представь,?\s+что\s+ты|веди\s+себя\s+как|"
            r"отвечай\s+как)"
        ),
    ),
    (
        "system_prompt_exfil",
        re.compile(
            r"(?i)(reveal|print|repeat|output|show|leak|выведи|покаж\w*|повтори|раскрой)"
            r"[^.\n]{0,40}(system\s+prompt|your\s+(initial\s+)?instructions?|"
            r"initial\s+prompt|систем\w*\s+промпт|свои\s+инструкц\w*|начальн\w*\s+инструкц\w*)"
        ),
    ),
    (
        "exfil_url",
        re.compile(r"!?\[[^\]]*\]\(\s*https?://[^)]+\)"),
    ),
]

# Любой URL в ответе модели (для выходного контроля эксфильтрации).
URL_PATTERN = re.compile(r"https?://[^\s)\]>]+")

# Невидимое смуглинг-кодирование: zero-width, bidi-override, word-joiner, BOM,
# Unicode-tag. Набор строится через chr() — в исходнике только ASCII, без
# невидимых символов и неоднозначных escape-последовательностей.
_SMUGGLING_RANGES = [
    (0x200B, 0x200F),    # zero-width space .. right-to-left mark
    (0x202A, 0x202E),    # bidi embeddings / overrides
    (0x2060, 0x2064),    # word joiner .. invisible plus
    (0xFEFF, 0xFEFF),    # BOM / zero-width no-break space
    (0xE0000, 0xE007F),  # Unicode tag block
]
_SMUGGLING_SET = "".join(
    chr(c) for lo, hi in _SMUGGLING_RANGES for c in range(lo, hi + 1)
)
SMUGGLING_CHARS = re.compile(f"[{re.escape(_SMUGGLING_SET)}]")
