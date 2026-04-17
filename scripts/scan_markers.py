#!/usr/bin/env python3
"""
scan_markers.py - pre-сканер русского текста на маркеры AI-слопа.

Проходит regex'ами по словарю маркеров из всех 4 групп (A, B, C, D),
находит позиции и считает AI-Slop Score по той же формуле, что и SKILL.md.

Назначение: запустить ПЕРЕД LLM-проходом, чтобы:
  1) понять, нужна ли вообще редактура (если score < 20 - пропустить)
  2) подсветить LLM конкретные позиции для исправления
  3) выдать numeric baseline для сравнения "до/после"

Зависимостей нет, Python 3.8+.

Использование:
  python scan_markers.py path/to/text.txt
  python scan_markers.py - < text.txt           # из stdin
  python scan_markers.py text.txt --json        # вывод в JSON
  python scan_markers.py text.txt --quiet       # только score

Выходы:
  exit 0 - score < 20 (текст чистый, редактура не нужна)
  exit 1 - score 20-60 (нужна редактура)
  exit 2 - score > 60 (сильный AI-слоп)
  exit 3 - ошибка ввода
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------------
# Словари маркеров по группам.
# Источник: references/word-scanner.md
# Регулярки чувствительны к регистру только там, где это важно (начало предложения).
# ---------------------------------------------------------------------------

# Группа A - критические (вес 4)
GROUP_A_PATTERNS = [
    # Copula avoidance
    r"\bявляется\b",
    r"\bпредставляет\s+собой\b",
    r"\bвыступает\s+(?:в\s+роли|катализатором|партн[её]ром)\b",
    r"\bслужит\s+(?:основой|фундаментом|партн[её]ром)\b",
    r"\bносит\s+характер\b",
    r"\bзнаменует\s+собой\b",
    # Отглагольные конструкции
    r"\bосуществля(?:ть|ет|ем|ют)\b",
    r"\bпровод(?:ить|ит|им|ят)\s+работ\w*",
    r"\bоказыва(?:ть|ет|ем|ют)\s+помощь\b",
    r"\bпринима(?:ть|ет|ем|ют)\s+участие\b",
    r"\bвест[иь]\s+борьбу\b",
    r"\bпроизвод(?:ить|ит|им|ят)\s+расч[её]т\w*",
    r"\bдава(?:ть|[её]т)\s+оценк\w+",
    # Негативные параллелизмы
    r"\bне\s+просто\s+\w+,?\s*а\b",
    r"\bне\s+только\s+[\w\s,]+но\s+и\b",
    r"^\s*Это\s+не\s+просто\b",
    r"\bречь\s+ид[её]т\s+не\s+только\b",
    # Артефакты чатбота
    r"\bнадеюсь,?\s+(?:это|данная)\s+(?:помож|информаци)\w*",
    r"\bбуду\s+рад[аы]?\s+помочь\b",
    r"\bесли\s+у\s+вас\s+есть\s+вопрос\w*",
    r"\bдайте\s+знать\b",
    r"^\s*Вот\s+(?:краткий\s+)?обзор\b",
    # Подобострастие
    r"\bотличный\s+вопрос[!\.]",
    r"\bвы\s+совершенно\s+прав\w*",
    r"\bзамечательное\s+наблюдение\b",
    r"\bпрекрасная\s+мысль\b",
]

# Группа B - высокие (вес 3)
GROUP_B_PATTERNS = [
    # AI-словарь
    r"\bключев(?:ой|ая|ое|ые|ым|ого|ому)\b",
    r"\bважнейш(?:ий|ая|ее|ие|им)\b",
    r"\bреша(?:ющ|вш)\w+",
    r"\bповоротн\w+",
    r"\bдемонстриру(?:ет|ют|я)\b",
    r"\bотража(?:ет|ют|я)\b",
    r"\bспособству(?:ет|ют|я)\b",
    r"\bобеспечива(?:ет|ют|я)\b",
    r"\bсодейству(?:ет|ют|я)\b",
    r"\bформиру(?:ет|ют|я)\b",
    r"\bсвидетельству(?:ет|ют|я)\b",
    r"\bподч[её]ркива(?:ет|ют|я)\b",
    # Шаблонные переходы
    r"\bважно\s+отметить,?\s+что\b",
    r"\bследует\s+подчеркнуть,?\s+что\b",
    r"\bнеобходимо\s+учитывать,?\s+что\b",
    r"\bстоит\s+обратить\s+внимание\b",
    r"\bнельзя\s+не\s+упомянуть\b",
    r"\bособо\s+следует\s+выделить\b",
    # Раздувание значимости
    r"\bвносит\s+вклад\s+в\b",
    r"\bзакладывает\s+основу\s+для\b",
    r"\bзнаменует\s+собой\s+ключев\w+\s+этап\b",
    r"\bсимволизирует\s+приверженность\b",
    # Деепричастные хвосты
    r",\s+подч[её]ркива(?:я|ющ\w+)\b",
    r",\s+обеспечива(?:я|ющ\w+)\b",
    r",\s+символизиру(?:я|ющ\w+)\b",
    r",\s+способству(?:я|ющ\w+)\b",
    r",\s+отража(?:я|ющ\w+)\b",
    r",\s+демонстриру(?:я|ющ\w+)\b",
    # Безличные конструкции (B7)
    r"\bсохраня(?:е|ю)тся\s+автоматически\b",
    r"\bне\s+требу(?:е|ю)тся\b",
    r"\bпредоставля(?:е|ю)тся\b(?!\s+\w+ом\b)",  # без явного субъекта
    r"\bобеспечива(?:е|ю)тся\b(?!\s+\w+ом\b)",
    r"\bдопуска(?:е|ю)тся\b(?!\s+\w+ом\b)",
    # Пассив без субъекта
    r"\bбыл[аиоы]?\s+принят\w*",
    r"\bбыл[аиоы]?\s+выполнен\w*",
    r"\bбыл[аиоы]?\s+достигнут\w*",
]

# Группа C - средние (вес 2)
GROUP_C_PATTERNS = [
    # Рекламный язык
    r"\bможет\s+похвастаться\b",
    r"\b(?:яркий|яркая|яркое|яркие)\s+(?:центр|представитель|пример)\w*",
    r"\bбогат(?:ый|ая|ое|ые)\s+(?:наследи|культур|истори)\w*",
    r"\bзахватывающ\w+\s+дух\b",
    r"\bпотрясающ\w+",
    r"\bноваторск\w+",
    r"\bраспол[ао]женн\w+\s+в\s+(?:сердце|самом)\b",
    # Размытые атрибуции
    r"\bпо\s+мнению\s+эксперт\w+",
    r"\bаналитики\s+отмеча\w+",
    r"\bисследователи\s+утвержд\w+",
    r"\bряд\s+источник\w+\s+указыва\w+",
    r"\bнекоторые\s+критики\s+счита\w+",
    # Ложные диапазоны (грубая эвристика)
    r"\bот\s+\w+\s+до\s+\w+,\s+от\s+\w+\s+до\s+\w+",
    # Хеджирование
    r"\bвозможно,?\s+можно\s+предположить\b",
    r"\bв\s+определ[её]нной\s+степени\b",
    r"\bможет\s+оказать\s+некоторое\s+влияние\b",
    # Авторитетные тропы (C7)
    r"\bпо\s+сути\b",
    r"\bна\s+самом\s+деле\b",
    r"\bв\s+действительности\b",
    r"\bглавный\s+вопрос\s+в\s+том\b",
    r"\bпо\s+большому\s+сч[её]ту\b",
    r"\bсуть\s+в\s+том\b",
    r"\bфундаментально,?\b",
    r"\bв\s+своей\s+основе\b",
    # Иностранщина
    r"\bимплементир\w+",
    r"\bдедлайн\w*",
    r"\bконсенсус\w*",
    r"\bфидбек\w*",
    r"\bколлаборац\w+",
]

# Группа D - стилистические (вес 1)
GROUP_D_PATTERNS = [
    # Формульные выводы
    r"\bнесмотря\s+на\s+(?:эти\s+)?вызов\w+",
    r"\bпродолжа(?:ет|ют)\s+процветать\b",
    # Общие позитивные заключения
    r"\bбудущее\s+выглядит\s+ярк\w+",
    r"\bвпереди\s+захватывающие\s+времена\b",
    r"\bпуть\s+к\s+совершенств\w+",
    r"\bважный\s+шаг\s+в\s+правильн\w+\s+направлен\w+",
    # Излишняя академичность
    r"\bв\s+контексте\s+вышеизложенн\w+",
    r"\bпредставляется\s+целесообразн\w+",
    # Сигнальные анонсы (D6)
    r"\bдавайте\s+(?:разбер[её]м|рассмотрим|погруз\w+)\b",
    r"\bвот\s+что\s+(?:нужно|важно|стоит)\s+знать\b",
    r"\bперейд[её]м\s+к\s+сути\b",
    r"\bтеперь\s+(?:посмотрим|рассмотрим)\s+на\b",
    r"\bбез\s+лишних\s+предислов\w+",
    r"\bсейчас\s+расскажу\b",
    # Эмодзи в заголовках/списках
    r"^[\s>\-\*\d\.]*[\U0001F300-\U0001FAFF\U00002600-\U000027BF]\s+\*\*",
]


@dataclass
class Finding:
    group: str          # "A" | "B" | "C" | "D"
    pattern: str        # regex, который сработал
    match: str          # реальный текст
    position: int       # символ-индекс в оригинале

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScanResult:
    text_length_words: int
    text_length_chars: int
    findings: list[Finding] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=lambda: {"A": 0, "B": 0, "C": 0, "D": 0})
    score: int = 0
    severity: str = "clean"  # clean | light | moderate | heavy | critical

    def to_dict(self) -> dict:
        return {
            "text_length_words": self.text_length_words,
            "text_length_chars": self.text_length_chars,
            "ai_slop_score": self.score,
            "severity": self.severity,
            "counts_by_group": self.counts,
            "total_findings": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
        }


# Веса групп (как в SKILL.md)
WEIGHTS = {"A": 4, "B": 3, "C": 2, "D": 1}


def _scan_group(text: str, patterns: Iterable[str], group: str) -> list[Finding]:
    out: list[Finding] = []
    for pat in patterns:
        try:
            rx = re.compile(pat, flags=re.IGNORECASE | re.UNICODE | re.MULTILINE)
        except re.error as exc:
            print(f"[scan_markers] WARN: bad regex {pat!r}: {exc}", file=sys.stderr)
            continue
        for m in rx.finditer(text):
            out.append(Finding(group=group, pattern=pat, match=m.group(0), position=m.start()))
    return out


def _severity(score: int) -> str:
    if score < 20:
        return "clean"
    if score < 41:
        return "light"
    if score < 61:
        return "moderate"
    if score < 81:
        return "heavy"
    return "critical"


def scan(text: str) -> ScanResult:
    """Сканирует текст по всем 4 группам и возвращает ScanResult со score."""
    if not text or not text.strip():
        return ScanResult(text_length_words=0, text_length_chars=0)

    findings: list[Finding] = []
    findings += _scan_group(text, GROUP_A_PATTERNS, "A")
    findings += _scan_group(text, GROUP_B_PATTERNS, "B")
    findings += _scan_group(text, GROUP_C_PATTERNS, "C")
    findings += _scan_group(text, GROUP_D_PATTERNS, "D")

    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for f in findings:
        counts[f.group] += 1

    words = len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))
    weighted_sum = sum(WEIGHTS[g] * counts[g] for g in counts)
    # Делитель выбран эмпирически: words/3 даёт хорошее распределение
    # на текстах от 30 до 5000 слов. Для очень коротких добавляем нижнюю границу.
    denom = max(words / 3, 5)
    score = min(100, round(100 * weighted_sum / denom))

    return ScanResult(
        text_length_words=words,
        text_length_chars=len(text),
        findings=findings,
        counts=counts,
        score=score,
        severity=_severity(score),
    )


def _format_human(result: ScanResult) -> str:
    lines = [
        f"AI-Slop Score: {result.score}/100  [{result.severity}]",
        f"Длина: {result.text_length_words} слов, {result.text_length_chars} символов",
        f"Находок: {len(result.findings)} всего "
        f"(A:{result.counts['A']} B:{result.counts['B']} "
        f"C:{result.counts['C']} D:{result.counts['D']})",
    ]
    if result.findings:
        lines.append("")
        lines.append("Топ-10 находок:")
        for f in result.findings[:10]:
            preview = f.match.replace("\n", " ")[:60]
            lines.append(f"  [{f.group}] поз.{f.position:>5}: {preview!r}")
        if len(result.findings) > 10:
            lines.append(f"  ... и ещё {len(result.findings) - 10}")
    return "\n".join(lines)


def _read_input(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {path}")
    if not p.is_file():
        raise ValueError(f"not a file: {path}")
    return p.read_text(encoding="utf-8")


def _exit_code(score: int) -> int:
    if score < 20:
        return 0
    if score <= 60:
        return 1
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="scan_markers",
        description="Pre-сканер русского текста на AI-слоп. См. SKILL.md.",
    )
    parser.add_argument("path", help="путь к файлу или '-' для stdin")
    parser.add_argument("--json", action="store_true", help="вывод в JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="только score числом")
    args = parser.parse_args(argv)

    try:
        text = _read_input(args.path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    result = scan(text)

    if args.quiet:
        print(result.score)
    elif args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(_format_human(result))

    return _exit_code(result.score)


if __name__ == "__main__":
    sys.exit(main())
