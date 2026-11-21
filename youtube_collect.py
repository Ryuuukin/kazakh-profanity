#!/usr/bin/env python3
"""
Сбор казахских комментариев с YouTube (YouTube Data API v3) для датасета мата в отзывах.

Режимы:
  collect   — ищет видео-обзоры товаров и выкачивает комментарии
  rehydrate — заново скачивает тексты по списку comment_id (для публикации только ID)

Примеры:
  export YT_API_KEY=ваш_ключ
  python youtube_collect.py collect --out raw/comments.jsonl
  python youtube_collect.py collect --video-ids my_videos.txt --out raw/comments.jsonl
  python youtube_collect.py rehydrate --ids data/comment_ids.txt --out raw/rehydrated.jsonl

Что НЕ сохраняется: имя автора, ID канала автора, аватар, ссылка на профиль.
В тексте маскируются e-mail, телефоны, ссылки и @упоминания.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

API = "https://www.googleapis.com/youtube/v3"

DEFAULT_QUERIES = [
    "тауар шолуы", "шолу қазақша", "телефон шолу", "kaspi шолу",
    "обзор қазақша", "распаковка қазақша", "сатып алдым шолу",
    "ноутбук шолу", "наушник шолу", "косметика шолу",
]

# --- Определение казахского текста -------------------------------------------
KK_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
KK_WORDS = {
    "мен", "және", "жоқ", "керек", "өте", "жақсы", "нашар", "емес", "бар",
    "ғой", "екен", "бұл", "осы", "сапасы", "рахмет", "ақша", "тауар", "алдым",
    "жаман", "неге", "кім", "қалай", "ма", "ме", "па", "пе", "ба", "бе",
    # латиница / транслит
    "jaqsy", "zhaksy", "jaman", "zhaman", "emes", "rahmet", "kerek", "ote",
}
WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁәғқңөұүһіӘҒҚҢӨҰҮҺІ]+")


def kk_score(text: str) -> float:
    """Доля сигналов казахского: спецбуквы + частотные казахские слова."""
    words = WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in KK_WORDS or any(c in KK_LETTERS for c in w))
    return hits / len(words)


# --- Удаление персональных данных --------------------------------------------
PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<EMAIL>"),
    (re.compile(r"https?://\S+|www\.\S+"), "<URL>"),
    (re.compile(r"(?:\+?7|8)[\s\-()]*\d{3}[\s\-()]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}"), "<PHONE>"),
    (re.compile(r"\b\d{12}\b"), "<ID>"),          # ИИН
    (re.compile(r"\b(?:\d[ -]?){16}\b"), "<CARD>"),  # номер карты
    (re.compile(r"@[\w.\-]+"), "<USER>"),
]


def scrub(text: str) -> str:
    for rx, token in PII_PATTERNS:
        text = rx.sub(token, text)
    return re.sub(r"\s+", " ", text).strip()


# --- API ---------------------------------------------------------------------
class YT:
    def __init__(self, key: str):
        self.key = key
        self.units = 0

    def get(self, endpoint: str, cost: int, **params):
        params["key"] = self.key
        for attempt in range(3):
            r = requests.get(f"{API}/{endpoint}", params=params, timeout=30)
            self.units += cost
            if r.status_code == 200:
                return r.json()
            reason = ""
            try:
                reason = r.json()["error"]["errors"][0]["reason"]
            except Exception:
                pass
            if reason in ("commentsDisabled", "videoNotFound", "forbidden"):
                return None
            if reason in ("quotaExceeded", "dailyLimitExceeded"):
                sys.exit(f"Квота исчерпана (потрачено ~{self.units} units). Продолжите завтра.")
            time.sleep(2 ** attempt)
        print(f"  ! {endpoint} {r.status_code}: {r.text[:200]}", file=sys.stderr)
        return None


def search_videos(yt: YT, queries, per_query: int):
    ids = []
    for q in queries:
        data = yt.get("search", 100, part="id", q=q, type="video",
                      regionCode="KZ", relevanceLanguage="kk",
                      maxResults=min(per_query, 50))
        if data:
            for it in data.get("items", []):
                ids.append(it["id"]["videoId"])
        print(f"  поиск «{q}»: всего видео {len(set(ids))}")
    return list(dict.fromkeys(ids))


def comment_record(c: dict, video_id: str, parent_id=None):
    s = c["snippet"]
    return {
        "comment_id": c["id"],
        "parent_id": parent_id,
        "video_id": video_id,
        "published": s.get("publishedAt", "")[:10],
        "text": scrub(s.get("textOriginal") or s.get("textDisplay", "")),
    }


def fetch_comments(yt: YT, video_id: str, max_pages: int):
    token, pages = None, 0
    while pages < max_pages:
        params = dict(part="snippet,replies", videoId=video_id,
                      maxResults=100, textFormat="plainText", order="relevance")
        if token:
            params["pageToken"] = token
        data = yt.get("commentThreads", 1, **params)
        if not data:
            return
        for th in data.get("items", []):
            top = th["snippet"]["topLevelComment"]
            yield comment_record(top, video_id)
            for rep in th.get("replies", {}).get("comments", []):
                yield comment_record(rep, video_id, parent_id=top["id"])
        token = data.get("nextPageToken")
        pages += 1
        if not token:
            return


# --- Режимы ------------------------------------------------------------------
def cmd_collect(a, yt: YT):
    if a.video_ids:
        vids = [l.strip() for l in Path(a.video_ids).read_text().splitlines() if l.strip()]
    else:
        queries = Path(a.queries).read_text().splitlines() if a.queries else DEFAULT_QUERIES
        vids = search_videos(yt, [q for q in queries if q.strip()], a.per_query)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if out.exists():  # дозапись без дублей
        for line in out.open(encoding="utf-8"):
            seen.add(json.loads(line)["comment_id"])

    kept = 0
    with out.open("a", encoding="utf-8") as f:
        for i, vid in enumerate(vids, 1):
            n = 0
            for rec in fetch_comments(yt, vid, a.max_pages):
                if rec["comment_id"] in seen or len(rec["text"]) < a.min_len:
                    continue
                rec["kk_score"] = round(kk_score(rec["text"]), 3)
                if rec["kk_score"] < a.min_kk:
                    continue
                seen.add(rec["comment_id"])
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            kept += n
            print(f"[{i}/{len(vids)}] {vid}: +{n} (всего {kept}, квота ~{yt.units})")


def cmd_rehydrate(a, yt: YT):
    ids = [l.strip() for l in Path(a.ids).read_text().splitlines() if l.strip()]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    got = 0
    with out.open("w", encoding="utf-8") as f:
        for i in range(0, len(ids), 50):
            data = yt.get("comments", 1, part="snippet", id=",".join(ids[i:i + 50]),
                          textFormat="plainText")
            for c in (data or {}).get("items", []):
                rec = comment_record(c, c["snippet"].get("videoId", ""),
                                     c["snippet"].get("parentId"))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                got += 1
    print(f"Восстановлено {got} из {len(ids)} (удалённые комментарии недоступны)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--out", default="raw/comments.jsonl")
    c.add_argument("--queries", help="файл с поисковыми запросами, по одному на строку")
    c.add_argument("--video-ids", help="файл с ID видео (экономит квоту на поиске)")
    c.add_argument("--per-query", type=int, default=50)
    c.add_argument("--max-pages", type=int, default=20, help="страниц по 100 тредов на видео")
    c.add_argument("--min-kk", type=float, default=0.15, help="порог доли казахских сигналов")
    c.add_argument("--min-len", type=int, default=3)

    r = sub.add_parser("rehydrate")
    r.add_argument("--ids", required=True)
    r.add_argument("--out", default="raw/rehydrated.jsonl")

    a = p.parse_args()
    key = os.environ.get("YT_API_KEY")
    if not key:
        sys.exit("Задайте ключ: export YT_API_KEY=...")
    yt = YT(key)
    (cmd_collect if a.mode == "collect" else cmd_rehydrate)(a, yt)
    print(f"Готово. Потрачено ~{yt.units} units квоты.")


if __name__ == "__main__":
    main()
