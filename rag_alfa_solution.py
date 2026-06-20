"""
Alfa-Bank x MIPT RAG answers solution.

Что делает скрипт:
1) читает questions.csv, websites.csv, sample_submission.csv
2) чистит корпус и режет страницы на смысловые чанки
3) строит гибридный retrieval: word TF-IDF + char TF-IDF + optional dense embeddings
4) optional reranking cross-encoder
5) optional local LLM generation. если LLM/модели недоступны - делает сильный extractive baseline
6) сохраняет submission.csv с колонками q_id, answer_new

Запуск, быстрый baseline без GPU:
python rag_alfa_solution.py --data_dir "./Задача 3 RAG-ответы по базе знаний" --out submission_fast.csv --mode fast

Запуск сильного варианта с открытыми HF-моделями локально:
python rag_alfa_solution.py --data_dir "./Задача 3 RAG-ответы по базе знаний" --out submission_strong.csv --mode strong --use_dense --use_reranker --use_llm
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import math
import os
import pickle
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import safe_sparse_dot



# Конфиг

@dataclasses.dataclass
class Config:
    # Chunking
    chunk_words: int = 170
    chunk_overlap: int = 45
    min_chunk_chars: int = 60
    max_chunk_chars: int = 1800
    max_chunks_per_doc: int = 900  # защита от одной гигантской страницы

    # Sparse retrieval
    sparse_top_n: int = 120
    word_top_n: int = 100
    char_top_n: int = 100
    word_max_features: int = 120_000
    char_max_features: int = 140_000

    # Dense / rerank / generation
    dense_model: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model: str = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls"
    llm_model: str = "Qwen/Qwen3-1.7B"
    dense_top_n: int = 100
    rerank_top_n: int = 60
    final_top_k: int = 8
    context_max_chars: int = 5200

    # Answers
    min_answer_chars: int = 18
    max_answer_chars: int = 1150
    fallback_threshold: float = 0.025
    no_answer_text: str = (
        "В предоставленной базе знаний нет точной информации по этому вопросу. "
        "Проверьте детали в приложении Альфа-Банка, Альфа-Онлайн или обратитесь в чат поддержки."
    )

    # Performance
    batch_size: int = 8
    seed: int = 42


CFG = Config()

RUS_STOP = {
    "и", "в", "во", "на", "с", "со", "к", "ко", "по", "за", "из", "от", "до", "для", "о", "об", "обо",
    "у", "как", "что", "это", "а", "но", "или", "если", "то", "же", "ли", "бы", "не", "нет", "да", "мне",
    "меня", "мой", "моя", "мои", "мы", "вы", "ваш", "ваша", "ваши", "он", "она", "они", "его", "ее", "их",
    "можно", "нужно", "надо", "где", "когда", "почему", "зачем", "какой", "какая", "какие", "какое",
    "альфа", "банк", "альфабанк", "альфа банк", "здравствуйте", "добрый", "день", "спасибо",
}

QUESTION_WORDS = (
    "как", "где", "когда", "можно", "что", "почему", "зачем", "какой", "какая", "какие", "какое",
    "сколько", "куда", "откуда", "кто", "чем", "нужно ли", "надо ли", "есть ли", "будет ли",
)

SYNONYMS = [
    (r"\bбик\b", " банковский идентификационный код реквизиты счет счёт "),
    (r"\bинн\b", " идентификационный номер налогоплательщика реквизиты "),
    (r"\bкпп\b", " код причины постановки на учет реквизиты "),
    (r"\bр/с\b|\bрасч[её]тн", " расчетный счёт номер счета реквизиты "),
    (r"\bк/с\b|корреспондент", " корреспондентский счет реквизиты "),
    (r"к[еэ]шб[еэ]к|cashback", " кэшбэк кешбэк cashback бонусы мили категории "),
    (r"пин[ -]?код|\bpin\b", " пин-код pin код карта установить сменить "),
    (r"альфа[ -]?онлайн|alfa[ -]?online", " приложение мобильный банк интернет-банк альфа-онлайн "),
    (r"альфа[ -]?клик", " интернет-банк альфа-клик личный кабинет "),
    (r"сбп|система быстрых платеж", " СБП система быстрых платежей перевод по номеру телефона "),
    (r"qr|куар", " QR код оплата по QR "),
    (r"кредитк", " кредитная карта задолженность платеж льготный период "),
    (r"ипотек", " ипотека кредит недвижимость ставка платеж "),
    (r"брокер|инвест", " Альфа-Инвестиции брокерский счет инвестиции ИИС "),
    (r"иис", " индивидуальный инвестиционный счет ИИС брокерский счет "),
    (r"смс|sms|пуш|push", " SMS пуш уведомление код подтверждения одноразовый код "),
    (r"заблок|блокир", " блокировка разблокировать карта счет доступ "),
    (r"достав", " доставка карты курьер отделение получить карту "),
    (r"паспорт", " паспортные данные документ удостоверение личности "),
    (r"приложени", " мобильное приложение Альфа-Банк Альфа-Онлайн "),
]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def find_csv(data_dir: Path, name: str) -> Path:
    direct = data_dir / name
    if direct.exists():
        return direct
    matches = list(data_dir.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Не найден файл {name} внутри {data_dir}")
    return matches[0]


def normalize_unicode(text: str) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\ufeff", " ")
    text = text.replace("ё", "е")
    text = text.replace("–", "-").replace("-", "-")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_boilerplate(text: str) -> str:
    text = normalize_unicode(text)
    # частые футеры/куки/юридические хвосты в корпусе. не удаляем полезные телефоны и условия
    patterns = [
        r"©\s*2001[-–]\d{4}.*?(?=$|\n\n)",
        r"АО «?Альфа-?Банк»? использует файлы.*?(?=$|\n\n)",
        r"Если вы не хотите, чтобы ваши пользовательские данные обрабатывались.*?(?=$|\n\n)",
        r"АО «?Альфа-?Банк»? является оператором по обработке персональных данных.*?(?=$|\n\n)",
        r"Информация о процентных ставках по договорам банковского вклада.*?(?=$|\n\n)",
        r"Центр раскрытия корпоративной информации.*?(?=$|\n\n)",
        r"Генеральная лицензия Банка России №\s*1326.*?(?=$|\n\n)",
        r"Список отделений доступен по ссылке",
    ]
    for p in patterns:
        text = re.sub(p, " ", text, flags=re.I | re.S)
    # табличную разметку оставляем как текст, но убираем лишние палки
    text = re.sub(r"\|[-: ]+\|", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def simple_tokenize(text: str) -> List[str]:
    text = normalize_unicode(text).lower()
    return re.findall(r"[a-zа-я0-9]+(?:-[a-zа-я0-9]+)?", text)


def content_tokens(text: str) -> List[str]:
    return [t for t in simple_tokenize(text) if len(t) > 1 and t not in RUS_STOP]


def expand_query(query: str) -> str:
    q = normalize_unicode(query).lower()
    add = []
    for pat, extra in SYNONYMS:
        if re.search(pat, q, flags=re.I):
            add.append(extra)
    # частые разговорные ошибки/опечатки пользователей
    q2 = q
    q2 = q2.replace("счет", "счет счёт")
    q2 = q2.replace("кешбек", "кэшбэк кешбек")
    q2 = q2.replace("бик", "БИК")
    if len(content_tokens(q)) <= 2:
        # коротким запросам добавляем банковский контекст, чтобы char/word retrieval не проваливался
        add.append(" карта счет приложение тариф условия реквизиты платеж перевод ")
    return (q2 + " " + " ".join(add)).strip()


def url_terms(url: str) -> str:
    url = normalize_unicode(url).lower()
    url = re.sub(r"https?://(www\.)?", " ", url)
    url = re.sub(r"[/?#=&_.%-]+", " ", url)
    url = re.sub(r"\balfabank\b|\bru\b|\bhtml\b", " ", url)
    return url.strip()


def stable_hash(text: str) -> str:
    key = re.sub(r"\s+", " ", normalize_unicode(text).lower()).strip()
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def split_sentences(text: str) -> List[str]:
    text = normalize_unicode(text)
    # многие страницы уже разложены по строкам. сначала строки, потом предложения
    units = []
    for part in re.split(r"\n+", text):
        part = part.strip(" -*•\t")
        if not part:
            continue
        subs = re.split(r"(?<=[.!?])\s+(?=[А-ЯA-Z0-9])", part)
        for s in subs:
            s = s.strip(" -*•\t")
            if s:
                units.append(s)
    return units


def is_question_line(line: str) -> bool:
    s = normalize_unicode(line).strip(" -*•\t")
    if len(s) < 8 or len(s) > 220:
        return False
    low = s.lower()
    if s.endswith("?"):
        return True
    return low.startswith(QUESTION_WORDS) and ("?" in s or len(s.split()) <= 14)


def is_heading_like(line: str) -> bool:
    s = normalize_unicode(line).strip()
    if not s or len(s) > 120:
        return False
    if is_question_line(s):
        return False
    words = s.split()
    if len(words) <= 6 and not re.search(r"[.!?]$", s):
        return True
    return False


# чанкинг

@dataclasses.dataclass
class Chunk:
    chunk_id: int
    web_id: int
    url: str
    title: str
    kind: str
    text: str
    search_text: str
    is_faq: bool = False


def extract_faq_chunks(web_id: int, url: str, title: str, kind: str, text: str, start_id: int) -> List[Chunk]:
    lines = [normalize_unicode(x).strip(" -*•\t") for x in text.splitlines()]
    lines = [x for x in lines if x]
    chunks: List[Chunk] = []
    cid = start_id
    i = 0
    while i < len(lines):
        if not is_question_line(lines[i]):
            i += 1
            continue
        q_line = lines[i]
        ans = []
        j = i + 1
        while j < len(lines):
            if is_question_line(lines[j]):
                break
            # ограничиваем слишком длинные FAQ-ответы. чаще всего первые строки самые важные
            ans.append(lines[j])
            if len(" ".join(ans)) > 1400:
                break
            j += 1
        answer = " ".join(ans).strip()
        if len(answer) >= 30:
            text_chunk = f"Вопрос: {q_line}\nОтвет: {answer}"
            st = f"{title}\n{q_line}\n{q_line}\n{answer}\n{url_terms(url)}"
            chunks.append(Chunk(cid, web_id, url, title, kind, text_chunk[:CFG.max_chunk_chars], st, True))
            cid += 1
        i = max(j, i + 1)
    return chunks


def words_window_chunks(text: str, max_words: int, overlap: int) -> List[str]:
    # сохраняем порядок строк, но окна строим по словам для устойчивой длины
    words = re.findall(r"\S+", text)
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]
    chunks = []
    step = max(1, max_words - overlap)
    for start in range(0, len(words), step):
        part = words[start:start + max_words]
        if len(part) < 25 and chunks:
            break
        chunks.append(" ".join(part))
        if start + max_words >= len(words):
            break
    return chunks


def build_chunks(websites: pd.DataFrame, cfg: Config = CFG) -> List[Chunk]:
    chunks: List[Chunk] = []
    seen = set()
    next_id = 0

    # поддержка разных названий колонок из условия и фактического файла
    col_url = "url" if "url" in websites.columns else "website"
    col_text = "text" if "text" in websites.columns else "web"
    col_title = "title" if "title" in websites.columns else None
    col_kind = "kind" if "kind" in websites.columns else None

    for _, row in tqdm(websites.iterrows(), total=len(websites), desc="chunking websites"):
        web_id = int(row.get("web_id", len(chunks)))
        url = normalize_unicode(row.get(col_url, ""))
        title = normalize_unicode(row.get(col_title, "")) if col_title else ""
        kind = normalize_unicode(row.get(col_kind, "")) if col_kind else ""
        text = clean_boilerplate(row.get(col_text, ""))
        if not text or len(text) < 2:
            continue

        doc_chunks: List[Chunk] = []
        # FAQ chunks получают сильный вес при поиске
        faq = extract_faq_chunks(web_id, url, title, kind, text, next_id)
        doc_chunks.extend(faq)
        next_id += len(faq)

        # семантические блоки: группируем строки до нужного размера, headings помогают не рвать структуру
        lines = [normalize_unicode(x).strip(" -*•\t") for x in text.splitlines()]
        lines = [x for x in lines if x]
        blocks = []
        cur = []
        cur_words = 0
        for line in lines:
            lw = len(line.split())
            # заголовок начинает новый блок, если текущий уже не пустой
            if cur and is_heading_like(line) and cur_words >= 35:
                blocks.append("\n".join(cur))
                cur, cur_words = [], 0
            cur.append(line)
            cur_words += lw
            if cur_words >= cfg.chunk_words:
                blocks.append("\n".join(cur))
                # небольшой overlap строками
                tail = cur[-3:] if len(cur) >= 3 else cur[-1:]
                cur = tail[:]
                cur_words = sum(len(x.split()) for x in cur)
        if cur:
            blocks.append("\n".join(cur))

        # если страница гигантская и строки плохо разбились, добиваем sliding window
        final_blocks = []
        for b in blocks:
            if len(b.split()) > cfg.chunk_words * 1.6:
                final_blocks.extend(words_window_chunks(b, cfg.chunk_words, cfg.chunk_overlap))
            else:
                final_blocks.append(b)

        # защита от страниц на сотни тысяч символов
        if len(final_blocks) > cfg.max_chunks_per_doc:
            final_blocks = final_blocks[:cfg.max_chunks_per_doc]

        for block in final_blocks:
            block = normalize_unicode(block)
            if len(block) < cfg.min_chunk_chars:
                continue
            if len(block) > cfg.max_chunk_chars:
                # режем длинный блок аккуратно по словам
                sub_blocks = words_window_chunks(block, cfg.chunk_words, cfg.chunk_overlap)
            else:
                sub_blocks = [block]
            for sb in sub_blocks:
                sb = normalize_unicode(sb)
                if len(sb) < cfg.min_chunk_chars:
                    continue
                # дедуп по тексту, но title/url добавим в search_text
                h = stable_hash(sb)
                if h in seen:
                    continue
                seen.add(h)
                search_text = (
                    f"{title}\n{title}\n{url_terms(url)}\n"
                    f"{sb}"
                )
                doc_chunks.append(Chunk(next_id, web_id, url, title, kind, sb[:cfg.max_chunk_chars], search_text))
                next_id += 1

        # title-only chunks помогают на коротких запросах по продукту
        if title and len(title) >= 15:
            tchunk = f"{title}\n{text[:700]}"
            h = stable_hash(tchunk)
            if h not in seen:
                seen.add(h)
                doc_chunks.append(Chunk(next_id, web_id, url, title, kind, tchunk[:cfg.max_chunk_chars], f"{title}\n{title}\n{url_terms(url)}\n{text[:700]}"))
                next_id += 1

        chunks.extend(doc_chunks)

    # переиндексация chunk_id подряд
    for i, ch in enumerate(chunks):
        ch.chunk_id = i
    return chunks


# retrieval

class SparseRetriever:
    def __init__(self, chunks: List[Chunk], cfg: Config = CFG):
        self.chunks = chunks
        self.cfg = cfg
        texts = [ch.search_text[:2200] for ch in chunks]
        texts_char = [(ch.title + "\n" + ch.text[:700])[:1000] for ch in chunks]
        print(f"Building sparse hashing index on {len(texts)} chunks...")
        # важна скорость итераций. HashingVectorizer не строит словарь,
        # поэтому на корпусе с длинными страницами работает заметно быстрее TF-IDF vocabulary fit
        # нормировка L2 дает cosine-like similarity
        self.word_vec = HashingVectorizer(
            token_pattern=r"(?u)\b[0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё-]+\b",
            lowercase=True,
            ngram_range=(1, 2),
            alternate_sign=False,
            norm="l2",
            n_features=2**19,
            dtype=np.float32,
        )
        self.char_vec = HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 4),
            lowercase=True,
            alternate_sign=False,
            norm="l2",
            n_features=2**18,
            dtype=np.float32,
        )
        t_start = time.time()
        self.Xw = self.word_vec.transform(texts)
        print(f"  word hashing done in {time.time() - t_start:.1f}s")
        t_start = time.time()
        self.Xc = self.char_vec.transform(texts_char)
        print(f"  char hashing done in {time.time() - t_start:.1f}s")

    def _top_from_scores(self, scores: np.ndarray, top_n: int) -> List[Tuple[int, float]]:
        if scores.size == 0:
            return []
        top_n = min(top_n, scores.size)
        idx = np.argpartition(-scores, top_n - 1)[:top_n]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]

    def search(self, query: str, top_n: Optional[int] = None) -> List[Tuple[int, float]]:
        top_n = top_n or self.cfg.sparse_top_n
        q = expand_query(query)
        qw = self.word_vec.transform([q])
        sw = safe_sparse_dot(qw, self.Xw.T, dense_output=True).ravel()
        # char_wb особенно полезен для коротких запросов и опечаток,
        # но на длинных клиентских сообщениях может быть медленным из-за большого числа n-грамм
        qtoks_for_char = simple_tokenize(q)
        if len(q) <= 110 and len(qtoks_for_char) <= 14:
            qc = self.char_vec.transform([q[:110]])
            sc = safe_sparse_dot(qc, self.Xc.T, dense_output=True).ravel()
            scores = 0.62 * sw + 0.38 * sc
        else:
            scores = sw.astype(np.float32, copy=False)

        # небольшой буст FAQ и title/url совпадений
        q_tokens = set(content_tokens(q))
        if q_tokens:
            for idx in np.argpartition(-scores, min(len(scores)-1, max(500, top_n*4)))[:min(len(scores), max(500, top_n*4))]:
                ch = self.chunks[int(idx)]
                title_tokens = set(content_tokens(ch.title))
                if ch.is_faq:
                    scores[idx] *= 1.08
                if title_tokens and len(q_tokens & title_tokens) >= 1:
                    scores[idx] *= 1.04
        return self._top_from_scores(scores, top_n)


class DenseRetriever:
    def __init__(self, chunks: List[Chunk], cfg: Config = CFG, cache_dir: str = "cache"):
        self.chunks = chunks
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.emb = None
        self.enabled = False
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import torch  # noqa
        except Exception as e:
            print(f"[WARN] sentence-transformers недоступен, dense retrieval выключен: {e}")
            return
        try:
            device = "cuda" if self._cuda_available() else "cpu"
            print(f"Loading dense model {cfg.dense_model} on {device}...")
            # на T4 лучше грузить transformer-модель в fp16, чтобы осталось место для reranker и LLM (если в колаб)
            try:
                import torch
                if device == "cuda":
                    self.model = SentenceTransformer(
                        cfg.dense_model,
                        device=device,
                        model_kwargs={"torch_dtype": torch.float16},
                    )
                else:
                    self.model = SentenceTransformer(cfg.dense_model, device=device)
            except TypeError:
                self.model = SentenceTransformer(cfg.dense_model, device=device)
            self.emb = self._load_or_build_embeddings()
            self.enabled = True
        except Exception as e:
            print(f"[WARN] dense retrieval выключен из-за ошибки загрузки модели: {e}")
            self.enabled = False
            self.model = None
            self.emb = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _prefix_doc(self, text: str) -> str:
        if "e5" in self.cfg.dense_model.lower():
            return "passage: " + text
        return text

    def _prefix_query(self, text: str) -> str:
        if "e5" in self.cfg.dense_model.lower():
            return "query: " + text
        return text

    def _cache_key(self) -> Path:
        h = hashlib.md5((self.cfg.dense_model + str(len(self.chunks)) + "v5_qwen3").encode()).hexdigest()[:12]
        return self.cache_dir / f"dense_{h}.npy"

    def _load_or_build_embeddings(self) -> np.ndarray:
        path = self._cache_key()
        if path.exists():
            print(f"Loading cached dense embeddings: {path}")
            emb = np.load(path)
            return emb.astype(np.float32)
        docs = [self._prefix_doc((ch.title + "\n" + ch.text)[:1800]) for ch in self.chunks]
        print(f"Encoding {len(docs)} chunks with dense model...")
        emb = self.model.encode(
            docs,
            batch_size=self.cfg.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        np.save(path, emb)
        return emb

    def search(self, query: str, top_n: Optional[int] = None) -> List[Tuple[int, float]]:
        if not self.enabled or self.model is None or self.emb is None:
            return []
        top_n = top_n or self.cfg.dense_top_n
        q = self._prefix_query(expand_query(query))
        if "qwen3-embedding" in self.cfg.dense_model.lower():
            # Qwen3 Embedding instruction-aware: query-side prompt обычно повышает retrieval
            task = "Given a Russian banking customer question, retrieve relevant passages from Alfa-Bank website that answer the query"
            try:
                q_emb = self.model.encode(
                    [q],
                    prompt_name="query",
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).astype(np.float32)[0]
            except Exception:
                q_emb = self.model.encode(
                    [task + "\nQuery: " + q],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).astype(np.float32)[0]
        else:
            q_emb = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)[0]
        scores = self.emb @ q_emb
        top_n = min(top_n, len(scores))
        idx = np.argpartition(-scores, top_n - 1)[:top_n]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


class HybridRetriever:
    def __init__(self, chunks: List[Chunk], cfg: Config, cache_dir: str, use_dense: bool):
        self.chunks = chunks
        self.cfg = cfg
        self.sparse = SparseRetriever(chunks, cfg)
        self.dense = DenseRetriever(chunks, cfg, cache_dir=cache_dir) if use_dense else None

    def search(self, query: str, top_n: int = 160) -> List[Tuple[int, float, Dict[str, float]]]:
        sparse = self.sparse.search(query, self.cfg.sparse_top_n)
        dense = self.dense.search(query, self.cfg.dense_top_n) if self.dense is not None else []

        score_map: Dict[int, Dict[str, float]] = defaultdict(lambda: {"sparse": 0.0, "dense": 0.0})
        # нормализация по max в каждом канале
        if sparse:
            mx = max(s for _, s in sparse) or 1.0
            for idx, s in sparse:
                score_map[idx]["sparse"] = max(score_map[idx]["sparse"], s / mx)
        if dense:
            mx = max(s for _, s in dense) or 1.0
            for idx, s in dense:
                # dense cosine может быть в узком диапазоне, min-max по top помогает
                score_map[idx]["dense"] = max(score_map[idx]["dense"], s / mx)

        items = []
        for idx, parts in score_map.items():
            if dense:
                score = 0.52 * parts["sparse"] + 0.48 * parts["dense"]
            else:
                score = parts["sparse"]
            if self.chunks[idx].is_faq:
                score *= 1.04
            items.append((idx, float(score), dict(parts)))
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:top_n]


# reranking

class CrossEncoderReranker:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.model = None
        self.enabled = False
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            device = "cuda" if self._cuda_available() else "cpu"
            print(f"Loading reranker {cfg.reranker_model} on {device}...")
            max_len = 1024 if "qwen3-reranker" in cfg.reranker_model.lower() else 512
            self.model = CrossEncoder(cfg.reranker_model, device=device, max_length=max_len)
            if device == "cuda":
                # crossEncoder часто грузится в fp32, half экономит VRAM на колаб T4
                try:
                    self.model.model.half()
                except Exception:
                    pass
            self.enabled = True
        except Exception as e:
            print(f"[WARN] CrossEncoder reranker выключен: {e}")
            self.enabled = False

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def rerank(self, query: str, chunks: List[Chunk], candidates: List[Tuple[int, float, Dict[str, float]]]) -> List[Tuple[int, float, Dict[str, float]]]:
        if not self.enabled or self.model is None or not candidates:
            return candidates
        cand = candidates[: self.cfg.rerank_top_n]
        docs = [(chunks[idx].title + "\n" + chunks[idx].text)[:1800] for idx, _, _ in cand]

        if "qwen3-reranker" in self.cfg.reranker_model.lower():
            task = "Given a Russian banking customer question, retrieve relevant passages from Alfa-Bank website that answer the query"

            def format_query(q: str) -> str:
                prefix = (
                    "<|im_start|>system\n"
                    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
                    'Note that the answer can only be "yes" or "no".'
                    "<|im_end|>\n<|im_start|>user\n"
                )
                return f"{prefix}<Instruct>: {task}\n<Query>: {q}\n"

            def format_doc(d: str) -> str:
                suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                return f"<Document>: {d}{suffix}"

            pairs = [(format_query(query), format_doc(doc)) for doc in docs]
            pred_bs = 4
        else:
            pairs = [(query, doc) for doc in docs]
            pred_bs = 16
        try:
            scores = self.model.predict(pairs, batch_size=pred_bs, show_progress_bar=False)
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            # sigmoid/min-max для устойчивого combine
            if scores.max() > scores.min():
                ns = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                ns = np.ones_like(scores) * 0.5
            reranked = []
            for (idx, old_score, parts), rs in zip(cand, ns):
                new_score = 0.72 * float(rs) + 0.28 * old_score
                parts = dict(parts)
                parts["rerank"] = float(rs)
                reranked.append((idx, new_score, parts))
            # добавляем хвост без cross-score ниже
            tail = candidates[self.cfg.rerank_top_n:]
            reranked.extend([(idx, sc * 0.75, parts) for idx, sc, parts in tail])
            reranked.sort(key=lambda x: x[1], reverse=True)
            return reranked
        except Exception as e:
            print(f"[WARN] rerank failed, fallback to hybrid scores: {e}")
            return candidates


def lexical_rerank(query: str, chunks: List[Chunk], candidates: List[Tuple[int, float, Dict[str, float]]]) -> List[Tuple[int, float, Dict[str, float]]]:
    # легкий fallback rerank на overlap содержательных токенов + числа/аббревиатуры
    q_tokens = set(content_tokens(expand_query(query)))
    q_numbers = set(re.findall(r"\d+", query))
    out = []
    for rank, (idx, base, parts) in enumerate(candidates):
        ch = chunks[idx]
        ct = content_tokens(ch.title + " " + ch.text)
        cset = set(ct)
        overlap = len(q_tokens & cset) / max(1, len(q_tokens))
        nums = set(re.findall(r"\d+", ch.text))
        num_bonus = 0.08 if q_numbers and (q_numbers & nums) else 0.0
        title_bonus = 0.05 if (set(content_tokens(ch.title)) & q_tokens) else 0.0
        faq_bonus = 0.06 if ch.is_faq else 0.0
        new = 0.70 * base + 0.30 * overlap + num_bonus + title_bonus + faq_bonus - rank * 0.0005
        p = dict(parts)
        p["lex"] = overlap
        out.append((idx, float(new), p))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


# контекст и answering

def dedup_context(chunks: List[Chunk], ranked: List[Tuple[int, float, Dict[str, float]]], top_k: int, max_chars: int) -> List[Tuple[Chunk, float]]:
    selected: List[Tuple[Chunk, float]] = []
    seen_texts: List[set] = []
    total = 0
    for idx, score, _ in ranked:
        ch = chunks[idx]
        toks = set(content_tokens(ch.text))
        if not toks:
            continue
        # убираем почти дубли соседних chunks с одной страницы
        duplicate = False
        for st in seen_texts:
            inter = len(toks & st)
            union = len(toks | st)
            if union and inter / union > 0.78:
                duplicate = True
                break
        if duplicate:
            continue
        add_len = len(ch.text) + len(ch.title) + 80
        if selected and total + add_len > max_chars:
            break
        selected.append((ch, score))
        seen_texts.append(toks)
        total += add_len
        if len(selected) >= top_k:
            break
    return selected


def format_context(selected: List[Tuple[Chunk, float]]) -> str:
    parts = []
    for i, (ch, score) in enumerate(selected, 1):
        title = ch.title[:180]
        text = ch.text[:1600]
        parts.append(f"[Фрагмент {i}]\nЗаголовок: {title}\nТекст: {text}")
    return "\n\n".join(parts)


def sentence_score(query: str, sentence: str, chunk_score: float, rank: int) -> float:
    q_tokens = set(content_tokens(expand_query(query)))
    s_tokens = content_tokens(sentence)
    if not s_tokens:
        return -1.0
    s_set = set(s_tokens)
    overlap = len(q_tokens & s_set) / max(1, len(q_tokens))
    density = len(q_tokens & s_set) / max(1, len(s_set))
    digit_bonus = 0.05 if re.search(r"\d", sentence) else 0.0
    length = len(sentence)
    # предпочитаем информативные, но не огромные предложения
    length_penalty = 0.0
    if length < 35:
        length_penalty -= 0.08
    if length > 420:
        length_penalty -= 0.08
    return 0.52 * chunk_score + 0.30 * overlap + 0.10 * density + digit_bonus + length_penalty - 0.012 * rank


def compress_extractive(query: str, selected: List[Tuple[Chunk, float]], cfg: Config = CFG) -> str:
    if not selected:
        return cfg.no_answer_text
    candidates = []
    for rank, (ch, cscore) in enumerate(selected):
        text = ch.text
        # FAQ: вопросная строка не должна попадать в ответ, но сама связка полезна
        text = re.sub(r"^Вопрос:.*?\nОтвет:\s*", "", text, flags=re.S)
        for s in split_sentences(text):
            s = re.sub(r"^[•*\-]+\s*", "", s).strip()
            s = re.sub(r"\s+", " ", s)
            if len(s) < 25:
                continue
            if len(s) > 520:
                # режем длинные строки по ; или , только если очень большие
                parts = re.split(r"(?<=[.;])\s+", s)
                if len(parts) > 1:
                    for p in parts:
                        if 25 <= len(p) <= 520:
                            candidates.append((sentence_score(query, p, cscore, rank), p, rank))
                    continue
                s = s[:520].rsplit(" ", 1)[0]
            candidates.append((sentence_score(query, s, cscore, rank), s, rank))
    if not candidates:
        return cfg.no_answer_text
    candidates.sort(key=lambda x: x[0], reverse=True)

    q_len = len(content_tokens(query))
    # короткий вопрос = короткий ответ. recall-l штрафует слишком длинные ответы
    if q_len <= 2:
        target_chars = 520
        max_sents = 3
    elif q_len <= 6:
        target_chars = 760
        max_sents = 4
    else:
        target_chars = 930
        max_sents = 5
    target_chars = min(target_chars, cfg.max_answer_chars)

    selected_sents = []
    selected_sets = []
    total = 0
    for sc, s, rank in candidates[:35]:
        toks = set(content_tokens(s))
        if not toks:
            continue
        too_similar = False
        for old in selected_sets:
            inter = len(toks & old)
            union = len(toks | old)
            if union and inter / union > 0.72:
                too_similar = True
                break
        if too_similar:
            continue
        if total + len(s) > target_chars and selected_sents:
            continue
        selected_sents.append(s)
        selected_sets.append(toks)
        total += len(s) + 1
        if len(selected_sents) >= max_sents or total >= target_chars * 0.9:
            break

    # возвращаем в порядке источников/появления, а не по score - так ответ читабельнее
    # но если выбран только один чанк, порядок уже естественный
    def pos_key(sent: str) -> int:
        ctx = " ".join(ch.text for ch, _ in selected)
        p = ctx.find(sent)
        return p if p >= 0 else 10**9

    selected_sents = sorted(selected_sents, key=pos_key)
    answer = " ".join(selected_sents)
    return postprocess_answer(answer, cfg)


def postprocess_answer(answer: str, cfg: Config = CFG) -> str:
    answer = normalize_unicode(answer)
    # убираем артефакты промпта/генерации
    answer = re.sub(r"(?i)^ответ\s*:\s*", "", answer).strip()
    answer = re.sub(r"(?i)согласно\s+(фрагменту|контексту|источнику)\s*\d*[,.:]?\s*", "", answer)
    answer = re.sub(r"(?i)в\s+фрагменте\s*\d+\s*(говорится|указано|сказано)[,.:]?\s*", "", answer)
    answer = re.sub(r"\[/?INST\]|<\|.*?\|>", " ", answer)
    answer = answer.replace("**", "")
    # убираем явные цитаты источников
    answer = re.sub(r"\s*Источник:\s*https?://\S+", "", answer, flags=re.I)
    answer = re.sub(r"\s+", " ", answer).strip(" -\n\t")
    # нормализуем частые варианты
    answer = answer.replace("Альфа - Онлайн", "Альфа-Онлайн").replace("Альфа - Банк", "Альфа-Банк")
    answer = answer.replace("кешбэк", "кэшбэк")
    if len(answer) > cfg.max_answer_chars:
        cut = answer[: cfg.max_answer_chars]
        # режем по последней точке, если она не слишком рано
        p = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        if p > cfg.max_answer_chars * 0.55:
            answer = cut[: p + 1]
        else:
            answer = cut.rsplit(" ", 1)[0] + "."
    if not answer or len(answer) < cfg.min_answer_chars:
        return cfg.no_answer_text
    return answer


class LocalLLMAnswerer:
    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.enabled = False
        self.tokenizer = None
        self.model = None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception as e:
            print(f"[WARN] transformers/bitsandbytes недоступны, LLM generation выключен: {e}")
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            if not torch.cuda.is_available():
                print("[WARN] CUDA/GPU недоступна. LLM выключена: на CPU генерация будет слишком медленной.")
                return

            print(f"Loading local LLM {cfg.llm_model} in 4-bit...")
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.llm_model,
                device_map="auto",
                dtype=torch.float16,
                trust_remote_code=True,
                quantization_config=bnb_config,
            )
            self.model.eval()
            self.enabled = True
        except Exception as e:
            print(f"[WARN] LLM generation выключен из-за ошибки загрузки модели: {e}")
            self.enabled = False

    def build_prompt(self, query: str, context: str, sample_answer: Optional[str] = None) -> str:
        sample_block = ""
        if sample_answer is not None and is_usable_sample_answer(sample_answer):
            sample = postprocess_answer(str(sample_answer), self.cfg)
            sample_block = (
                "\nПРЕДВАРИТЕЛЬНЫЙ ОТВЕТ ИЗ sample_submission. "
                "Он может быть полезен, но его нужно проверить по найденной информации. "
                "Не копируй его, если он противоречит найденным фрагментам.\n"
                f"{sample[:1400]}\n"
            )
        return (
            "Ты ассистент Альфа-Банка. Нужно ответить на вопрос клиента строго по найденной информации.\n"
            "Цель: дать ответ, который максимально покрывает эталонный смысл, но без лишней воды.\n"
            "Правила ответа:\n"
            "1. Используй только факты из найденной информации и проверенного предварительного ответа.\n"
            "2. Не выдумывай условия, суммы, сроки, комиссии, документы и требования.\n"
            "3. Не пиши слова: контекст, фрагмент, источник, RAG, база знаний.\n"
            "4. Ответ должен быть коротким, но содержательным: обычно 3-5 предложений.\n"
            "5. Если вопрос просит инструкцию, дай шаги коротко.\n"
            "6. Если точных данных нет, не отвечай 'нет ответа' сразу: укажи, где проверить - приложение, Альфа-Онлайн, чат поддержки или офис.\n"
            "7. Без markdown-таблиц. Маркированный список используй только если он реально нужен.\n\n"
            f"ВОПРОС КЛИЕНТА:\n{query}\n"
            f"{sample_block}\n"
            f"НАЙДЕННАЯ ИНФОРМАЦИЯ:\n{context}\n\n"
            "КРАТКИЙ ОТВЕТ:"
        )

    def generate(self, query: str, selected: List[Tuple[Chunk, float]], sample_answer: Optional[str] = None) -> str:
        if not self.enabled or self.model is None or self.tokenizer is None:
            return ""
        context = format_context(selected)
        prompt = self.build_prompt(query, context, sample_answer=sample_answer)
        try:
            import torch
            messages = [
                {"role": "system", "content": "Ты аккуратный банковский ассистент. Отвечай только по данным, кратко и без рассуждений."},
                {"role": "user", "content": prompt},
            ]

            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    prompt_text = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            else:
                prompt_text = prompt

            enc = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=3900,
            ).to(self.model.device)
            input_ids = enc["input_ids"]
            attention_mask = enc.get("attention_mask")
            with torch.no_grad():
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=180,
                    do_sample=False,
                    repetition_penalty=1.08,
                    no_repeat_ngram_size=6,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            gen = out[0][input_ids.shape[-1]:]
            ans = self.tokenizer.decode(gen, skip_special_tokens=True)
            # на всякий случай удаляем thinking-теги, если модель их всё-таки вывела
            ans = re.sub(r"(?is)<think>.*?</think>", " ", ans)
            return postprocess_answer(ans, self.cfg)
        except Exception as e:
            print(f"[WARN] LLM generate failed: {e}")
            return ""


# пайплайн

class RAGPipeline:
    def __init__(self, websites: pd.DataFrame, cfg: Config, cache_dir: str, use_dense: bool, use_reranker: bool, use_llm: bool):
        self.cfg = cfg
        seed_everything(cfg.seed)
        chunk_cache = Path(cache_dir) / "chunks_v4.pkl"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        if chunk_cache.exists():
            print(f"Loading cached chunks: {chunk_cache}")
            with open(chunk_cache, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            self.chunks = build_chunks(websites, cfg)
            with open(chunk_cache, "wb") as f:
                pickle.dump(self.chunks, f)
        print(f"Total chunks: {len(self.chunks)}")
        self.retriever = HybridRetriever(self.chunks, cfg, cache_dir=cache_dir, use_dense=use_dense)
        self.reranker = CrossEncoderReranker(cfg) if use_reranker else None
        self.llm = LocalLLMAnswerer(cfg) if use_llm else None

    def answer_one(self, query: str, sample_answer: Optional[str] = None, sample_strategy: str = "none") -> Tuple[str, Dict]:
        dbg = bool(os.getenv("RAG_DEBUG"))
        if sample_strategy in {"fill_missing", "keep_usable"} and is_usable_sample_answer(sample_answer):
            ans = postprocess_answer(str(sample_answer), self.cfg)
            return ans, {"confidence": 1.0, "top": [], "used_sample": True}
        if dbg:
            print(f"DEBUG query: {query[:120]}", flush=True)
        t = time.time()
        ranked = self.retriever.search(query, top_n=max(self.cfg.sparse_top_n, self.cfg.dense_top_n))
        if dbg:
            print(f"  retrieval {len(ranked)} in {time.time()-t:.2f}s", flush=True)
        if not ranked:
            return self.cfg.no_answer_text, {"confidence": 0.0, "top": []}
        t = time.time()
        if self.reranker is not None and self.reranker.enabled:
            ranked = self.reranker.rerank(query, self.chunks, ranked)
        else:
            ranked = lexical_rerank(query, self.chunks, ranked)
        if dbg:
            print(f"  rerank in {time.time()-t:.2f}s", flush=True)
        t = time.time()
        selected = dedup_context(self.chunks, ranked, self.cfg.final_top_k, self.cfg.context_max_chars)
        confidence = float(ranked[0][1]) if ranked else 0.0
        if dbg:
            print(f"  dedup selected {len(selected)} in {time.time()-t:.2f}s", flush=True)

        t = time.time()
        extractive = compress_extractive(query, selected, self.cfg)
        if dbg:
            print(f"  extractive in {time.time()-t:.2f}s", flush=True)
        answer = ""
        if self.llm is not None and self.llm.enabled and confidence >= self.cfg.fallback_threshold:
            hint = sample_answer if sample_strategy == "hint" else None
            answer = self.llm.generate(query, selected, sample_answer=hint)
            # если LLM ушла в отказ/воду, лучше extractive или usable sample
            bad = ["нет информации", "не могу", "не указан", "не найдена", "нет точной информации"]
            if len(answer) < 30 or (any(x in answer.lower() for x in bad) and len(extractive) > len(answer) + 40):
                if sample_strategy == "hint" and is_usable_sample_answer(sample_answer):
                    answer = postprocess_answer(str(sample_answer), self.cfg)
                else:
                    answer = extractive
        else:
            answer = extractive

        # при очень низкой уверенности не оставляем опасную конкретику
        if confidence < self.cfg.fallback_threshold and len(answer) < 80:
            answer = self.cfg.no_answer_text
        answer = postprocess_answer(answer, self.cfg)
        meta = {
            "confidence": confidence,
            "top": [
                {
                    "web_id": self.chunks[idx].web_id,
                    "score": score,
                    "title": self.chunks[idx].title,
                    "url": self.chunks[idx].url,
                    "is_faq": self.chunks[idx].is_faq,
                }
                for idx, score, _ in ranked[:5]
            ],
        }
        return answer, meta

    def run_questions(
        self,
        questions: pd.DataFrame,
        limit: Optional[int] = None,
        sample_answers: Optional[Dict[int, str]] = None,
        sample_strategy: str = "none",
    ) -> pd.DataFrame:
        rows = []
        it = questions if limit is None else questions.head(limit)
        sample_answers = sample_answers or {}
        for _, row in tqdm(it.iterrows(), total=len(it), desc="answering"):
            qid = int(row["q_id"])
            query = normalize_unicode(row["query"])
            sample_answer = sample_answers.get(qid)
            ans, meta = self.answer_one(query, sample_answer=sample_answer, sample_strategy=sample_strategy)
            rows.append({"q_id": qid, "answer_new": ans})
        return pd.DataFrame(rows)


# хелперы локальной валидации

def quick_quality_report(sub: pd.DataFrame) -> None:
    lens = sub["answer_new"].astype(str).str.len()
    print("\nSubmission quality sanity check")
    print("rows:", len(sub))
    print("empty:", int((lens == 0).sum()))
    print("very_short<=15:", int((lens <= 15).sum()))
    print("len mean/median/p90/max:", round(lens.mean(), 1), int(lens.median()), int(lens.quantile(0.9)), int(lens.max()))
    print("'нет ответа' count:", int(sub["answer_new"].astype(str).str.lower().str.contains(r"^нет ответа").sum()))


def is_usable_sample_answer(x: str) -> bool:
    x = normalize_unicode(str(x)).strip()
    low = x.lower().strip(" .")
    if not x or len(x) <= 20:
        return False
    if low.startswith("нет ответа") or low.startswith("в предоставленной базе знаний нет") or low in {"нет", "да"}:
        return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Папка с questions.csv, websites.csv, sample_submission.csv")
    p.add_argument("--out", type=str, default="submission.csv")
    p.add_argument("--cache_dir", type=str, default="cache_alfa_rag")
    p.add_argument("--mode", type=str, choices=["fast", "strong"], default="fast")
    p.add_argument("--use_dense", action="store_true", help="Включить dense embeddings через sentence-transformers")
    p.add_argument("--use_reranker", action="store_true", help="Включить cross-encoder reranker")
    p.add_argument("--use_llm", action="store_true", help="Включить локальную LLM генерацию через transformers")
    p.add_argument("--dense_model", type=str, default=CFG.dense_model)
    p.add_argument("--reranker_model", type=str, default=CFG.reranker_model)
    p.add_argument("--llm_model", type=str, default=CFG.llm_model)
    p.add_argument("--limit", type=int, default=None, help="Для отладки: обработать первые N вопросов")
    p.add_argument("--sample_blend", action="store_true", help="Старый режим: использовать непустые ответы из sample_submission как baseline, RAG запускать для Нет ответа")
    p.add_argument("--sample_strategy", type=str, default="none", choices=["none", "hint", "fill_missing", "keep_usable"], help="Как использовать sample_submission: none=игнорировать; hint=дать LLM как черновик; fill_missing/keep_usable=оставить хорошие sample-ответы и генерировать только отсутствующие")
    p.add_argument("--final_top_k", type=int, default=CFG.final_top_k)
    p.add_argument("--context_max_chars", type=int, default=CFG.context_max_chars)
    p.add_argument("--max_answer_chars", type=int, default=CFG.max_answer_chars)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dataclasses.replace(
        CFG,
        dense_model=args.dense_model,
        reranker_model=args.reranker_model,
        llm_model=args.llm_model,
        final_top_k=args.final_top_k,
        context_max_chars=args.context_max_chars,
        max_answer_chars=args.max_answer_chars,
    )
    if args.mode == "strong":
        no_flags = not (args.use_dense or args.use_reranker or args.use_llm)
        use_dense = bool(args.use_dense or no_flags)
        use_reranker = bool(args.use_reranker or no_flags)
        use_llm = bool(args.use_llm or no_flags)
    else:
        use_dense = args.use_dense
        use_reranker = args.use_reranker
        use_llm = args.use_llm

    data_dir = Path(args.data_dir)
    questions_path = find_csv(data_dir, "questions.csv")
    websites_path = find_csv(data_dir, "websites.csv")
    sample_path = find_csv(data_dir, "sample_submission.csv")
    print("questions:", questions_path)
    print("websites:", websites_path)
    print("sample:", sample_path)

    questions = pd.read_csv(questions_path)
    websites = pd.read_csv(websites_path)
    sample = pd.read_csv(sample_path)
    assert "q_id" in questions.columns and "query" in questions.columns, "questions.csv должен содержать q_id, query"
    assert "q_id" in sample.columns, "sample_submission.csv должен содержать q_id"
    answer_col = "answer_new" if "answer_new" in sample.columns else [c for c in sample.columns if c != "q_id"][0]
    if answer_col != "answer_new":
        print(f"[INFO] В sample колонка ответа называется {answer_col}; итог будет сохранен с тем же названием.")

    t0 = time.time()
    rag = RAGPipeline(websites, cfg, args.cache_dir, use_dense=use_dense, use_reranker=use_reranker, use_llm=use_llm)

    col = answer_col
    sample_map = dict(zip(sample["q_id"].astype(int), sample[col].astype(str)))
    if args.sample_blend:
        # оставлено для совместимости. эквивалентно --sample_strategy fill_missing
        args.sample_strategy = "fill_missing"

    if args.sample_strategy in {"fill_missing", "keep_usable"}:
        print(f"[INFO] sample_strategy={args.sample_strategy}: хорошие sample-ответы сохраняются, RAG/LLM генерирует только плохие/Нет ответа.")
    elif args.sample_strategy == "hint":
        print("[INFO] sample_strategy=hint: sample_answer передается LLM как черновик/подсказка, но ответ генерируется заново.")

    sub = rag.run_questions(
        questions,
        limit=args.limit,
        sample_answers=sample_map if args.sample_strategy != "none" else None,
        sample_strategy=args.sample_strategy,
    )

    if answer_col != "answer_new":
        sub = sub.rename(columns={"answer_new": answer_col})

    # гарантируем порядок и полный набор q_id как в sample_submission
    out = sample[["q_id"]].merge(sub, on="q_id", how="left")
    out[col] = out[col].fillna(CFG.no_answer_text).astype(str).map(lambda x: postprocess_answer(x, cfg))
    out_path = Path(args.out)
    out.to_csv(out_path, index=False)
    quick_quality_report(out.rename(columns={col: "answer_new"}))
    print(f"Saved: {out_path.resolve()}")
    print(f"Elapsed: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
