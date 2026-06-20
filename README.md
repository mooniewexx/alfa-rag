# Финальное решение Alfa-Банк × МФТИ: RAG-ответы по базе знаний

## Что загружать на платформу

На платформу загружается только итоговый CSV-файл, например:

```text
submission_qwen3_17b_hint.csv
```

Файл должен иметь структуру как `sample_submission.csv`: колонка `q_id` и колонка ответа (`answer_new`, если такая колонка есть в sample).

Код и ноутбук на платформу обычно не загружаются, но их нужно сохранить: топ-10 решений могут пройти code review.

## Что внутри решения

Pipeline:

1. чтение `questions.csv`, `websites.csv`, `sample_submission.csv`;
2. очистка и чанкинг страниц;
3. sparse retrieval: word hashing + char hashing;
4. dense retrieval: `Qwen/Qwen3-Embedding-0.6B`;
5. reranking: `tomaarsen/Qwen3-Reranker-0.6B-seq-cls`;
6. generation: `Qwen/Qwen3-1.7B` в 4-bit через `bitsandbytes`;
7. сохранение `submission.csv`.

Закрытые API не используются. Все модели запускаются локально в Colab.

## Как запустить в Colab

1. Открой `Alfa_RAG_FINAL_WORKING_Colab.ipynb` в Google Colab.
2. Включи GPU: `Runtime → Change runtime type → T4 GPU`.
3. Запусти раздел 1 установки. Runtime перезапустится — это нормально.
4. После перезапуска продолжи с раздела 2.
5. Загрузи zip-архив с данными задачи.
6. Запусти проверку окружения, LLM-check и smoke-test.
7. Запусти основной сабмит.
8. Скачай `submission_qwen3_17b_hint.csv` и загрузи его на платформу.

## Если возникла проблема с окружением

Не ставь зависимости так:

```bash
pip install -U bitsandbytes>=0.46.1
```

Правильно:

```bash
pip install -U "bitsandbytes>=0.46.1"
```

В финальном ноутбуке зависимости уже зафиксированы безопасно:

```text
pandas==2.2.2
numpy==2.2.6
bitsandbytes>=0.46.1
transformers>=4.51.0
```

## Основная команда внутри Colab

```bash
python /content/rag_alfa_solution.py \
  --data_dir /content/alfa_data \
  --out /content/submission_qwen3_17b_hint.csv \
  --cache_dir /content/rag_cache_qwen3_17b \
  --mode strong \
  --use_dense \
  --use_reranker \
  --use_llm \
  --dense_model Qwen/Qwen3-Embedding-0.6B \
  --reranker_model tomaarsen/Qwen3-Reranker-0.6B-seq-cls \
  --llm_model Qwen/Qwen3-1.7B \
  --sample_strategy hint \
  --final_top_k 7 \
  --context_max_chars 4200 \
  --max_answer_chars 800
```

## Опционально

Если есть время и Colab не падает по памяти, можно попробовать генератор `Qwen/Qwen3-4B-Instruct-2507`. Команда есть в последнем разделе ноутбука.
