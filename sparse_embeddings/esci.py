import logging
from collections.abc import Iterable

from datasets import Dataset, load_dataset

LOGGER = logging.getLogger(__name__)

ESCI_DATASET_NAME = "tasksource/esci"
POSITIVE_LABELS = frozenset({"Exact", "Substitute"})
SUPPORTED_LOCALES = ("us", "es", "jp", "all")


def normalize_locale(locale: str) -> str:
    normalized = locale.lower()
    if normalized not in SUPPORTED_LOCALES:
        raise ValueError(f"Unsupported locale {locale!r}. Expected one of {SUPPORTED_LOCALES}.")
    return normalized


def normalize_text(value) -> str:
    if value is None:
        return ""
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"none", "null", "nan"}:
        return ""
    return normalized


def split_bullets(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = value.splitlines()
    elif isinstance(value, Iterable):
        candidates = list(value)
    else:
        candidates = [value]
    bullets = []
    for bullet in candidates:
        normalized = normalize_text(bullet)
        if normalized:
            bullets.append(normalized)
    return bullets


def build_product_text(
    title,
    brand="",
    description="",
    bullets=None,
    max_chars: int = 512,
) -> str:
    title_text = normalize_text(title)
    if not title_text:
        return ""

    parts = []
    brand_text = normalize_text(brand)
    if brand_text:
        parts.append(f"[{brand_text}]")
    parts.append(title_text)

    description_text = normalize_text(description)
    if description_text:
        parts.append(f"| {description_text[:200]}")

    bullet_items = split_bullets(bullets)[:3]
    if bullet_items:
        parts.append(f"| {' | '.join(bullet_items)}")

    text = " ".join(parts)
    return text[:max_chars].strip()


def is_locale_match(row_locale: str, locale: str) -> bool:
    return locale == "all" or row_locale == locale


def is_positive_label(label: str) -> bool:
    return normalize_text(label) in POSITIVE_LABELS


def make_query_key(locale: str, query_id) -> str:
    return f"{locale}:{query_id}"


def make_corpus_key(locale: str, product_id) -> str:
    return f"{locale}:{product_id}"


def iter_esci_rows(split: str):
    return load_dataset(ESCI_DATASET_NAME, split=split, streaming=True)


def build_pair_dataset(
    split: str,
    locale: str = "all",
    max_samples: int | None = None,
    selected_query_keys: set[str] | list[str] | tuple[str, ...] | None = None,
) -> Dataset:
    locale = normalize_locale(locale)
    queries: list[str] = []
    documents: list[str] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    selected_query_key_set = set(selected_query_keys) if selected_query_keys is not None else None
    seen_selected_query_keys: set[str] = set()

    for row in iter_esci_rows(split):
        row_locale = normalize_text(row.get("product_locale")).lower()
        if not is_locale_match(row_locale, locale):
            continue

        if not is_positive_label(row.get("esci_label", "")):
            continue

        query_key = make_query_key(row_locale, row.get("query_id"))
        if selected_query_key_set is not None and query_key not in selected_query_key_set:
            if seen_selected_query_keys and len(seen_selected_query_keys) == len(selected_query_key_set):
                break
            continue

        pair_key = (row_locale, str(row.get("query_id")), str(row.get("product_id")))
        if pair_key in seen_pairs:
            continue

        query_text = normalize_text(row.get("query"))
        product_text = build_product_text(
            title=row.get("product_title"),
            brand=row.get("product_brand"),
            description=row.get("product_description"),
            bullets=row.get("product_bullet_point"),
        )
        if not query_text or not product_text:
            continue

        if selected_query_key_set is not None:
            seen_selected_query_keys.add(query_key)
        seen_pairs.add(pair_key)
        queries.append(query_text)
        documents.append(product_text)
        if max_samples is not None and len(queries) >= max_samples:
            break

    if not queries:
        raise ValueError(f"No ESCI positive pairs were created for split={split!r}, locale={locale!r}.")

    LOGGER.info("Built %d positive ESCI pairs for split=%s locale=%s.", len(queries), split, locale)
    return Dataset.from_dict({"query": queries, "document": documents})


def build_ir_eval_data(
    split: str,
    locale: str = "all",
    max_queries: int | None = 1000,
) -> tuple[dict[str, str], dict[str, str], dict[str, set[str]], list[str]]:
    locale = normalize_locale(locale)
    if max_queries is not None and max_queries <= 0:
        raise ValueError("max_queries must be positive when provided.")

    selected_query_keys: list[str] = []
    selected_query_key_set: set[str] = set()

    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}

    for row in iter_esci_rows(split):
        row_locale = normalize_text(row.get("product_locale")).lower()
        if not is_locale_match(row_locale, locale):
            continue

        query_key = make_query_key(row_locale, row.get("query_id"))
        if query_key not in selected_query_key_set:
            if max_queries is not None and len(selected_query_keys) >= max_queries:
                break
            selected_query_keys.append(query_key)
            selected_query_key_set.add(query_key)

        query_text = normalize_text(row.get("query"))
        product_text = build_product_text(
            title=row.get("product_title"),
            brand=row.get("product_brand"),
            description=row.get("product_description"),
            bullets=row.get("product_bullet_point"),
        )
        if not query_text or not product_text:
            continue

        corpus_key = make_corpus_key(row_locale, row.get("product_id"))
        queries[query_key] = query_text
        corpus[corpus_key] = product_text

        if is_positive_label(row.get("esci_label", "")):
            relevant_docs.setdefault(query_key, set()).add(corpus_key)

    filtered_query_keys = [query_key for query_key in selected_query_keys if query_key in relevant_docs]
    filtered_queries = {query_key: queries[query_key] for query_key in filtered_query_keys if query_key in queries}
    filtered_relevant_docs = {
        query_key: relevant_docs[query_key]
        for query_key in filtered_query_keys
        if relevant_docs.get(query_key)
    }

    if not filtered_queries:
        raise ValueError(f"No ESCI evaluation queries with positives were found for split={split!r}, locale={locale!r}.")

    LOGGER.info(
        "Built ESCI evaluator data: %d queries, %d corpus docs, %d relevant pairs.",
        len(filtered_queries),
        len(corpus),
        sum(len(doc_ids) for doc_ids in filtered_relevant_docs.values()),
    )
    return filtered_queries, corpus, filtered_relevant_docs, filtered_query_keys
