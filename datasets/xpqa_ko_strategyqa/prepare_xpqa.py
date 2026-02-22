#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from datasets import Dataset

LANGUAGE_NAME_MAP = {
    "ar": "arabic",
    "de": "german",
    "es": "spanish",
    "fr": "french",
    "hi": "hindi",
    "it": "italian",
    "ja": "japanese",
    "ko": "korean",
    "pl": "polish",
    "pt": "portuguese",
    "ta": "tamil",
    "zh": "chinese",
}
LANGUAGE_CODE_MAP = {value: key for key, value in LANGUAGE_NAME_MAP.items()}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def language_to_config_prefix(language: str) -> str:
    return LANGUAGE_NAME_MAP.get(language, language)


def normalize_language(language: str) -> str:
    token = language.strip().lower()
    if token in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[token]
    return token


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert local XPQA CSV files to Ko-StrategyQA-compatible files and optionally upload to Hub."
    )
    parser.add_argument(
        "--data-dir",
        default="datasets/xpqa_ko_strategyqa",
        help="Directory containing local train.csv/dev.csv/test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/xpqa_ko_strategyqa",
        help="Output folder for qrels/queries/corpus JSONL files.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Hugging Face dataset repo id to upload to (for example: whooray/ko-strategyqa-xpqa).",
    )
    parser.add_argument("--private", action="store_true", help="Create/upload dataset as private.")
    parser.add_argument("--token", default=None, help="HF token (optional, uses cached login if omitted).")
    parser.add_argument("--create-pr", action="store_true", help="Upload changes via PR instead of direct push.")
    parser.add_argument(
        "--upload-max-attempts",
        type=int,
        default=6,
        help="Maximum upload attempts per push operation (default: 6).",
    )
    parser.add_argument(
        "--upload-backoff-seconds",
        type=float,
        default=5.0,
        help="Base seconds for exponential retry backoff (default: 5.0).",
    )
    parser.add_argument(
        "--default-language",
        default=None,
        help="Language to use as default Hub config (default: first discovered language).",
    )
    return parser.parse_args()


def build_dataset_from_rows(rows: list[dict], empty_schema: dict[str, list]) -> Dataset:
    if rows:
        return Dataset.from_list(rows)
    return Dataset.from_dict(empty_schema)


def is_retriable_upload_error(error: Exception) -> bool:
    name = error.__class__.__name__.lower()
    text = str(error).lower()
    retriable_names = {
        "readtimeout",
        "connecttimeout",
        "connecterror",
        "remoteprotocolerror",
    }
    retriable_tokens = (
        "read timeout",
        "timed out",
        "connection reset",
        "temporarily unavailable",
        "server disconnected",
        "502",
        "503",
        "504",
    )
    return name in retriable_names or any(token in text for token in retriable_tokens)


def push_with_retry(
    dataset: Dataset,
    operation_name: str,
    max_attempts: int,
    base_backoff_seconds: float,
    **push_kwargs,
) -> None:
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            dataset.push_to_hub(**push_kwargs)
            return
        except Exception as error:  # noqa: BLE001
            last_error = error
            if attempt >= max_attempts or not is_retriable_upload_error(error):
                raise
            wait_seconds = base_backoff_seconds * (2 ** (attempt - 1))
            print(
                f"[retry {attempt}/{max_attempts}] {operation_name} failed with {error.__class__.__name__}: {error}. "
                f"Retrying in {wait_seconds:.1f}s..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Upload failed for {operation_name}") from last_error


def split_files(data_dir: Path) -> dict[str, Path]:
    paths = {
        "train": data_dir / "train.csv",
        "dev": data_dir / "dev.csv",
        "test": data_dir / "test.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required XPQA CSV files: {', '.join(missing)}")
    return paths


def discover_languages(paths: dict[str, Path]) -> list[str]:
    discovered = set()
    for path in paths.values():
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                continue
            for required in ["lang", "label"]:
                if required not in reader.fieldnames:
                    raise ValueError(f"Column {required!r} not found in {path}")
            for row in reader:
                if row.get("label") == "2":
                    lang = (row.get("lang") or "").strip().lower()
                    if lang:
                        discovered.add(lang)

    preferred_order = list(LANGUAGE_NAME_MAP.keys())
    ordered = [lang for lang in preferred_order if lang in discovered]
    ordered.extend(sorted(discovered - set(preferred_order)))
    if not ordered:
        raise ValueError("No positive XPQA rows (label == '2') were found in local CSV files.")
    return ordered


def collect_language_rows(paths: dict[str, Path], language: str) -> tuple[dict[str, list[dict]], list[dict], list[dict]]:
    qrels_rows: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    queries_by_id: dict[str, dict] = {}
    corpus_id_by_key: dict[tuple[str, str], str] = {}
    corpus_rows: list[dict] = []

    for split_name, path in paths.items():
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                continue

            required = ["qid", "qa_id", "lang", "label", "question", "context", "title"]
            missing = [column for column in required if column not in reader.fieldnames]
            if missing:
                raise ValueError(f"Missing columns in {path}: {missing}")

            for row in reader:
                if (row.get("lang") or "").strip().lower() != language:
                    continue
                if row.get("label") != "2":
                    continue

                question = (row.get("question") or "").strip()
                context = (row.get("context") or "").strip()
                title = (row.get("title") or "").strip()
                if not question or not context:
                    continue

                qid = (row.get("qid") or "").strip()
                qa_id = (row.get("qa_id") or "").strip()
                query_uid = f"{qid}-{qa_id}" if qa_id else qid
                query_id = f"{language}-{split_name}-{query_uid}"
                queries_by_id[query_id] = {"_id": query_id, "text": question}

                corpus_key = (title, context)
                corpus_id = corpus_id_by_key.get(corpus_key)
                if corpus_id is None:
                    corpus_id = f"{language}-corpus-{len(corpus_rows)}"
                    corpus_id_by_key[corpus_key] = corpus_id
                    corpus_rows.append({"_id": corpus_id, "title": title, "text": context})

                qrels_rows[split_name].append({"query-id": query_id, "corpus-id": corpus_id, "score": 1})

    queries_rows = [queries_by_id[query_id] for query_id in sorted(queries_by_id)]
    return qrels_rows, queries_rows, corpus_rows


def upload_to_hub(
    repo_id: str,
    language: str,
    qrels_rows: dict[str, list[dict]],
    queries_rows: list[dict],
    corpus_rows: list[dict],
    set_default: bool,
    private: bool,
    token: str | None,
    create_pr: bool,
    upload_max_attempts: int,
    upload_backoff_seconds: float,
) -> None:
    config_prefix = language_to_config_prefix(language)
    qrels_config = f"{config_prefix}-qrels"
    queries_config = f"{config_prefix}-query"
    corpus_config = f"{config_prefix}-corpus"

    # Upload qrels split-by-split to reduce single-commit payload size.
    for split_name in ["train", "dev", "test"]:
        qrels_dataset = build_dataset_from_rows(
            qrels_rows[split_name],
            empty_schema={"query-id": [], "corpus-id": [], "score": []},
        )
        push_with_retry(
            qrels_dataset,
            operation_name=f"{language} {split_name} qrels",
            max_attempts=upload_max_attempts,
            base_backoff_seconds=upload_backoff_seconds,
            repo_id=repo_id,
            config_name=qrels_config,
            split=split_name,
            set_default=set_default if split_name == "train" else False,
            private=private,
            token=token,
            create_pr=create_pr,
            max_shard_size="100MB",
            commit_message=f"Upload {language} qrels split={split_name} converted from local XPQA CSV",
        )

    queries_dataset = build_dataset_from_rows(
        queries_rows,
        empty_schema={"_id": [], "text": []},
    )
    push_with_retry(
        queries_dataset,
        operation_name=f"{language} queries",
        max_attempts=upload_max_attempts,
        base_backoff_seconds=upload_backoff_seconds,
        repo_id=repo_id,
        config_name=queries_config,
        split="train",
        set_default=False,
        private=private,
        token=token,
        create_pr=create_pr,
        max_shard_size="100MB",
        commit_message=f"Upload {language} queries config",
    )

    corpus_dataset = build_dataset_from_rows(
        corpus_rows,
        empty_schema={"_id": [], "title": [], "text": []},
    )
    push_with_retry(
        corpus_dataset,
        operation_name=f"{language} corpus",
        max_attempts=upload_max_attempts,
        base_backoff_seconds=upload_backoff_seconds,
        repo_id=repo_id,
        config_name=corpus_config,
        split="train",
        set_default=False,
        private=private,
        token=token,
        create_pr=create_pr,
        max_shard_size="100MB",
        commit_message=f"Upload {language} corpus config",
    )


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    paths = split_files(data_dir)
    languages = discover_languages(paths)

    default_language = normalize_language(args.default_language) if args.default_language else languages[0]
    if default_language not in languages:
        raise ValueError(f"--default-language {default_language!r} must be one of: {languages}")

    split_map = {"train": "train", "dev": "dev", "test": "test"}
    metadata = {
        "source": "local_csv",
        "data_dir": str(data_dir),
        "languages": languages,
        "default_language": default_language,
        "split_mapping": split_map,
        "per_language": {},
    }

    for language in languages:
        qrels_rows, queries_rows, corpus_rows = collect_language_rows(paths, language)
        config_prefix = language_to_config_prefix(language)

        write_jsonl(output_dir / "queries" / f"{config_prefix}-query.jsonl", queries_rows)
        write_jsonl(output_dir / "corpus" / f"{config_prefix}-corpus.jsonl", corpus_rows)
        write_jsonl(output_dir / "qrels" / f"{config_prefix}-qrels" / "train.jsonl", qrels_rows["train"])
        write_jsonl(output_dir / "qrels" / f"{config_prefix}-qrels" / "dev.jsonl", qrels_rows["dev"])
        write_jsonl(output_dir / "qrels" / f"{config_prefix}-qrels" / "test.jsonl", qrels_rows["test"])

        metadata["per_language"][language] = {
            "config_prefix": config_prefix,
            "num_queries": len(queries_rows),
            "num_corpus": len(corpus_rows),
            "num_qrels_train": len(qrels_rows["train"]),
            "num_qrels_dev": len(qrels_rows["dev"]),
            "num_qrels_test": len(qrels_rows["test"]),
        }

        if args.repo_id:
            upload_to_hub(
                repo_id=args.repo_id,
                language=language,
                qrels_rows=qrels_rows,
                queries_rows=queries_rows,
                corpus_rows=corpus_rows,
                set_default=(language == default_language),
                private=args.private,
                token=args.token,
                create_pr=args.create_pr,
                upload_max_attempts=args.upload_max_attempts,
                upload_backoff_seconds=args.upload_backoff_seconds,
            )

    if args.repo_id:
        metadata["uploaded_repo_id"] = args.repo_id

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
