import argparse
import logging
import os
import sys
from collections import defaultdict

import torch
from datasets import Dataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

if __package__:
    from .mps_cached_multiple_negatives_ranking_loss import MPSCachedMultipleNegativesRankingLoss
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from mps_cached_multiple_negatives_ranking_loss import MPSCachedMultipleNegativesRankingLoss

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "whooray/Ko-StrategyQA"
DEFAULT_MODEL_NAME = "google/embeddinggemma-300m"
DEFAULT_RUN_NAME = "embeddinggemma-300m-ko-strategyqa"


def configure_logging() -> None:
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def load_config_split(dataset_name: str, config_name: str, split_candidates: tuple[str, ...]):
    last_error = None
    for split_name in split_candidates:
        try:
            return load_dataset(dataset_name, config_name, split=split_name)
        except Exception as error:  # noqa: BLE001
            last_error = error
    raise RuntimeError(
        f"Could not load {dataset_name!r} config {config_name!r} with any split in {split_candidates}. "
        f"Last error: {last_error}"
    )


def build_query_lookup(queries_dataset):
    return {row["_id"]: row["text"] for row in queries_dataset}


def build_corpus_lookup(corpus_dataset):
    corpus_lookup = {}
    for row in corpus_dataset:
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        passage_text = f"{title}\n{text}" if title else text
        corpus_lookup[row["_id"]] = passage_text
    return corpus_lookup


def build_pair_dataset(qrels_dataset, query_lookup, corpus_lookup):
    questions = []
    passages = []
    skipped = 0

    for row in qrels_dataset:
        if row.get("score", 0) <= 0:
            continue
        query_text = query_lookup.get(row["query-id"])
        passage_text = corpus_lookup.get(row["corpus-id"])
        if query_text is None or passage_text is None:
            skipped += 1
            continue
        questions.append(query_text)
        passages.append(passage_text)

    if skipped:
        logging.warning("Skipped %d qrels rows due to missing query/corpus IDs.", skipped)
    if not questions:
        raise ValueError("No positive training pairs were created from qrels.")

    return Dataset.from_dict({"question": questions, "passage_text": passages})


def build_ir_eval_data(qrels_dataset, query_lookup, corpus_lookup):
    queries = {}
    relevant_docs = defaultdict(set)

    for row in qrels_dataset:
        if row.get("score", 0) <= 0:
            continue
        query_id = row["query-id"]
        corpus_id = row["corpus-id"]
        query_text = query_lookup.get(query_id)
        if query_text is None or corpus_id not in corpus_lookup:
            continue
        queries[query_id] = query_text
        relevant_docs[query_id].add(corpus_id)

    if not queries:
        raise ValueError("No dev queries were created from qrels.")

    relevant_docs = {query_id: sorted(doc_ids) for query_id, doc_ids in relevant_docs.items()}
    return queries, corpus_lookup, relevant_docs


def using_cuda() -> bool:
    return torch.cuda.is_available()


def using_mps() -> bool:
    return torch.backends.mps.is_available()


def create_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(
        model_name,
        model_card_data=SentenceTransformerModelCardData(
            language="ko",
            license="apache-2.0",
            model_name="EmbeddingGemma-300m trained on Ko-StrategyQA",
        ),
    )


def build_prompts(model: SentenceTransformer) -> dict[str, str] | None:
    model_prompts = getattr(model, "prompts", None)
    if not model_prompts:
        return None
    query_prompt = model_prompts.get("query")
    document_prompt = model_prompts.get("document")
    if not query_prompt or not document_prompt:
        return None
    return {
        "question": query_prompt,
        "passage_text": document_prompt,
    }


def add_common_training_args(
    training_args: dict[str, object],
    report_to: str,
    run_name: str,
) -> dict[str, object]:
    training_args["report_to"] = report_to
    training_args["run_name"] = run_name
    if report_to != "none":
        training_args["project"] = "nanoembeddings"
        training_args["trackio_space_id"] = None
    return training_args


def run(args: argparse.Namespace) -> str:
    configure_logging()

    dataset_name = args.dataset_name
    model = create_model(args.model_name)

    train_qrels = load_dataset(dataset_name, split="train")
    dev_qrels = load_dataset(dataset_name, split="dev")
    queries_dataset = load_config_split(dataset_name, "queries", ("ko", "queries", "train"))
    corpus_dataset = load_config_split(dataset_name, "corpus", ("ko", "corpus", "train"))

    query_lookup = build_query_lookup(queries_dataset)
    corpus_lookup = build_corpus_lookup(corpus_dataset)

    train_dataset = build_pair_dataset(train_qrels, query_lookup, corpus_lookup)
    eval_dataset = build_pair_dataset(dev_qrels, query_lookup, corpus_lookup)
    LOGGER.info(
        "Loaded Ko-StrategyQA: %d train pairs, %d dev pairs, %d corpus docs.",
        len(train_dataset),
        len(eval_dataset),
        len(corpus_lookup),
    )

    if using_cuda():
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)
        train_batch_size = 128
        eval_batch_size = 128
    elif using_mps():
        loss = MPSCachedMultipleNegativesRankingLoss(model, mini_batch_size=8)
        train_batch_size = 32
        eval_batch_size = 32
    else:
        loss = MultipleNegativesRankingLoss(model)
        train_batch_size = 32
        eval_batch_size = 32
        LOGGER.warning("Neither CUDA nor MPS is available. Falling back to MultipleNegativesRankingLoss on CPU.")

    use_cuda = using_cuda()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and not use_bf16

    run_name = args.run_name or DEFAULT_RUN_NAME
    output_dir = args.output_dir or f"models/{run_name}"
    training_args = {
        "output_dir": output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.1,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "batch_sampler": BatchSamplers.NO_DUPLICATES,
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_strategy": "steps",
        "save_steps": 100,
        "save_total_limit": 2,
        "logging_steps": 5,
    }
    prompts = build_prompts(model)
    if prompts is not None:
        training_args["prompts"] = prompts
    add_common_training_args(training_args, args.report_to, run_name)
    trainer_args = SentenceTransformerTrainingArguments(**training_args)

    queries, corpus, relevant_docs = build_ir_eval_data(dev_qrels, query_lookup, corpus_lookup)
    dev_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="ko-strategyqa-dev",
        show_progress_bar=True,
    )
    dev_evaluator(model)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    trainer.train()
    dev_evaluator(model)

    final_output_dir = f"{output_dir}/final"
    model.save_pretrained(final_output_dir)
    return final_output_dir


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Retrieval dataset on the Hugging Face Hub.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Base dense model checkpoint.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Training run name.")
    parser.add_argument("--output-dir", default=None, help="Override the checkpoint output directory.")
    parser.add_argument("--num-train-epochs", type=float, default=1, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate.")
    parser.add_argument(
        "--report-to",
        default="trackio",
        help="Tracking integration name. Use 'none' to disable external logging.",
    )
    return parser


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("dense-finetune", help="Run the existing Ko-StrategyQA dense training recipe.")
    add_arguments(parser)
    parser.set_defaults(func=run)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the dense Ko-StrategyQA finetuning recipe.")
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> str:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
