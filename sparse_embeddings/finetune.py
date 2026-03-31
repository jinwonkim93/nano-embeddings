import argparse
import logging
import os
import re
import sys
from types import SimpleNamespace

import torch
from sentence_transformers import (
    SparseEncoder,
    SparseEncoderModelCardData,
    SparseEncoderTrainer,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.sparse_encoder.evaluation import SparseInformationRetrievalEvaluator
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss
from sentence_transformers.sparse_encoder.models import MLMTransformer, SpladePooling
from sentence_transformers.training_args import BatchSamplers

if __package__:
    from .esci import SUPPORTED_LOCALES, build_ir_eval_data, build_pair_dataset
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from esci import SUPPORTED_LOCALES, build_ir_eval_data, build_pair_dataset

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "bert-base-multilingual-cased"
DEFAULT_LOCALE = "all"
DEFAULT_MAX_SAMPLES = 100_000
DEFAULT_EVAL_MAX_QUERIES = 1_000


def configure_logging() -> None:
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)


def using_cuda() -> bool:
    return torch.cuda.is_available()


def using_mps() -> bool:
    return torch.backends.mps.is_available()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def resolve_device_settings(
    requested_train_batch_size: int | None,
    requested_chunk_size: int | None,
) -> SimpleNamespace:
    if using_cuda():
        bf16 = torch.cuda.is_bf16_supported()
        return SimpleNamespace(
            train_batch_size=requested_train_batch_size or 32,
            eval_batch_size=requested_train_batch_size or 32,
            chunk_size=requested_chunk_size,
            fp16=not bf16,
            bf16=bf16,
            device_label="cuda",
        )
    if using_mps():
        train_batch_size = requested_train_batch_size or 4
        return SimpleNamespace(
            train_batch_size=train_batch_size,
            eval_batch_size=min(train_batch_size, 4),
            chunk_size=requested_chunk_size if requested_chunk_size is not None else 64,
            fp16=False,
            bf16=False,
            device_label="mps",
        )
    train_batch_size = requested_train_batch_size or 4
    return SimpleNamespace(
        train_batch_size=train_batch_size,
        eval_batch_size=min(train_batch_size, 4),
        chunk_size=requested_chunk_size if requested_chunk_size is not None else 64,
        fp16=False,
        bf16=False,
        device_label="cpu",
    )


def build_run_name(base_model: str, locale: str) -> str:
    return f"splade-esci-{locale}-{slugify(base_model)}"


def create_model(base_model: str, locale: str, chunk_size: int | None) -> SparseEncoder:
    mlm_transformer = MLMTransformer(base_model, max_seq_length=512)
    pooling = SpladePooling(
        pooling_strategy="max",
        word_embedding_dimension=mlm_transformer.get_sentence_embedding_dimension(),
        chunk_size=chunk_size,
    )
    return SparseEncoder(
        modules=[mlm_transformer, pooling],
        model_card_data=SparseEncoderModelCardData(
            language=["en", "es", "ja"],
            license="apache-2.0",
            model_name=f"SPLADE {base_model} trained on ESCI ({locale})",
            train_datasets=[{"name": "tasksource/esci", "split": "train"}],
            eval_datasets=[{"name": "tasksource/esci", "split": "test"}],
        ),
        similarity_fn_name="dot",
    )


def create_loss(
    model: SparseEncoder,
    query_regularizer_weight: float,
    document_regularizer_weight: float,
) -> SpladeLoss:
    return SpladeLoss(
        model=model,
        loss=SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=query_regularizer_weight,
        document_regularizer_weight=document_regularizer_weight,
    )


def build_training_args(args: argparse.Namespace, device_settings: SimpleNamespace, run_name: str):
    output_dir = f"models/{run_name}"
    return SparseEncoderTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=device_settings.train_batch_size,
        per_device_eval_batch_size=device_settings.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        fp16=device_settings.fp16,
        bf16=device_settings.bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=1_000,
        save_strategy="steps",
        save_steps=1_000,
        save_total_limit=2,
        logging_steps=100,
        report_to=args.report_to,
        project="nanoembeddings",
        trackio_space_id=None,
        run_name=run_name,
    )


def run(args: argparse.Namespace) -> str:
    configure_logging()
    device_settings = resolve_device_settings(args.per_device_train_batch_size, args.chunk_size)
    LOGGER.info(
        "Running sparse ESCI recipe on %s with train_batch_size=%d eval_batch_size=%d chunk_size=%s.",
        device_settings.device_label,
        device_settings.train_batch_size,
        device_settings.eval_batch_size,
        device_settings.chunk_size,
    )

    train_dataset = build_pair_dataset(
        split="train",
        locale=args.locale,
        max_samples=args.max_samples,
    )
    queries, corpus, relevant_docs, eval_query_keys = build_ir_eval_data(
        split="test",
        locale=args.locale,
        max_queries=args.eval_max_queries,
    )
    eval_dataset = build_pair_dataset(
        split="test",
        locale=args.locale,
        selected_query_keys=set(eval_query_keys),
    )

    model = create_model(args.base_model, args.locale, device_settings.chunk_size)
    loss = create_loss(model, args.query_regularizer_weight, args.document_regularizer_weight)
    run_name = build_run_name(args.base_model, args.locale)
    trainer_args = build_training_args(args, device_settings, run_name)
    evaluator = SparseInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name=f"esci-{args.locale}-test",
        show_progress_bar=not args.dry_run,
        batch_size=device_settings.eval_batch_size,
    )

    LOGGER.info(
        "Prepared sparse ESCI recipe with %d train pairs, %d eval pairs, %d eval queries, %d eval docs.",
        len(train_dataset),
        len(eval_dataset),
        len(queries),
        len(corpus),
    )
    if args.dry_run:
        LOGGER.info("Dry run complete; skipping evaluator execution and training.")
        return trainer_args.output_dir

    evaluator(model)
    trainer = SparseEncoderTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    evaluator(model)

    final_output_dir = f"{trainer_args.output_dir}/final"
    model.save_pretrained(final_output_dir)
    LOGGER.info("Saved sparse model to %s.", final_output_dir)
    return final_output_dir


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base MLM checkpoint used for SPLADE.")
    parser.add_argument(
        "--locale",
        default=DEFAULT_LOCALE,
        choices=SUPPORTED_LOCALES,
        help="ESCI locale subset to train on.",
    )
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES, help="Maximum positive training pairs.")
    parser.add_argument(
        "--eval-max-queries",
        type=int,
        default=DEFAULT_EVAL_MAX_QUERIES,
        help="Maximum unique ESCI test queries used for IR evaluation.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1, help="Number of training epochs.")
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=None,
        help="Override the per-device train batch size.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Optimizer learning rate.")
    parser.add_argument(
        "--query-regularizer-weight",
        type=float,
        default=5e-5,
        help="SPLADE query regularizer weight.",
    )
    parser.add_argument(
        "--document-regularizer-weight",
        type=float,
        default=3e-5,
        help="SPLADE document regularizer weight.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Optional SpladePooling chunk size for lower-memory devices.",
    )
    parser.add_argument("--max-steps", type=int, default=-1, help="Override the trainer max_steps value.")
    parser.add_argument(
        "--report-to",
        default="none",
        help="Tracking integration name. Use 'none' to disable external logging.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Initialize everything without evaluating or training.")
    return parser


def add_parser(subparsers) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("sparse-esci-finetune", help="Run the ESCI SPLADE training recipe.")
    add_arguments(parser)
    parser.set_defaults(func=run)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the sparse ESCI SPLADE finetuning recipe.")
    add_arguments(parser)
    return parser


def main(argv: list[str] | None = None) -> str:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    main()
