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

# Set the log level to INFO to get more information
logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

DATASET_NAME = "whooray/Ko-StrategyQA"


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


# 1. Load a model to finetune with 2. (Optional) model card data
model = SentenceTransformer(
    "google/embeddinggemma-300m",
    model_card_data=SentenceTransformerModelCardData(
        language="ko",
        license="apache-2.0",
        model_name="EmbeddingGemma-300m trained on Ko-StrategyQA",
    ),
)

# 3. Load Ko-StrategyQA and create pair datasets from qrels + queries + corpus.
train_qrels = load_dataset(DATASET_NAME, split="train")
dev_qrels = load_dataset(DATASET_NAME, split="dev")
queries_dataset = load_config_split(DATASET_NAME, "queries", ("ko", "queries", "train"))
corpus_dataset = load_config_split(DATASET_NAME, "corpus", ("ko", "corpus", "train"))

query_lookup = build_query_lookup(queries_dataset)
corpus_lookup = build_corpus_lookup(corpus_dataset)

train_dataset = build_pair_dataset(train_qrels, query_lookup, corpus_lookup)
eval_dataset = build_pair_dataset(dev_qrels, query_lookup, corpus_lookup)
logging.info(
    "Loaded Ko-StrategyQA: %d train pairs, %d dev pairs, %d corpus docs.",
    len(train_dataset),
    len(eval_dataset),
    len(corpus_lookup),
)

# 4. Use CachedMNRL on CUDA/MPS. CPU falls back to MNRL.
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
    logging.warning("Neither CUDA nor MPS is available. Falling back to MultipleNegativesRankingLoss on CPU.")

use_cuda = using_cuda()
use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
use_fp16 = use_cuda and not use_bf16

# 5. (Optional) Specify training arguments
run_name = "embeddinggemma-300m-ko-strategyqa"
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=f"models/{run_name}",
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    fp16=use_fp16,
    bf16=use_bf16,
    batch_sampler=BatchSamplers.NO_DUPLICATES,  # (Cached)MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
    prompts={  # Map training column names to model prompts
        "question": model.prompts["query"],
        "passage_text": model.prompts["document"],
    },
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    logging_steps=5,
    report_to="trackio",
    project="nanoembeddings",
    trackio_space_id=None,  # Keep logs local by default; set a Space ID to publish.
    run_name=run_name,  # Used by Trackio as the run name.
)

# 6. (Optional) Create a retrieval evaluator from the Ko-StrategyQA dev qrels.
queries, corpus, relevant_docs = build_ir_eval_data(dev_qrels, query_lookup, corpus_lookup)
dev_evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name="ko-strategyqa-dev",
    show_progress_bar=True,
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

# (Optional) Evaluate the trained model on the evaluation set once more, this will also log the results
# and include them in the model card
dev_evaluator(model)

# 8. Save the trained model
final_output_dir = f"models/{run_name}/final"
model.save_pretrained(final_output_dir)

# 9. (Optional) Push it to the Hugging Face Hub
# It is recommended to run `huggingface-cli login` to log into your Hugging Face account first
# try:
#     model.push_to_hub(run_name)
# except Exception:
#     logging.error(
#         f"Error uploading model to the Hugging Face Hub:\n{traceback.format_exc()}To upload it manually, you can run "
#         f"`huggingface-cli login`, followed by loading the model using `model = SentenceTransformer({final_output_dir!r})` "
#         f"and saving it using `model.push_to_hub('{run_name}')`."
#     )
