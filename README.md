# nanoembeddings

Small training repo for dense and sparse embedding recipes.

## Commands

Dense Ko-StrategyQA finetuning:

```bash
uv run python main.py dense-finetune
```

Sparse ESCI SPLADE finetuning:

```bash
uv run python main.py sparse-esci-finetune
```

If you prefer the project entrypoint, the same commands also work as:

```bash
uv run nanoembeddings dense-finetune
uv run nanoembeddings sparse-esci-finetune
```

## Sparse ESCI Recipe

The sparse path follows the Standard SPLADE recipe from the Hugging Face sparse encoder tooling and the Qdrant ESCI article:

- dataset: `tasksource/esci`
- positives: `esci_label in {"Exact", "Substitute"}`
- locales: `us`, `es`, `jp`, or `all`
- default base model: `bert-base-multilingual-cased`
- model: `MLMTransformer + SpladePooling("max")`
- loss: `SpladeLoss(SparseMultipleNegativesRankingLoss)`

The ESCI loader uses the raw dataset schema directly:

- `esci_label`
- `product_locale`
- `product_bullet_point`

Product text is normalized into:

```text
[brand] title | description | bullet1 | bullet2 | bullet3
```

Empty values such as `None`, empty strings, and `"None"` are removed before formatting. The final product text is truncated to 512 characters to match the recipe defaults.

## Useful Sparse Flags

```bash
uv run python main.py sparse-esci-finetune \
  --locale all \
  --max-samples 100000 \
  --eval-max-queries 1000 \
  --report-to none
```

Quick smoke test:

```bash
uv run python main.py sparse-esci-finetune \
  --max-samples 32 \
  --eval-max-queries 20 \
  --dry-run
```

Tiny one-step training check:

```bash
uv run python main.py sparse-esci-finetune \
  --max-samples 64 \
  --eval-max-queries 20 \
  --max-steps 1
```

## MPS Notes

- CUDA enables fp16 or bf16 automatically.
- MPS and CPU disable mixed precision by default.
- MPS and CPU also use smaller default batch sizes.
- `--chunk-size` can reduce SPLADE memory usage on MPS if the MLM logits become too large.

## Existing Dense Recipe

The dense recipe remains the Ko-StrategyQA path built around `SentenceTransformerTrainer` and the MPS-safe cached multiple negatives ranking loss implementation in [`dense_embeddings/mps_cached_multiple_negatives_ranking_loss.py`](/Users/jw93.dev/Desktop/workspace/nanoembeddings/dense_embeddings/mps_cached_multiple_negatives_ranking_loss.py).
