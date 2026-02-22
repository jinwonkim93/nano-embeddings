# XPQA Local CSV -> Hub Format

This folder stores a local XPQA-to-retrieval conversion workflow.
The script reads local CSV files and generates Ko-StrategyQA-compatible outputs.

## Required local files

Place these files in `datasets/xpqa` (or point `--data-dir` elsewhere):

- `train.csv`
- `dev.csv`
- `test.csv`

Expected columns include:

- `qid`, `qa_id`, `lang`, `label`, `question`, `context`, `title`

Only rows with `label == "2"` are treated as positive query-document pairs.

## Output format

After conversion, files are written as JSONL:

- `qrels/<language>-qrels/train.jsonl`
- `qrels/<language>-qrels/dev.jsonl`
- `qrels/<language>-qrels/test.jsonl`
- `queries/<language>-query.jsonl`
- `corpus/<language>-corpus.jsonl`

Schema:

- qrels rows: `{"query-id": str, "corpus-id": str, "score": int}`
- queries rows: `{"_id": str, "text": str}`
- corpus rows: `{"_id": str, "title": str, "text": str}`

## Build local outputs

```bash
python datasets/xpqa_ko_strategyqa/prepare_xpqa.py
```

Optional:

- `--data-dir` (default: `datasets/xpqa`)
- `--output-dir` (default: `datasets/xpqa`)
- `--default-language` (set default Hub config language)

## Upload to Hub

```bash
python datasets/xpqa/prepare_xpqa.py \
  --repo-id whooray/ko-strategyqa-xpqa \
  --default-language en
```

Optional upload flags:

- `--private`
- `--token <hf_token>`
- `--create-pr`
- `--upload-max-attempts 6`
- `--upload-backoff-seconds 5`

Config layout on Hub:

- `<language>-qrels`: `train/dev/test`
- `<language>-query`: `train`
- `<language>-corpus`: `train`
