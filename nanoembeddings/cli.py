import argparse

from dense_embeddings.finetune import add_parser as add_dense_finetune_parser
from sparse_embeddings.finetune import add_parser as add_sparse_esci_finetune_parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanoembeddings",
        description="Training entrypoints for dense and sparse embedding recipes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_dense_finetune_parser(subparsers)
    add_sparse_esci_finetune_parser(subparsers)
    return parser


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
