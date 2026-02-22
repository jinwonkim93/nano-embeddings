import csv
from typing import List

import datasets

LANGUAGES = ["ar", "de", "es", "fr", "hi", "it", "ja", "ko", "pl", "pt", "ta", "zh"]
DATA_PATH = "test.csv"


class XPQAConfig(datasets.BuilderConfig):
    def __init__(self, language, **kwargs):
        super().__init__(**kwargs)
        self.language = language


class XPQA(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = XPQAConfig

    BUILDER_CONFIGS = [
        XPQAConfig(name=language, language=language) for language in LANGUAGES
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description="xPQA is a large-scale annotated cross-lingual Product QA dataset.",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answer": datasets.Value("string"),
                }
            ),
            homepage="https://github.com/amazon-science/contextual-product-qa/tree/main?tab=readme-ov-file#xpqa",
            citation="https://arxiv.org/abs/2305.09249",
        )

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        downloaded_file = dl_manager.download_and_extract(DATA_PATH)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": downloaded_file}
            ),
        ]

    def _generate_examples(self, filepath):
        id_ = 0
        with open(filepath, newline="") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=",")
            header = next(csvreader)
            lang_pos = header.index("lang")
            answer_pos = header.index("answer")
            question_pos = header.index("question")
            label_pos = header.index("label")
            for row in csvreader:
                if row[lang_pos] == self.config.language and row[label_pos] == "2":
                    answer = row[answer_pos]
                    question = row[question_pos]
                    if not answer or not question:
                        continue
                    yield id_, {"id": id_, "question": question, "answer": answer}
                    id_ += 1