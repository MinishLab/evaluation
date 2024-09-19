import json
import time
from logging import getLogger
from pathlib import Path

from datasets import Dataset, load_dataset
from mteb.encoder_interface import Encoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = getLogger(__name__)

datasets = [
    {"ds_name": "sst2", "text_name": "sentence", "label_name": "label", "type": "classification"},
    {"ds_name": "imdb", "text_name": "text", "label_name": "label", "type": "classification"},
    {"ds_name": "trec", "text_name": "text", "label_name": "coarse_label", "type": "classification"},
    {"ds_name": "yelp_polarity", "text_name": "text", "label_name": "label", "type": "classification"},
    {"ds_name": "ag_news", "text_name": "text", "label_name": "label", "type": "classification"},
]


class ClassificationBenchmark:
    def __init__(self, encoder: Encoder, save_path: str) -> None:
        """
        Initialize the classification benchmark.

        :param encoder: The encoder to use. Should be an implementation of an MTEB Encoder protocol.
        :param save_path: The path to save the results to.
        """
        self.encoder = encoder
        model_name = getattr(encoder.mteb_model_meta, "name", "no_model_name_available")
        if model_name == "no_model_name_available":
            logger.warning(
                "Encoder does not have a model name or mteb_model_meta attribute. Defaulting model name to 'no_model_name_available'."
            )

        self.model_name = model_name
        self.save_path = Path(save_path) / f"{model_name}_classification_results.json"
        self.results: dict[str, dict] = {self.model_name: {}}

    def train_test_classification(
        self, encoder: Encoder, dataset: Dataset, text_name: str, label_name: str
    ) -> tuple[list[str], list[str]]:
        """
        Train and test a classification model for a specific encoder.

        :param encoder: The encoder to use.
        :param dataset: The dataset to use.
        :param text_name: The name of the text column in the dataset.
        :param label_name: The name of the label column in the dataset.
        :return: The predictions and labels.
        """
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        X_train = encoder.encode(split["train"][text_name])
        y_train = split["train"][label_name]

        X_dev = encoder.encode(split["test"][text_name])
        y_dev = split["test"][label_name]

        model.fit(X_train, y_train)
        pred = model.predict(X_dev)

        return pred, y_dev

    def run(self) -> None:
        """Run the classification benchmark."""
        for dataset_config in datasets:
            ds_name = dataset_config["ds_name"]
            dataset = load_dataset(ds_name)
            task_type = dataset_config["type"]

            logger.info(f"Evaluating {ds_name} for task type: {task_type}")
            text_name = dataset_config["text_name"]
            label_name = dataset_config["label_name"]

            start_time = time.time()

            pred, gold = self.train_test_classification(self.encoder, dataset, text_name, label_name)
            korok_metrics = precision_recall_fscore_support(gold, pred, average="micro")
            runtime = time.time() - start_time

            self.results[self.model_name][ds_name] = {
                "dataset": ds_name,
                "main_score": korok_metrics[2],  # Main score
                "runtime": runtime,
            }

            # Save the results to a JSON file
            self.save_results(self.save_path)

    def save_results(self, save_path: Path) -> None:
        """Save the results to a JSON file."""
        with open(save_path, "w") as file:
            json.dump(self.results, file, indent=4)
