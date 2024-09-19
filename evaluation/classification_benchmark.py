import json
import time
from logging import getLogger
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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
        # First check if the encoder has the 'mteb_model_meta' attribute, and if it does, check for 'name'
        if hasattr(encoder, "mteb_model_meta") and hasattr(encoder.mteb_model_meta, "name"):
            model_name = encoder.mteb_model_meta.name
        else:
            model_name = "no_model_name_available"
            logger.warning(
                "Encoder does not have a model name or mteb_model_meta attribute. Defaulting model name to 'no_model_name_available'."
            )

        self.model_name = model_name
        self.save_path = Path(save_path) / f"{model_name}_classification_results.json"
        # Make sure the save directory exists
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
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
        X_train = encoder.encode(split["train"][text_name], show_progress_bar=True)
        y_train = split["train"][label_name]

        X_dev = encoder.encode(split["test"][text_name], show_progress_bar=True)
        y_dev = split["test"][label_name]

        model.fit(X_train, y_train)
        pred = model.predict(X_dev)

        return pred, y_dev

    def run(self) -> None:
        """Run the classification benchmark."""
        for dataset_config in datasets:
            ds_name = dataset_config["ds_name"]
            dataset = load_dataset(ds_name)

            logger.info(f"Evaluating {ds_name}")
            text_name = dataset_config["text_name"]
            label_name = dataset_config["label_name"]

            start_time = time.time()

            pred, gold = self.train_test_classification(self.encoder, dataset, text_name, label_name)
            metrics = precision_recall_fscore_support(gold, pred, average="micro")
            runtime = time.time() - start_time

            self.results[self.model_name][ds_name] = {
                "dataset": ds_name,
                "main_score": metrics[2],  # Main score
                "runtime": runtime,
            }

            # Save the results to a JSON file
            self.save_results(self.save_path)

    def save_results(self, save_path: Path) -> None:
        """Save the results to a JSON file."""
        with open(save_path, "w") as file:
            json.dump(self.results, file, indent=4)


def summarize_classification_results(results_path: str) -> pd.DataFrame:
    """
    Summarize the results by generating a pandas DataFrame and a scatterplot.

    :param results_path: Path to the directory containing the results JSON files.
    :return: A pandas DataFrame containing the results.
    """
    result_files = Path(results_path).glob("*.json")

    data = []
    model_averages = []

    # Process each file and extract the model name, dataset scores, and runtimes
    for file in result_files:
        with open(file, "r") as f:
            result_data = json.load(f)

        model_name = list(result_data.keys())[0]  # Extract model name
        model_info = result_data[model_name]

        row = {"model": model_name}
        total_score = 0
        total_time = 0
        dataset_count = 0

        # Extract dataset scores and runtimes
        for dataset_name, metrics in model_info.items():
            row[dataset_name] = metrics["main_score"]
            total_score += metrics["main_score"]
            total_time += metrics["runtime"]
            dataset_count += 1

        # Append data for the DataFrame
        data.append(row)

        # Calculate averages for scatterplot
        avg_score = total_score / dataset_count
        avg_time = total_time / dataset_count
        model_averages.append({"model": model_name, "avg_score": avg_score, "avg_time": avg_time})

    # Create DataFrame for scores
    df = pd.DataFrame(data)

    # Generate scatterplot for average score vs average time
    avg_df = pd.DataFrame(model_averages)
    plt.figure(figsize=(8, 6))
    plt.scatter(avg_df["avg_score"], avg_df["avg_time"], color="b", alpha=0.6)

    for i, model in enumerate(avg_df["model"]):
        plt.text(avg_df["avg_score"][i], avg_df["avg_time"][i], model)

    plt.xlabel("Average Score")
    plt.ylabel("Average Runtime (s)")
    plt.title("Model Performance: Average Score vs Average Runtime")
    plt.grid(True)
    plt.show()

    return df
