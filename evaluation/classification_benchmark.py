import json
import time
from logging import getLogger
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from mteb.encoder_interface import Encoder
from plotnine import aes, geom_point, ggplot, guides, scale_size, theme, theme_classic
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
    ) -> tuple[list[str], list[str], float]:
        """
        Train and test a classification model for a specific encoder.

        :param encoder: The encoder to use.
        :param dataset: The dataset to use.
        :param text_name: The name of the text column in the dataset.
        :param label_name: The name of the label column in the dataset.
        :return: The predictions and labels.
        """
        encode_time = 0.0
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        split = dataset["train"].train_test_split(test_size=0.1, seed=42)
        s = time.time()
        X_train = encoder.encode(split["train"][text_name], show_progress_bar=True)
        encode_time += time.time() - s
        y_train = split["train"][label_name]

        s = time.time()
        X_dev = encoder.encode(split["test"][text_name], show_progress_bar=True)
        encode_time += time.time() - s
        y_dev = split["test"][label_name]

        model.fit(X_train, y_train)
        pred = model.predict(X_dev)

        return pred, y_dev, encode_time

    def run(self) -> None:
        """Run the classification benchmark."""
        for dataset_config in datasets:
            ds_name = dataset_config["ds_name"]
            dataset = load_dataset(ds_name)

            logger.info(f"Evaluating {ds_name}")
            text_name = dataset_config["text_name"]
            label_name = dataset_config["label_name"]

            start_time = time.time()

            pred, gold, encode_time = self.train_test_classification(self.encoder, dataset, text_name, label_name)
            metrics = precision_recall_fscore_support(gold, pred, average="micro")
            runtime = time.time() - start_time

            self.results[self.model_name][ds_name] = {
                "dataset": ds_name,
                "main_score": metrics[2],  # Main score
                "runtime": runtime,
                "encode_time": encode_time,
                "dataset_length": len(dataset["train"]),
                "samples_second": len(dataset["train"]) / encode_time,
            }

            # Save the results to a JSON file
            self.save_results(self.save_path)

    def save_results(self, save_path: Path) -> None:
        """Save the results to a JSON file."""
        with open(save_path, "w") as file:
            json.dump(self.results, file, indent=4)


def summarize_classification_results(results_path: str) -> pd.DataFrame:
    """
    Summarize the results by generating a pandas DataFrame and an enhanced scatterplot.

    The bubble colors transition from grey (left, slower models) to green (right, faster models)
    using logarithmic scaling for a smoother gradient and more gradual transitions.

    :param results_path: Path to the directory containing the results JSON files.
    :return: A pandas DataFrame containing the results.
    """
    result_files = Path(results_path).glob("*.json")

    data = []
    model_averages = []

    names = {"GloVe_300d": "GloVe 6B 300d"}

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
        total_samples = 0

        # Extract params and dataset scores and runtimes
        params = model_info["params"]  # Extract params from the file

        for dataset_name, metrics in model_info.items():
            if dataset_name == "params":
                continue  # Skip the params entry
            row[dataset_name] = metrics["main_score"]
            total_score += metrics["main_score"]
            total_time += metrics["encode_time"]
            total_samples += metrics["dataset_length"]
            dataset_count += 1

        # Append data for the DataFrame
        data.append(row)

        # Calculate averages for scatterplot
        avg_score = total_score / dataset_count
        samples_second = total_samples / total_time

        model_averages.append(
            {
                "Model": names.get(model_name, model_name),
                "Accuracy": avg_score,
                "Samples per second": samples_second,
                "Params (Million)": params / 1_000_000,  # Use the params from the file
            }
        )

    # Generate enhanced scatterplot for sentences per second vs average score
    avg_df = pd.DataFrame(model_averages)

    return avg_df


def plot_avg_df(df: pd.DataFrame) -> ggplot:
    """Creates a plot of the average df returned by the summarization."""
    plot = (
        ggplot(df, aes(x="Samples per second", y="Accuracy", size="Params (Million)", color="Model"))
        + geom_point()  # Plot points with variable size
        + scale_size(range=(5, 15))  # Adjust the range: min size = 5, max size = 15
        + theme(figure_size=(10, 6))  # Adjust figure size (width, height) in inches
        + theme_classic()
        + guides(None)
    )

    return plot
