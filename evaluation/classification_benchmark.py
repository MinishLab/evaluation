import json
import time
from logging import getLogger
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset, load_dataset
from matplotlib.colors import LogNorm
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
    # Set a more readable font
    plt.rcParams["font.family"] = "Verdana"  # Change to 'DejaVu Sans' if preferred
    plt.rcParams["font.weight"] = "normal"  # Set all text to normal weight by default

    # Apply the correct Seaborn style
    plt.style.use("seaborn-v0_8-darkgrid")  # Using the available Seaborn v0_8 style

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
                "model": model_name,
                "avg_score": avg_score,
                "samples_second": samples_second,
                "params": params,  # Use the params from the file
            }
        )

    # Create DataFrame for scores
    df = pd.DataFrame(data)

    # Generate enhanced scatterplot for sentences per second vs average score
    avg_df = pd.DataFrame(model_averages)
    fig, ax = plt.subplots(figsize=(10, 7))

    # Create a colormap with more fine-grained differences
    cmap = mcolors.LinearSegmentedColormap.from_list("bubble_color", ["#888888", "#a8d08d", "green"])  # Grey to green

    # Use a log scale normalization for the x-values (sentences per second) for smoother transitions
    norm = LogNorm(vmin=min(avg_df["samples_second"]), vmax=max(avg_df["samples_second"]))

    # Increase bubble size scaling factor
    size_factor = 5000  # Increased this to make the bubbles larger
    sizes = avg_df["params"] / max(avg_df["params"]) * size_factor  # Larger scaling factor

    # Plot scatter with bubble colors based on the x-values (sentences per second) using log scale
    scatter = ax.scatter(
        avg_df["samples_second"],
        avg_df["avg_score"],
        s=sizes,  # Use model parameter sizes for bubble size
        c=avg_df["samples_second"],  # Color based on the "sentences per second" value
        cmap=cmap,  # Use an expanded colormap for finer differences
        alpha=0.7,
        edgecolors="w",
        norm=norm,  # Logarithmic normalization for smoother color scaling
    )

    # Place the model names in the center of the bubbles (bold)
    for i, model in enumerate(avg_df["model"]):
        ax.text(
            avg_df["samples_second"][i],
            avg_df["avg_score"][i],
            model,
            fontsize=10,  # Font size
            fontweight="bold",  # Bold text for model names
            ha="center",  # Center horizontally
            va="center",  # Center vertically
        )

    # Set the axis limits as tuples (fixing the mypy issue)
    ax.set_xlim(
        (
            min(avg_df["samples_second"]) - 0.05 * max(avg_df["samples_second"]),
            max(avg_df["samples_second"]) + 0.05 * max(avg_df["samples_second"]),
        )
    )
    ax.set_ylim((min(avg_df["avg_score"]) - 0.05, max(avg_df["avg_score"]) + 0.05))

    # Set the axis labels and title (normal weight)
    ax.set_xlabel("Sentences per second", fontsize=12, fontweight="normal")
    ax.set_ylabel("Average Score", fontsize=12, fontweight="normal")
    ax.set_title(
        "Model Performance: Sentences per second vs Average Score (bubble size = parameters)",
        fontsize=14,
        fontweight="normal",
    )
    ax.grid(True)

    plt.show()

    return df
