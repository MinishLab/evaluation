from typing import Any

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

STEP_OUTPUT = Tensor | dict[str, Any]
EPOCH_OUTPUT = list[STEP_OUTPUT]


class ParaphraseDataset(Dataset):
    """Dataset for paraphrase probing task."""

    def __init__(self, X: torch.Tensor, label_tensor: torch.Tensor) -> None:
        """
        Initialize the dataset.

        :param X: The input data.
        :param label_tensor: The labels.
        """
        self.concat_input = X.float()
        self.label = label_tensor.float()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the item at the given index."""
        return self.concat_input[index], self.label[index]

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.concat_input)


class ProbingModel(LightningModule):
    """Probing model for paraphrase detection."""

    def __init__(self, input_dim: int, train_dataset: Dataset, valid_dataset: Dataset, test_dataset: Dataset) -> None:
        """
        Initialize the probing model.

        :param input_dim: The input dimension.
        :param train_dataset: The training dataset.
        :param valid_dataset: The validation dataset.
        :param test_dataset: The test dataset.
        """
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(self.input_dim, 256)
        self.linear2 = nn.Linear(256, 1)
        self.output = nn.Sigmoid()

        # Hyper-parameters, that we will auto-tune using lightning.
        self.lr = 0.0001
        self.batch_size = 200

        # datasets
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        # Store validation and test outputs
        self.validation_outputs = []
        self.test_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x1 = F.relu(self.linear(x))
        x2 = self.linear2(x1)
        output: torch.Tensor = self.output(x2)
        return output.reshape((-1,))

    def configure_optimizers(self) -> optim.Adam:
        """Configure the optimizer."""
        return optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader."""
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Get the test dataloader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def compute_accuracy(self, y_hat: torch.Tensor, y: torch.Tensor) -> float:
        """Compute the accuracy of the model."""
        y_pred = (y_hat >= 0.5).long()
        num_correct = (y_pred == y).long().sum().item()
        accuracy = num_correct / len(y_hat)
        return accuracy

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> dict[str, Any]:
        """Training step of the model."""
        mode = "train"
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)
        return {f"loss": loss, f"{mode}_accuracy": accuracy}

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> dict[str, Any]:
        """Validation step of the model."""
        mode = "val"
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)
        self.log(f"{mode}_loss", loss, on_epoch=True, on_step=False)
        self.log(f"{mode}_accuracy", accuracy, on_epoch=True, on_step=False)

        # Store the outputs for aggregation later
        self.validation_outputs.append({"val_loss": loss, "val_accuracy": accuracy})
        return {"val_loss": loss, "val_accuracy": accuracy}

    def on_validation_epoch_end(self) -> None:
        """Validation epoch end hook."""
        mode = "val"
        loss_mean = torch.stack([x["val_loss"] for x in self.validation_outputs]).mean()
        accuracy_mean = torch.tensor([x["val_accuracy"] for x in self.validation_outputs]).mean()
        self.log(f"epoch_{mode}_loss", loss_mean, on_epoch=True, on_step=False)
        self.log(f"epoch_{mode}_accuracy", accuracy_mean, on_epoch=True, on_step=False)

        # Clear the outputs for the next epoch
        self.validation_outputs.clear()

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_nb: int) -> dict[str, Any]:
        """Test step of the model."""
        mode = "test"
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        accuracy = self.compute_accuracy(y_hat, y)
        self.log(f"{mode}_loss", loss, on_epoch=True, on_step=False)
        self.log(f"{mode}_accuracy", accuracy, on_epoch=True, on_step=False)

        # Store the outputs for aggregation later
        self.test_outputs.append({"test_loss": loss, "test_accuracy": accuracy})
        return {"test_loss": loss, "test_accuracy": accuracy}

    def on_test_epoch_end(self) -> None:
        """Test epoch end hook."""
        mode = "test"
        loss_mean = torch.stack([x["test_loss"] for x in self.test_outputs]).mean()
        accuracy_mean = torch.tensor([x["test_accuracy"] for x in self.test_outputs]).mean()
        self.log(f"epoch_{mode}_loss", loss_mean, on_epoch=True, on_step=False)
        self.log(f"epoch_{mode}_accuracy", accuracy_mean, on_epoch=True, on_step=False)

        # Clear the outputs for the next epoch
        self.test_outputs.clear()


def run_probing_model(X: np.ndarray, y: list[int]) -> float:
    """
    Run the probing model.

    :param X: The input data.
    :param y: The labels.
    :return: The test accuracy.
    """
    X_train, X_to_split, y_train, y_to_split = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_dev, y_test, y_dev = train_test_split(X_to_split, y_to_split, test_size=0.5, random_state=42)

    train_dataset = ParaphraseDataset(torch.from_numpy(X_train), torch.LongTensor(y_train))
    valid_dataset = ParaphraseDataset(torch.from_numpy(X_dev), torch.LongTensor(y_dev))
    test_dataset = ParaphraseDataset(torch.from_numpy(X_test), torch.LongTensor(y_test))

    model = ProbingModel(
        input_dim=X.shape[1],
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        test_dataset=test_dataset,
    )

    early_stop_callback = EarlyStopping(
        monitor="epoch_val_accuracy", min_delta=0.00, patience=5, verbose=False, mode="max"
    )

    trainer = Trainer(max_epochs=100, min_epochs=3, callbacks=[early_stop_callback])
    trainer.fit(model)
    result = trainer.test(dataloaders=model.test_dataloader())

    return result[0]["epoch_test_accuracy"]
