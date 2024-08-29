import logging
from typing import Literal, cast

import numpy as np
from autofj.datasets import load_data
from datasets import Dataset
from mteb.encoder_interface import Encoder
from reach import Reach, normalize
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score

from evaluation.pearl.probing import run_probing_model

logger = logging.getLogger(__name__)


def eval_bird(model: Encoder, dataset: Dataset) -> float:
    """
    Evaluate the BIRD dataset.

    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate.
    :return: The accuracy of the model on the dataset.
    """
    input1 = normalize(model.encode(dataset["term1"]))
    input2 = normalize(model.encode(dataset["term2"]))

    sim = (input1 * input2).sum(1)
    sim = (sim + 1) / 2.0
    cor, _ = pearsonr(sim, dataset["relatedness score"])

    return cor


def eval_turney(model: Encoder, dataset: Dataset) -> float:
    """
    Evaluate the Turney dataset.

    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate.
    :return: The accuracy of the model on the dataset.
    """
    data_list = []
    for row in dataset:
        data_list.append(
            list(
                (
                    row["query"],
                    row["label"],
                    row["candidate_1"],
                    row["candidate_2"],
                    row["candidate_3"],
                    row["candidate_4"],
                )
            )
        )

    num_correct = 0
    for components in data_list:
        emb = cast(np.ndarray, model.encode(components))
        query = emb[0, :]
        matrix = emb[1:, :]
        scores = np.dot(matrix, query)
        chosen = np.argmax(scores)

        if chosen == 0:
            num_correct += 1
    accuracy = num_correct / len(data_list)

    return accuracy


def eval_ppdb(model: Encoder, dataset: Dataset) -> float:
    """
    Evaluate the PPDB dataset.

    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate.
    :return: The accuracy of the model on the dataset.
    """
    phrase1_emb = model.encode(dataset["phrase_1"])
    phrase2_emb = model.encode(dataset["phrase_2"])
    label_list = [1 if e == "pos" else 0 for e in dataset["label"]]

    score = run_probing_model(np.concatenate([phrase1_emb, phrase2_emb], axis=1), label_list)

    return score


def eval_clustering(model: Encoder, dataset: Dataset, name: Literal["conll", "bc5cdr"]) -> float:
    """
    Evaluate the clustering dataset.

    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate.
    :param name: The name of the dataset. Can be "conll" or "bc5cdr".
    :return: The normalized mutual information score of the model on the dataset.
    :raises ValueError: If the dataset name is invalid.
    """
    label_dict = dict()
    match name:
        case "conll":
            label_dict = {"PER": 0, "LOC": 1, "ORG": 2}
        case "bc5cdr":
            label_dict = {"Chemical": 0, "Disease": 1}
        case _:
            raise ValueError(f"Invalid dataset name: {name}")

    num_class = len(label_dict)

    phrases, labels = [], []
    for row in dataset:
        phrases.append(row["entity"] or "NA")
        labels.append(row["label"])

    phrase_emb = model.encode(phrases)
    kmeans = KMeans(n_clusters=num_class, random_state=0).fit(phrase_emb)
    nmi_score = normalized_mutual_info_score(labels, kmeans.labels_)

    return nmi_score


def eval_retrieval(model: Encoder, kb_dataset: Dataset, test_dataset: Dataset) -> float:
    """
    Evaluate the retrieval dataset.

    :param model: The model to evaluate.
    :param kb_dataset: The dataset containing the knowledge base.
    :param test_dataset: The dataset to evaluate.
    :return: The accuracy of the model on the dataset.
    """
    e_names = [x for x in kb_dataset["entity_name"] if x is not None]
    sen_embeddings = model.encode(e_names)

    emb_index = Reach(sen_embeddings, e_names)

    cnt, wrong_cnt = 0, 0
    mentions = test_dataset["query"]
    labels = test_dataset["label"]

    batch_emb = model.encode(mentions)

    I = emb_index.nearest_neighbor(batch_emb)
    predicted = [i[0][0] for i in I]
    for label, predict in zip(labels, predicted):
        cnt += 1
        if predict != label:
            wrong_cnt += 1
    acc = (cnt - wrong_cnt) * 1.0 / cnt

    return acc


def eval_single_autofj(dataset_name: str, model: Encoder) -> float:
    """
    Evaluate a single dataset from the AutoFJ benchmark.

    :param dataset_name: The name of the dataset to evaluate.
    :param model: The model to evaluate.
    :return: The accuracy of the model on the dataset.
    """
    left_table, right_table, gt_table = load_data(dataset_name)
    left_table_list: list[str] = list(left_table.title)
    right_table_list: list[str] = list(right_table.title)
    left_label, right_label = list(gt_table.title_l), list(gt_table.title_r)
    gt_label = dict(zip(right_label, left_label))

    left_embs = normalize(model.encode(left_table_list))
    right_embs = normalize(model.encode(right_table_list))

    acc_cnt, total = 0, 0

    for index, r_t_emb in enumerate(right_embs):
        r_t = right_table_list[index]
        try:
            g_t = gt_label[r_t]
        except KeyError:
            continue

        score = r_t_emb @ left_embs.T
        pred_i = np.argmax(score)
        predicted = left_table_list[pred_i]

        if predicted == g_t:
            acc_cnt += 1
        total += 1
    return acc_cnt * 1.0 / total


def eval_autofj(model: Encoder, dataset: Dataset) -> float:
    """
    Evaluate the AutoFJ benchmark.

    :param model: The model to evaluate.
    :param dataset: The dataset to evaluate.
    :return: The accuracy of the model on the dataset.
    """
    table_names: list[str] = [row["Dataset"] for row in dataset]
    acc_list = []
    for table_name in table_names:
        acc_list.append(eval_single_autofj(dataset_name=table_name, model=model))

    return sum(acc_list) / len(acc_list)
