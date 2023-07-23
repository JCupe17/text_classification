import numpy as np
import pandas as pd
from evaluate import load
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def custom_metrics(eval_pred):
    metric1 = load("precision")
    metric2 = load("recall")
    metric3 = load("f1")
    metric4 = load("accuracy")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision = metric1.compute(predictions=predictions, references=labels, average="micro")["precision"]
    recall = metric2.compute(predictions=predictions, references=labels, average="micro")["recall"]
    f1 = metric3.compute(predictions=predictions, references=labels, average="micro")["f1"]
    accuracy = metric4.compute(predictions=predictions, references=labels)["accuracy"]

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def compute_metrics_by_category(df: pd.DataFrame, category_column: str, true_column: str, pred_column: str, pos_label: str = "ESG") -> pd.DataFrame:

    category_metrics_df = pd.DataFrame()

    categories = df[category_column].unique()

    for category in categories:
        sub_df = df[df[category_column] == category].reset_index(drop=True)
        metrics = {
            "category": category,
            "support": len(sub_df),
            "accuracy": accuracy_score(sub_df[true_column].values, sub_df[pred_column].values),
            "precision": precision_score(sub_df[true_column].values, sub_df[pred_column].values, pos_label=pos_label),
            "recall": recall_score(sub_df[true_column].values, sub_df[pred_column].values, pos_label=pos_label),
            "f1_score": f1_score(sub_df[true_column].values, sub_df[pred_column].values, pos_label=pos_label),
        }
        category_metrics_df = pd.concat([category_metrics_df, pd.DataFrame([metrics])], ignore_index=True)
    
    return category_metrics_df
