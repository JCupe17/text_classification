import logging
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import src.constants as const
from src.preprocessing import preprocessing
from src.ESGDataset import ESGDataset
from src.utils import custom_metrics


def train():

    # Load data
    df = pd.read_csv("data/data.csv")
    # Preprocessing
    df = preprocessing(df)

    # Split in train, val and test datasets
    train_df, test_df = train_test_split(df, test_size=.2, random_state=17, stratify=df["lang"].values)
    train_df, val_df = train_test_split(train_df, test_size=.2, random_state=17, stratify=train_df["lang"].values)

    # Loading tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(const.MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(const.MODEL)

    # Loading Datasets to train and evaluate the model after fine-tuning
    train_dataset = ESGDataset(df=train_df, tokenizer=tokenizer)
    val_dataset = ESGDataset(df=val_df, tokenizer=tokenizer)
    test_dataset = ESGDataset(df=test_df, tokenizer=tokenizer)

    # Training
    training_args = TrainingArguments(
        output_dir='./results',                              # output directory
        num_train_epochs=3,                                  # total number of training epochs
        per_device_train_batch_size=const.TRAIN_BATCH_SIZE,  # batch size per device during training
        per_device_eval_batch_size=const.EVAL_BATCH_SIZE,    # batch size for evaluation
        warmup_steps=500,                                    # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                                   # strength of weight decay
        logging_dir='./logs',                                # directory for storing logs
        logging_steps=50,                                    # N steps to log / compute metrics
        load_best_model_at_end=True,                         # load best model at the end of training
        evaluation_strategy="steps",                         # compute train and val training every N steps
    )

    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset,            # evaluation dataset
        compute_metrics=custom_metrics,      # compute custom metrics
    )

    trainer.train()

    # Saving the best model
    trainer.save_model("model/")

    # Evaluate the results
    test_predictions, test_labels, test_metrics = trainer.predict(test_dataset, metric_key_prefix="predict")
    test_pred = np.argmax(test_predictions, axis=1)
    y_test = [const.tag2idx[label] for label in train_df["label"].tolist()]
    logging.info(classification_report(y_test, test_pred))
    test_df["pred"] = test_pred
    test_df["pred"] = test_df["pred"].map(const.idx2tag)
    # Evaluation by language


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s %(message)s')
    logging.info("Running the training script ...")
    train()
    logging.info("Training DONE!")