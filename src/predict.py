import numpy as np
import torch
from torch.utils.data import DataLoader

import src.constants as const
from src.ESGDataset import ESGDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_sentence(model, tokenizer, sentence):

    text = tokenizer(sentence, padding='max_length', max_length=const.MAX_LENGTH, truncation=True, return_tensors="pt")
    mask = text['attention_mask'].to(DEVICE)
    input_id = text['input_ids'].to(DEVICE)

    model.to(DEVICE)
    with torch.no_grad():
        output = model(input_id, token_type_ids=None, attention_mask=mask, labels=None)
    logits = output.logits.to('cpu').numpy()
    predictions = np.argmax(np.array(logits), axis=1)
    prediction_label = [const.idx2tag[i] for i in predictions]
    
    return prediction_label


def predict_dataframe(model, tokenizer, df) -> list[str]:
    """Predicts the class for an input dataframe"""

    # Prepare the DataFrame to load by batch for the inference
    dataset = ESGDataset(df=df, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=const.EVAL_BATCH_SIZE)

    # Send the model to the GPU if available
    model.to(DEVICE)

    # Set the model in inference mode
    model.eval()

    predictions = []

    for data in dataloader:

        # Moving to GPU if available
        mask = data['attention_mask'].squeeze(1).to(DEVICE)
        input_id = data['input_ids'].squeeze(1).to(DEVICE)

        # Telling the model not to compute or store gradients
        # This saves memory and speeds up validation
        with torch.no_grad():
            output = model(input_id, token_type_ids=None, attention_mask=mask, labels=None)
        logits = output.logits.to('cpu').numpy()
        pred = np.argmax(np.array(logits), axis=1)
        pred_label = [const.idx2tag[i] for i in pred]

        predictions.extend(pred_label)

    return predictions
