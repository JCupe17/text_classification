import torch

import src.constants as const


class ESGDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer):
        texts = df[const.TEXT_COLUMN].tolist()
        labels = df[const.TARGET].tolist()

        self.tokenizer = tokenizer
        self.encodings = self.tokenizer(texts, padding='max_length', max_length=const.MAX_LENGTH, truncation=True, return_tensors="pt")
        self.labels = [const.tag2idx[label] for label in labels]

    def __getitem__(self, idx: int):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
