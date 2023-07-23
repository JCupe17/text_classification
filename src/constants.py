TAG_VALUES = ["ESG", "NOT ESG"]
TEXT_COLUMN = "title+sentences"
TARGET = "label"
ID = "article_id"
MAX_LENGTH = 512

tag2idx = {k: v for v, k in enumerate(sorted(TAG_VALUES))}
idx2tag = {v: k for v, k in enumerate(sorted(TAG_VALUES))}

MODEL = "bert-base-multilingual-cased"
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 64