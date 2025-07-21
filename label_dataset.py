import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from tqdm import tqdm

# Settings
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INPUT_FILE = 'processed_datasets/combined_dataset_preprocessed.csv'
OUTPUT_FILE = 'processed_datasets/combined_dataset_with_labels.csv'
TEXT_COLUMN = 'preprocessed_text'
MAX_LENGTH = 128  # Safe for most models

# Load model & tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.to(DEVICE)
model.eval()

# Map label index to sentiment
id2label = {0: "Negative", 1: "Neutral", 2: "Positive"}
label2id = {"Negative": 0, "Neutral": 1, "Positive": 2}

# Label a batch of texts
def label_batch(texts):
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        probs = softmax(outputs.logits, dim=1)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        labels = [id2label[i] for i in predictions]
    return labels

# Convert label to int
def convert_label(label):
    return label2id.get(label, -1)

# Process and write in chunks
chunksize = 10_000
with pd.read_csv(INPUT_FILE, chunksize=chunksize) as reader:
    for i, chunk in enumerate(tqdm(reader, desc="Processing chunks")):
        texts = chunk[TEXT_COLUMN].astype(str).tolist()

        # Break into mini-batches
        all_labels = []
        for j in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[j:j + BATCH_SIZE]
            batch_labels = label_batch(batch_texts)
            all_labels.extend(batch_labels)

        chunk['review_label'] = [convert_label(lbl) for lbl in all_labels]

        # Write to file
        if i == 0:
            chunk.to_csv(OUTPUT_FILE, index=False, mode='w')
        else:
            chunk.to_csv(OUTPUT_FILE, index=False, header=False, mode='a')
