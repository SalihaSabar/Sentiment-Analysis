import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pickle
import warnings
import os
from collections import Counter

warnings.filterwarnings("ignore")

# === CONFIGURATION ===
INPUT_FILE = "processed_datasets/combined_dataset_with_labels.csv"
TEXT_COL = "preprocessed_text"
LABEL_COL = "review_label"
SAMPLE_FRAC = 0.1
RANDOM_STATE = 42
MAX_VOCAB_SIZE = 50000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 100
GLOVE_PATH = "glove.6B.100d.txt"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === LOAD & SAMPLE DATA ===
print("Loading data...")
df = pd.read_csv(INPUT_FILE, usecols=[TEXT_COL, LABEL_COL])
df = df[df[LABEL_COL].isin([0, 1, 2])].dropna()
df_sample = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)

print(f"Sample size: {len(df_sample)}")
print(f"Label distribution:\n{df_sample[LABEL_COL].value_counts()}")

# === TEXT PREPROCESSING ===
print("\nPreprocessing text...")
texts = df_sample[TEXT_COL].astype(str).values
labels = df_sample[LABEL_COL].values

# Build vocabulary
def build_vocab(texts, max_vocab_size):
    """Build vocabulary from texts"""
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    # Create word to index mapping
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.most_common(max_vocab_size - 2):
        vocab[word] = len(vocab)
    
    return vocab

vocab = build_vocab(texts, MAX_VOCAB_SIZE)
print(f"Vocabulary size: {len(vocab)}")

# Convert text to sequences
def text_to_sequence(text, vocab, max_length):
    """Convert text to sequence of indices"""
    words = text.lower().split()
    sequence = [vocab.get(word, vocab['<UNK>']) for word in words[:max_length]]
    return torch.tensor(sequence, dtype=torch.long)

sequences = [text_to_sequence(text, vocab, MAX_SEQUENCE_LENGTH) for text in texts]

# === LOAD GLOVE EMBEDDINGS ===
def load_glove_embeddings(glove_path, vocab, embedding_dim):
    """Load GloVe embeddings and create embedding matrix"""
    embeddings_index = {}
    
    if os.path.exists(glove_path):
        print(f"Loading GloVe embeddings from {glove_path}...")
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        print(f"Found {len(embeddings_index)} word vectors.")
    else:
        print(f"GloVe file not found at {glove_path}")
        print("Please download GloVe embeddings from: https://nlp.stanford.edu/projects/glove/")
        print("Using random embeddings instead...")
        return None
    
    # Create embedding matrix
    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    found_words = 0
    
    for word, idx in vocab.items():
        if word in ['<PAD>', '<UNK>']:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
            found_words += 1
    
    print(f"Found embeddings for {found_words} words out of {len(vocab)}")
    return embedding_matrix

embedding_matrix = load_glove_embeddings(GLOVE_PATH, vocab, EMBEDDING_DIM)

# === DATASET CLASS ===
class SentimentDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    sequences, labels = zip(*batch)
    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.stack(labels)

# === BI-LSTM MODEL ===
class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, embedding_matrix=None):
        super(BiLSTMSentiment, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            self.embedding.weight.requires_grad = True
        
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim * 2)
        
        # Use the last output for classification
        lstm_out = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

# === SPLIT DATA ===
X_train, X_val, y_train, y_val = train_test_split(
    sequences,
    labels,
    test_size=0.2,
    stratify=labels,
    random_state=RANDOM_STATE
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

# Create datasets and dataloaders
train_dataset = SentimentDataset(X_train, y_train)
val_dataset = SentimentDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# === BUILD MODEL ===
print("\nBuilding Bi-LSTM model...")
model = BiLSTMSentiment(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=128,
    num_layers=2,
    num_classes=3,
    embedding_matrix=embedding_matrix
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# === TRAINING FUNCTION ===
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    
    return total_loss / len(dataloader), correct / total

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())
    
    return total_loss / len(dataloader), correct / total, all_preds, all_targets

# === TRAIN MODEL ===
print("\nTraining Bi-LSTM model...")
best_val_acc = 0
patience_counter = 0
patience = 5

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, device)
    
    scheduler.step(val_loss)
    
    print(f'Epoch {epoch+1}/{EPOCHS}:')
    print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'best_bilstm_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

# Load best model
model.load_state_dict(torch.load('best_bilstm_model.pth'))

# === EVALUATE MODEL ===
print("\nEvaluating best model...")
val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, device)

print("\nValidation Performance:")
print(classification_report(val_targets, val_preds))
print("Confusion Matrix:")
print(confusion_matrix(val_targets, val_preds))

# === TRAIN ON FULL DATASET ===
print("\nTraining final model on full dataset...")
full_sequences = [text_to_sequence(text, vocab, MAX_SEQUENCE_LENGTH) for text in df[TEXT_COL].astype(str)]
full_labels = df[LABEL_COL].values

full_dataset = SentimentDataset(full_sequences, full_labels)
full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# Create final model
final_model = BiLSTMSentiment(
    vocab_size=len(vocab),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=128,
    num_layers=2,
    num_classes=3,
    embedding_matrix=embedding_matrix
).to(device)

final_optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
final_scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, mode='min', factor=0.5, patience=3)

# Train on full dataset
for epoch in range(15):  # Fewer epochs for full dataset
    train_loss, train_acc = train_epoch(final_model, full_loader, criterion, final_optimizer, device)
    final_scheduler.step(train_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch {epoch+1}/15: Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')

# === SAVE MODEL ===
print("\nSaving model and vocabulary...")
torch.save(final_model.state_dict(), "bilstm_glove_model.pth")

# Save vocabulary
with open("bilstm_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("\n✅ Done. Model and vocabulary saved.")
print("Files saved:")
print("- bilstm_glove_model.pth (PyTorch model)")
print("- bilstm_vocab.pkl (Vocabulary)")

# === PREDICTION FUNCTION ===
def predict_sentiment(text, model, vocab, device, max_length=200):
    """Predict sentiment for a given text"""
    model.eval()
    
    # Preprocess text
    sequence = text_to_sequence(text, vocab, max_length)
    sequence = sequence.unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        output = model(sequence)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = torch.max(probabilities).item()
    
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[predicted_class], confidence

# Example usage
print("\nExample prediction:")
sample_text = "This movie was absolutely fantastic!"
sentiment, confidence = predict_sentiment(sample_text, final_model, vocab, device)
print(f"Text: '{sample_text}'")
print(f"Predicted sentiment: {sentiment} (confidence: {confidence:.3f})") 