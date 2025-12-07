import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_DIM = 100
BATCH_SIZE = 16
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GLOVE_FILE = "/u/jc3496/sml301_final_project-1/glove.6B.100d.txt"
LEARNING_RATE = 1e-3
CLASS_LOSS_WEIGHT = 0.6
REG_LOSS_WEIGHT = 0.4

def load_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=["cleaned_text"], inplace=True)
    return df

def load_glove_embeddings(glove_file, tokenizer, embedding_dim, max_words):
    """
    Load GloVe embeddings and create embedding matrix for tokenizer vocabulary.
    Returns numpy array of shape (max_words, embedding_dim).
    """
    embedding_matrix = np.random.randn(max_words, embedding_dim).astype(np.float32) * 0.01
    
    if not os.path.exists(glove_file):
        print(f"GloVe file not found at {glove_file}. Using random initialization.")
        return embedding_matrix
    
    print(f"Loading GloVe embeddings from {glove_file}...")
    embeddings_index = {}
    word_count = 0
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            word_count += 1
            if word_count % 100000 == 0:
                print(f"  Loaded {word_count} words from GloVe...")
    
    # populate embedding matrix using tokenizer word index
    matched_count = 0
    for word, idx in tokenizer.word_index.items():
        if idx < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
                matched_count += 1
    
    print(f"Loaded {len(embeddings_index)} GloVe words. Matched {matched_count} words in vocab.")
    return embedding_matrix

class ReviewDataset(Dataset):
    def __init__(self, X, y_class, y_reg):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_reg = torch.tensor(y_reg, dtype=torch.float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_class[idx], self.y_reg[idx]

class MultiKernelCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pretrained_weights=None, num_classes=4, kernel_sizes=[3,4,5], num_filters=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))
            self.embedding.weight.requires_grad = False  # freeze

        # multiple convolution kernels
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_shared = nn.Linear(len(kernel_sizes)*num_filters, 128)

        # heads
        self.fc_class = nn.Linear(128, num_classes)
        self.fc_reg = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)           # (batch, seq, embed)
        x = x.permute(0,2,1)            # (batch, embed, seq)
        x_list = [self.relu(conv(x)).max(dim=2)[0] for conv in self.convs]  # max-over-time
        x = torch.cat(x_list, dim=1)
        x = self.dropout(x)
        x = self.relu(self.fc_shared(x))
        return self.fc_class(x), self.fc_reg(x).squeeze(-1)

def get_category_from_rating(rating):
    if rating < 3:
        return "poor"
    elif rating < 4:
        return "okay"
    elif rating < 5:
        return "good"
    else:
        return "amazing"

def predict_review(model, tokenizer, review_text):
    seq = tokenizer.texts_to_sequences([review_text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    x = torch.tensor(padded, dtype=torch.long).to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits_class, pred_reg = model(x)
        pred_class = torch.argmax(logits_class, dim=1).item()
        pred_rating = pred_reg.item()
    category = get_category_from_rating(pred_rating)
    return pred_rating, category, pred_class + 1

if __name__ == "__main__":
    print("Starting model training with GloVe embeddings...")
    filepath = "/u/jc3496/sml301_final_project-1/yelp_cleaned_reviews_1000.csv"
    df = load_data(filepath)
    print(f"Loaded {len(df)} reviews")

    # classification (4 classes)
    y_class = df["stars"].apply(lambda x: 0 if x<3 else 1 if x==3 else 2 if x==4 else 3).values
    y_reg = df["stars"].astype(float).values

    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        df["cleaned_text"], y_class, y_reg, test_size=0.3, random_state=42
    )

    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train)
    X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
    X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN)

    train_dataset = ReviewDataset(X_train_padded, y_train_class, y_train_reg)
    test_dataset = ReviewDataset(X_test_padded, y_test_class, y_test_reg)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # load GloVe embeddings
    pretrained_weights = load_glove_embeddings(GLOVE_FILE, tokenizer, EMBEDDING_DIM, MAX_WORDS)

    model = MultiKernelCNN(MAX_WORDS, EMBEDDING_DIM, pretrained_weights).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    print("Training model...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_class_batch, y_reg_batch in train_loader:
            X_batch, y_class_batch, y_reg_batch = X_batch.to(DEVICE), y_class_batch.to(DEVICE), y_reg_batch.to(DEVICE)

            optimizer.zero_grad()
            logits_class, pred_reg = model(X_batch)
            loss_class = criterion_class(logits_class, y_class_batch)
            loss_reg = criterion_reg(pred_reg, y_reg_batch)
            loss = 0.7*loss_class + 0.3*loss_reg  # weighted multi-task loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    model.eval()
    all_true_class, all_preds_class = [], []
    all_true_reg, all_preds_reg = [], []

    with torch.no_grad():
        for X_batch, y_class_batch, y_reg_batch in test_loader:
            X_batch, y_class_batch, y_reg_batch = X_batch.to(DEVICE), y_class_batch.to(DEVICE), y_reg_batch.to(DEVICE)
            logits_class, pred_reg = model(X_batch)
            preds_class = torch.argmax(logits_class, dim=1)
            all_true_class.extend(y_class_batch.cpu().numpy())
            all_preds_class.extend(preds_class.cpu().numpy())
            all_true_reg.extend(y_reg_batch.cpu().numpy())
            all_preds_reg.extend(pred_reg.cpu().numpy())

    print("\n--- Classification Report ---")
    print(classification_report(all_true_class, all_preds_class,
                                target_names=["poor","okay","good","amazing"]))

    mse = mean_squared_error(all_true_reg, all_preds_reg)
    print(f"\nRegression MSE: {mse:.4f}")

    print("\n--- Example Predictions ---")
    examples = [
        "The food was absolutely amazing and the service was the best I have ever had. I will be back!",
        "It was okay, but the food was cold and the waiter was rude. I will not be returning.",
        "A decent experience. The burger was good, not great. Service was a little slow."
    ]
    for text in examples:
        rating, cat, cls = predict_review(model, tokenizer, text)
        print(f"\nReview: {text}")
        print(f"Predicted Rating: {rating:.2f}")
        print(f"Predicted Category: {cat} (class {cls})")
