import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             mean_squared_error, mean_absolute_error, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = "yelp_cleaned_reviews_1000.csv"
GLOVE_FILE = "glove.6B.100d.txt"
MAX_WORDS = 10000
MAX_LEN = 200
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 15
RANDOM_SEED = 42

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
    def __init__(self, vocab_size, embedding_dim, pretrained_weights=None, 
                 num_classes=4, kernel_sizes=[2,3,4,5], num_filters=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_weights is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weights))
            self.embedding.weight.requires_grad = False
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes
        ])
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(len(kernel_sizes) * num_filters)
        self.dropout1 = nn.Dropout(0.4)
        self.fc_shared = nn.Linear(len(kernel_sizes) * num_filters, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc_class = nn.Linear(256, num_classes)
        self.fc_reg = nn.Linear(256, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x_list = [self.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x_list, dim=1)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.relu(self.fc_shared(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        return self.fc_class(x), self.fc_reg(x).squeeze(-1)

def main():
    print("=" * 70)
    print("CLASSIC ML vs NEURAL NETWORK COMPARISON")
    print("=" * 70)

    print("\n[1/5] Loading and preparing data...")
    df = pd.read_csv(DATA_FILE)
    df.dropna(subset=["cleaned_text"], inplace=True)

    # Labels
    y_class = df["stars"].apply(lambda x: 0 if x < 3 else 1 if x == 3 else 2 if x == 4 else 3).values
    y_reg = df["stars"].astype(float).values

    # Train/test split
    X_train_text, X_test_text, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        df["cleaned_text"].values, y_class, y_reg, test_size=0.3, random_state=RANDOM_SEED
    )
    print(f"Train: {len(X_train_text)}, Test: {len(X_test_text)}")

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    classic_results = {}

    lr_clf = LogisticRegression(max_iter=1000, multi_class='multinomial', C=1.0)
    lr_clf.fit(X_train_tfidf, y_train_class)
    lr_preds_class = lr_clf.predict(X_test_tfidf)
    lr_probs = lr_clf.predict_proba(X_test_tfidf)
    lr_preds_reg = np.sum(lr_probs * np.array([1.5, 3, 4, 5]), axis=1)

    classic_results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test_class, lr_preds_class),
        'precision': precision_score(y_test_class, lr_preds_class, average='weighted'),
        'recall': recall_score(y_test_class, lr_preds_class, average='weighted'),
        'f1': f1_score(y_test_class, lr_preds_class, average='weighted'),
        'mse': mean_squared_error(y_test_reg, lr_preds_reg),
        'mae': mean_absolute_error(y_test_reg, lr_preds_reg),
        'preds_class': lr_preds_class,
        'preds_reg': lr_preds_reg
    }

    nb_clf = MultinomialNB(alpha=0.1)
    nb_clf.fit(X_train_tfidf, y_train_class)
    nb_preds_class = nb_clf.predict(X_test_tfidf)
    nb_probs = nb_clf.predict_proba(X_test_tfidf)
    nb_preds_reg = np.sum(nb_probs * np.array([1.5, 3, 4, 5]), axis=1)

    classic_results['Naive Bayes'] = {
        'accuracy': accuracy_score(y_test_class, nb_preds_class),
        'precision': precision_score(y_test_class, nb_preds_class, average='weighted'),
        'recall': recall_score(y_test_class, nb_preds_class, average='weighted'),
        'f1': f1_score(y_test_class, nb_preds_class, average='weighted'),
        'mse': mean_squared_error(y_test_reg, nb_preds_reg),
        'mae': mean_absolute_error(y_test_reg, nb_preds_reg),
        'preds_class': nb_preds_class,
        'preds_reg': nb_preds_reg
    }

    svm_clf = LinearSVC(max_iter=2000, C=0.5)
    svm_clf.fit(X_train_tfidf, y_train_class)
    svm_preds_class = svm_clf.predict(X_test_tfidf)
    svm_scores = svm_clf.decision_function(X_test_tfidf)
    svm_preds_reg = np.clip(1 + 4 * (svm_scores.argmax(axis=1) / 3), 1, 5)

    classic_results['SVM'] = {
        'accuracy': accuracy_score(y_test_class, svm_preds_class),
        'precision': precision_score(y_test_class, svm_preds_class, average='weighted'),
        'recall': recall_score(y_test_class, svm_preds_class, average='weighted'),
        'f1': f1_score(y_test_class, svm_preds_class, average='weighted'),
        'mse': mean_squared_error(y_test_reg, svm_preds_reg),
        'mae': mean_absolute_error(y_test_reg, svm_preds_reg),
        'preds_class': svm_preds_class,
        'preds_reg': svm_preds_reg
    }
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=RANDOM_SEED, n_jobs=-1)
    rf_clf.fit(X_train_tfidf, y_train_class)
    rf_preds_class = rf_clf.predict(X_test_tfidf)
    rf_probs = rf_clf.predict_proba(X_test_tfidf)
    rf_preds_reg = np.sum(rf_probs * np.array([1.5, 3, 4, 5]), axis=1)

    classic_results['Random Forest'] = {
        'accuracy': accuracy_score(y_test_class, rf_preds_class),
        'precision': precision_score(y_test_class, rf_preds_class, average='weighted'),
        'recall': recall_score(y_test_class, rf_preds_class, average='weighted'),
        'f1': f1_score(y_test_class, rf_preds_class, average='weighted'),
        'mse': mean_squared_error(y_test_reg, rf_preds_reg),
        'mae': mean_absolute_error(y_test_reg, rf_preds_reg),
        'preds_class': rf_preds_class,
        'preds_reg': rf_preds_reg
    }

    print("   Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_tfidf, y_train_reg)
    ridge_preds_reg = np.clip(ridge.predict(X_test_tfidf), 1, 5)
    ridge_preds_class = np.digitize(ridge_preds_reg, [3, 4, 5])

    classic_results['Ridge Regression'] = {
        'accuracy': accuracy_score(y_test_class, ridge_preds_class),
        'precision': precision_score(y_test_class, ridge_preds_class, average='weighted', zero_division=0),
        'recall': recall_score(y_test_class, ridge_preds_class, average='weighted', zero_division=0),
        'f1': f1_score(y_test_class, ridge_preds_class, average='weighted', zero_division=0),
        'mse': mean_squared_error(y_test_reg, ridge_preds_reg),
        'mae': mean_absolute_error(y_test_reg, ridge_preds_reg),
        'preds_class': ridge_preds_class,
        'preds_reg': ridge_preds_reg
    }

    # Tokenize
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train_text)
    X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN)
    X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN)

    # Load GloVe
    embedding_matrix = np.random.randn(MAX_WORDS, EMBEDDING_DIM).astype(np.float32) * 0.01
    if os.path.exists(GLOVE_FILE):
        embeddings_index = {}
        with open(GLOVE_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
        for word, idx in tokenizer.word_index.items():
            if idx < MAX_WORDS and word in embeddings_index:
                embedding_matrix[idx] = embeddings_index[word]

    # Create datasets
    train_dataset = ReviewDataset(X_train_padded, y_train_class, y_train_reg)
    test_dataset = ReviewDataset(X_test_padded, y_test_class, y_test_reg)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    model = MultiKernelCNN(MAX_WORDS, EMBEDDING_DIM, embedding_matrix)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # Train
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_class_batch, y_reg_batch in train_loader:
            optimizer.zero_grad()
            logits, pred_reg = model(X_batch)
            loss = 0.6 * criterion_class(logits, y_class_batch) + 0.4 * criterion_reg(pred_reg, y_reg_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    # Evaluate NN
    model.eval()
    nn_preds_class, nn_preds_reg = [], []
    with torch.no_grad():
        for X_batch, _, _ in test_loader:
            logits, pred_reg = model(X_batch)
            nn_preds_class.extend(torch.argmax(logits, dim=1).numpy())
            nn_preds_reg.extend(pred_reg.numpy())

    nn_preds_class = np.array(nn_preds_class)
    nn_preds_reg = np.array(nn_preds_reg)

    classic_results['Neural Network (CNN)'] = {
        'accuracy': accuracy_score(y_test_class, nn_preds_class),
        'precision': precision_score(y_test_class, nn_preds_class, average='weighted'),
        'recall': recall_score(y_test_class, nn_preds_class, average='weighted'),
        'f1': f1_score(y_test_class, nn_preds_class, average='weighted'),
        'mse': mean_squared_error(y_test_reg, nn_preds_reg),
        'mae': mean_absolute_error(y_test_reg, nn_preds_reg),
        'preds_class': nn_preds_class,
        'preds_reg': nn_preds_reg
    }

    # Summary DataFrame
    summary_data = []
    for model_name, metrics in classic_results.items():
        summary_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'MSE': metrics['mse'],
            'MAE': metrics['mae'],
            'RMSE': np.sqrt(metrics['mse'])
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Accuracy', ascending=False)
    summary_df.to_csv('model_comparison_metrics.csv', index=False)

    fig1, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.style.use('seaborn-v0_8-whitegrid')

    models = list(classic_results.keys())
    colors = ['#3498db', '#2980b9', '#1abc9c', '#27ae60', '#9b59b6', '#e74c3c']

    ax1 = axes[0, 0]
    accuracies = [classic_results[m]['accuracy'] for m in models]
    bars = ax1.barh(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Classification Accuracy: Classic ML vs Neural Network', fontsize=13, fontweight='bold')
    ax1.set_xlim([0, 1])
    for bar, acc in zip(bars, accuracies):
        ax1.text(acc + 0.01, bar.get_y() + bar.get_height()/2, f'{acc:.3f}',
                 va='center', fontweight='bold', fontsize=10)

    ax2 = axes[0, 1]
    f1_scores = [classic_results[m]['f1'] for m in models]
    bars = ax2.barh(models, f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('F1-Score (Weighted)', fontsize=12, fontweight='bold')
    ax2.set_title('F1-Score: Classic ML vs Neural Network', fontsize=13, fontweight='bold')
    ax2.set_xlim([0, 1])
    for bar, f1 in zip(bars, f1_scores):
        ax2.text(f1 + 0.01, bar.get_y() + bar.get_height()/2, f'{f1:.3f}',
                 va='center', fontweight='bold', fontsize=10)

    ax3 = axes[1, 0]
    maes = [classic_results[m]['mae'] for m in models]
    bars = ax3.barh(models, maes, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Mean Absolute Error (lower is better)', fontsize=12, fontweight='bold')
    ax3.set_title('Regression MAE: Classic ML vs Neural Network', fontsize=13, fontweight='bold')
    for bar, mae_val in zip(bars, maes):
        ax3.text(mae_val + 0.02, bar.get_y() + bar.get_height()/2, f'{mae_val:.3f}',
                 va='center', fontweight='bold', fontsize=10)

    ax4 = axes[1, 1]
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics_to_plot))
    width = 0.12
    offsets = np.linspace(-0.3, 0.3, len(models))

    for i, model_name in enumerate(models):
        values = [summary_df[summary_df['Model']==model_name][m].values[0] for m in metrics_to_plot]
        ax4.bar(x + offsets[i], values, width, label=model_name, color=colors[i], edgecolor='black')

    ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax4.set_title('All Classification Metrics Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_to_plot)
    ax4.legend(loc='lower right', fontsize=8)
    ax4.set_ylim([0, 1.1])
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('ml_vs_nn_comparison.png', dpi=300, bbox_inches='tight')

    plt.close()
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))

    best_classic = max([(m, r['accuracy']) for m, r in classic_results.items() 
                        if m != 'Neural Network (CNN)'], key=lambda x: x[1])
    best_classic_name = best_classic[0]
    print(f"   Best Classic ML: {best_classic_name} ({best_classic[1]:.3f})")

    ax1 = axes[0, 0]
    cm_classic = confusion_matrix(y_test_class, classic_results[best_classic_name]['preds_class'])
    sns.heatmap(cm_classic, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Poor', 'Okay', 'Good', 'Amazing'],
                yticklabels=['Poor', 'Okay', 'Good', 'Amazing'])
    ax1.set_title(f'{best_classic_name}\nAccuracy: {classic_results[best_classic_name]["accuracy"]:.3f}',
                  fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label', fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontweight='bold')

    ax2 = axes[0, 1]
    cm_nn = confusion_matrix(y_test_class, nn_preds_class)
    sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Oranges', ax=ax2,
                xticklabels=['Poor', 'Okay', 'Good', 'Amazing'],
                yticklabels=['Poor', 'Okay', 'Good', 'Amazing'])
    ax2.set_title(f'Neural Network (CNN)\nAccuracy: {classic_results["Neural Network (CNN)"]["accuracy"]:.3f}',
                  fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label', fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontweight='bold')

    ax3 = axes[0, 2]
    class_names = ['Poor', 'Okay', 'Good', 'Amazing']
    classic_per_class = []
    nn_per_class = []
    for i in range(4):
        mask = y_test_class == i
        if mask.sum() > 0:
            classic_per_class.append((classic_results[best_classic_name]['preds_class'][mask] == i).sum() / mask.sum())
            nn_per_class.append((nn_preds_class[mask] == i).sum() / mask.sum())

    x = np.arange(4)
    width = 0.35
    ax3.bar(x - width/2, classic_per_class, width, label=best_classic_name, color='#3498db', edgecolor='black')
    ax3.bar(x + width/2, nn_per_class, width, label='Neural Network', color='#e74c3c', edgecolor='black')
    ax3.set_ylabel('Accuracy', fontweight='bold')
    ax3.set_title('Per-Class Accuracy Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names)
    ax3.legend()
    ax3.set_ylim([0, 1.1])

    ax4 = axes[1, 0]
    ax4.scatter(y_test_reg, classic_results[best_classic_name]['preds_reg'], 
                alpha=0.5, s=40, color='#3498db', edgecolors='black', linewidth=0.3)
    ax4.plot([1, 5], [1, 5], 'r--', lw=2, label='Perfect')
    ax4.set_xlabel('Actual Rating', fontweight='bold')
    ax4.set_ylabel('Predicted Rating', fontweight='bold')
    ax4.set_title(f'{best_classic_name}\nMAE: {classic_results[best_classic_name]["mae"]:.3f}', 
                  fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.set_xlim([0.5, 5.5])
    ax4.set_ylim([0.5, 5.5])

    ax5 = axes[1, 1]
    ax5.scatter(y_test_reg, nn_preds_reg, alpha=0.5, s=40, color='#e74c3c', 
                edgecolors='black', linewidth=0.3)
    ax5.plot([1, 5], [1, 5], 'r--', lw=2, label='Perfect')
    ax5.set_xlabel('Actual Rating', fontweight='bold')
    ax5.set_ylabel('Predicted Rating', fontweight='bold')
    ax5.set_title(f'Neural Network (CNN)\nMAE: {classic_results["Neural Network (CNN)"]["mae"]:.3f}', 
                  fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.set_xlim([0.5, 5.5])
    ax5.set_ylim([0.5, 5.5])

    ax6 = axes[1, 2]
    ax6.axis('off')

    nn_acc = classic_results['Neural Network (CNN)']['accuracy']
    classic_acc = classic_results[best_classic_name]['accuracy']
    nn_mae = classic_results['Neural Network (CNN)']['mae']
    classic_mae = classic_results[best_classic_name]['mae']

    ax6.text(0.02, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1))

    plt.tight_layout()
    plt.savefig('ml_vs_nn_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 90)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MAE':>10} {'MSE':>10}")
    print("=" * 90)
    for _, row in summary_df.iterrows():
        print(f"{row['Model']:<25} {row['Accuracy']:>10.4f} {row['Precision']:>10.4f} {row['Recall']:>10.4f} {row['F1-Score']:>10.4f} {row['MAE']:>10.4f} {row['MSE']:>10.4f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
