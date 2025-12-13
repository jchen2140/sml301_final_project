import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
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
    print("CLASSIFICATION SOFTMAX OUTPUT VISUALIZATION (REAL DATA)")
    print("=" * 70)

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

    # Tokenize
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train_text)
    X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train_text), maxlen=MAX_LEN)
    X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN)

    # Load GloVe embeddings
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
        print("GloVe embeddings loaded")

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

    model.eval()
    all_logits = []
    all_probs = []
    all_preds = []
    all_true = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for X_batch, y_class_batch, _ in test_loader:
            logits, _ = model(X_batch)
            probs = softmax(logits)
            preds = torch.argmax(logits, dim=1)
            
            all_logits.extend(logits.numpy())
            all_probs.extend(probs.numpy())
            all_preds.extend(preds.numpy())
            all_true.extend(y_class_batch.numpy())

    all_logits = np.array(all_logits)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)

    print(f"Extracted probabilities for {len(all_probs)} test samples")

    class_names = ['Poor (1-2★)', 'Okay (3★)', 'Good (4★)', 'Amazing (5★)']
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60']

    # Calculate metrics
    max_probs = all_probs.max(axis=1)
    correct_mask = all_preds == all_true
    incorrect_mask = ~correct_mask
    entropy = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=1)

    fig1, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.suptitle('Softmax Probability Distributions by True Class\n(Real Yelp Review Data - 300 Test Samples)', 
                 fontsize=14, fontweight='bold', y=1.02)

    for true_class in range(4):
        ax = axes[true_class // 2, true_class % 2]
        mask = all_true == true_class
        class_probs = all_probs[mask]
        
        if len(class_probs) > 0:
            bp = ax.boxplot([class_probs[:, i] for i in range(4)],
                            labels=class_names, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            bp['boxes'][true_class].set_linewidth(3)
            bp['boxes'][true_class].set_edgecolor('black')
            
            ax.set_ylabel('Softmax Probability', fontweight='bold')
            ax.set_title(f'True Class: {class_names[true_class]}\n(n={mask.sum()} samples)', 
                         fontsize=11, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
            ax.tick_params(axis='x', rotation=15)
            
            means = [class_probs[:, i].mean() for i in range(4)]
            for i, mean in enumerate(means):
                ax.text(i+1, mean + 0.05, f'{mean:.2f}', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('softmax_distributions_real.png', dpi=300, bbox_inches='tight')
    plt.close()
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.suptitle('Softmax Confidence Analysis\n(Real Yelp Review Data)', 
                 fontsize=14, fontweight='bold', y=1.02)

    ax1 = axes[0, 0]
    ax1.hist(max_probs[correct_mask], bins=20, alpha=0.7, 
             label=f'Correct (n={correct_mask.sum()})', color='#27ae60', edgecolor='black')
    ax1.hist(max_probs[incorrect_mask], bins=20, alpha=0.7, 
             label=f'Incorrect (n={incorrect_mask.sum()})', color='#e74c3c', edgecolor='black')
    ax1.set_xlabel('Maximum Softmax Probability (Confidence)', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Prediction Confidence Distribution', fontweight='bold')
    ax1.legend()
    ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)

    ax2 = axes[0, 1]
    confidence_bins = np.linspace(0.25, 1.0, 8)
    bin_accuracies = []
    bin_counts = []
    bin_centers = []

    for i in range(len(confidence_bins) - 1):
        mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i+1])
        if mask.sum() > 0:
            acc = (all_preds[mask] == all_true[mask]).mean()
            bin_accuracies.append(acc)
            bin_counts.append(mask.sum())
            bin_centers.append((confidence_bins[i] + confidence_bins[i+1]) / 2)

    ax2.bar(bin_centers, bin_accuracies, width=0.08, color='#3498db', edgecolor='black', alpha=0.8)
    ax2.plot([0.25, 1.0], [0.25, 1.0], 'r--', lw=2, label='Perfect calibration')
    ax2.set_xlabel('Confidence Bin', fontweight='bold')
    ax2.set_ylabel('Accuracy in Bin', fontweight='bold')
    ax2.set_title('Confidence vs Accuracy (Calibration)', fontweight='bold')
    ax2.set_xlim([0.2, 1.05])
    ax2.set_ylim([0, 1.1])
    ax2.legend()

    for center, acc, count in zip(bin_centers, bin_accuracies, bin_counts):
        ax2.text(center, acc + 0.05, f'n={count}', ha='center', fontsize=8)

    # 2c. Entropy distribution
    ax3 = axes[1, 0]
    max_entropy = np.log(4)
    ax3.hist(entropy[correct_mask], bins=20, alpha=0.7, label='Correct', color='#27ae60', edgecolor='black')
    ax3.hist(entropy[incorrect_mask], bins=20, alpha=0.7, label='Incorrect', color='#e74c3c', edgecolor='black')
    ax3.axvline(x=max_entropy, color='purple', linestyle='--', lw=2, label=f'Max entropy ({max_entropy:.2f})')
    ax3.set_xlabel('Prediction Entropy', fontweight='bold')
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title('Entropy Distribution (Uncertainty Measure)', fontweight='bold')
    ax3.legend()

    # 2d. Top-2 probability comparison
    ax4 = axes[1, 1]
    sorted_probs = np.sort(all_probs, axis=1)[:, ::-1]
    ax4.scatter(sorted_probs[:, 0], sorted_probs[:, 1],
                c=['#27ae60' if c else '#e74c3c' for c in correct_mask],
                alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax4.set_xlabel('Highest Probability (P₁)', fontweight='bold')
    ax4.set_ylabel('Second Highest Probability (P₂)', fontweight='bold')
    ax4.set_title('Top-2 Probability Comparison', fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#27ae60', label='Correct'),
                       Patch(facecolor='#e74c3c', label='Incorrect')]
    ax4.legend(handles=legend_elements)

    plt.tight_layout()
    plt.savefig('softmax_confidence_analysis_real.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: softmax_confidence_analysis_real.png")
    plt.close()

    fig3, axes = plt.subplots(4, 4, figsize=(16, 14))
    plt.suptitle('Sample Softmax Outputs by True Class\n(Real Yelp Review Predictions)', 
                 fontsize=14, fontweight='bold', y=1.01)

    for true_class in range(4):
        mask = all_true == true_class
        indices = np.where(mask)[0]
        
        correct_idx = indices[all_preds[indices] == true_class][:2]
        incorrect_idx = indices[all_preds[indices] != true_class][:2]
        sample_indices = np.concatenate([correct_idx, incorrect_idx])[:4]
        
        for col, idx in enumerate(sample_indices):
            if col >= 4:
                break
            ax = axes[true_class, col]
            probs = all_probs[idx]
            pred = all_preds[idx]
            is_correct = pred == true_class
            
            bars = ax.bar(range(4), probs, color=colors, edgecolor='black', linewidth=1.5)
            bars[pred].set_edgecolor('gold' if is_correct else 'red')
            bars[pred].set_linewidth(3)
            
            ax.set_xticks(range(4))
            ax.set_xticklabels(['Poor', 'Okay', 'Good', 'Amaz'], fontsize=8)
            ax.set_ylim([0, 1])
            ax.set_ylabel('Prob' if col == 0 else '')
            
            status = '✓' if is_correct else '✗'
            conf = probs[pred]
            ax.set_title(f'{status} Pred: {class_names[pred].split()[0]}\nConf: {conf:.2f}', 
                         fontsize=9, color='green' if is_correct else 'red')
            
            if col == 0:
                ax.set_ylabel(f'True: {class_names[true_class].split()[0]}', 
                             fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('softmax_samples_real.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: softmax_samples_real.png")
    plt.close()

    fig4, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim([0, 16])
    ax.set_ylim([0, 10])
    ax.axis('off')
    ax.set_title('Softmax Classification Pipeline (Multi-Kernel CNN)\nWith Real Output Examples', 
                 fontsize=14, fontweight='bold', pad=20)

    def draw_box(ax, x, y, w, h, text, color='lightblue', fontsize=10):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

    # Pipeline boxes
    draw_box(ax, 0.5, 4, 2.5, 2, 'Input Text\n(Tokenized)', 'lightyellow')
    draw_box(ax, 3.5, 3, 2.5, 4, 'Multi-Kernel\nCNN\n(2,3,4,5)', 'lightgreen')
    draw_box(ax, 6.5, 4, 2, 2, 'Feature\nVector\n(1024-d)', 'lightcoral')
    draw_box(ax, 9, 4, 1.5, 2, 'FC Layer\n(256-d)', 'plum')
    draw_box(ax, 11, 4, 1.5, 2, 'Logits\n(4-d)', 'lightskyblue')
    draw_box(ax, 13, 4, 2, 2, 'Softmax\nP(class)', 'lightgreen')

    # Arrows
    arrow_props = dict(arrowstyle='->', color='black', lw=2)
    ax.annotate('', xy=(3.4, 5), xytext=(3.0, 5), arrowprops=arrow_props)
    ax.annotate('', xy=(6.4, 5), xytext=(6.0, 5), arrowprops=arrow_props)
    ax.annotate('', xy=(8.9, 5), xytext=(8.5, 5), arrowprops=arrow_props)
    ax.annotate('', xy=(10.9, 5), xytext=(10.5, 5), arrowprops=arrow_props)
    ax.annotate('', xy=(12.9, 5), xytext=(12.5, 5), arrowprops=arrow_props)

    # Real example
    example_idx = np.argmax(max_probs)
    example_probs = all_probs[example_idx]
    example_pred = all_preds[example_idx]

    ax.text(8, 0.5, f'Example Output (Sample #{example_idx}):', fontsize=11, fontweight='bold')

    bar_x = [9, 10.5, 12, 13.5]
    bar_heights = example_probs * 1.5
    for i, (x, h, c) in enumerate(zip(bar_x, bar_heights, colors)):
        rect = plt.Rectangle((x, 1.2), 1, h, facecolor=c, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + 0.5, 1.2 + h + 0.1, f'{example_probs[i]:.2f}', ha='center', fontsize=9, fontweight='bold')
        ax.text(x + 0.5, 0.9, class_names[i].split()[0], ha='center', fontsize=8)

    pred_x = bar_x[example_pred] + 0.5
    ax.annotate('Predicted', xy=(pred_x, 1.2 + bar_heights[example_pred] + 0.4), 
                xytext=(pred_x, 1.2 + bar_heights[example_pred] + 0.8),
                arrowprops=dict(arrowstyle='->', color='gold', lw=2),
                fontsize=9, fontweight='bold', ha='center', color='darkgreen')

    ax.text(8, 8.5, r'Softmax: $P(class_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$', fontsize=12, style='italic')
    ax.text(8, 7.8, 'Converts logits to probabilities that sum to 1', fontsize=10)

    plt.savefig('softmax_pipeline_real.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: softmax_pipeline_real.png")
    plt.close()

    print("\n" + "=" * 70)
    print("SOFTMAX OUTPUT STATISTICS (REAL DATA)")
    print("=" * 70)
    print(f"\nTotal test samples: {len(all_probs)}")
    print(f"Correct predictions: {correct_mask.sum()} ({100*correct_mask.mean():.1f}%)")
    print(f"Incorrect predictions: {incorrect_mask.sum()} ({100*incorrect_mask.mean():.1f}%)")
    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence (correct):   {max_probs[correct_mask].mean():.3f}")
    print(f"  Mean confidence (incorrect): {max_probs[incorrect_mask].mean():.3f}")
    print(f"\nEntropy Statistics:")
    print(f"  Mean entropy (correct):   {entropy[correct_mask].mean():.3f}")
    print(f"  Mean entropy (incorrect): {entropy[incorrect_mask].mean():.3f}")

if __name__ == "__main__":
    main()
