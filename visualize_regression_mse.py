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
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
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

    # Track losses
    train_losses = []
    train_mse_losses = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_mse = 0
        for X_batch, y_class_batch, y_reg_batch in train_loader:
            optimizer.zero_grad()
            logits, pred_reg = model(X_batch)
            mse_loss = criterion_reg(pred_reg, y_reg_batch)
            loss = 0.6 * criterion_class(logits, y_class_batch) + 0.4 * mse_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_mse += mse_loss.item()
        scheduler.step()
        train_losses.append(total_loss / len(train_loader))
        train_mse_losses.append(total_mse / len(train_loader))
    

    model.eval()
    all_preds_reg = []
    all_true_reg = []

    with torch.no_grad():
        for X_batch, _, y_reg_batch in test_loader:
            _, pred_reg = model(X_batch)
            all_preds_reg.extend(pred_reg.numpy())
            all_true_reg.extend(y_reg_batch.numpy())

    all_preds_reg = np.array(all_preds_reg)
    all_true_reg = np.array(all_true_reg)

    # Calculate errors
    errors = all_preds_reg - all_true_reg
    squared_errors = errors ** 2
    absolute_errors = np.abs(errors)

    # Overall metrics
    mse = mean_squared_error(all_true_reg, all_preds_reg)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_true_reg, all_preds_reg)

    colors_rating = ['#e74c3c', '#f39c12', '#f1c40f', '#3498db', '#27ae60']
    fig1, axes = plt.subplots(2, 2, figsize=(14, 12))
    plt.suptitle('Regression Analysis: Predicted vs Actual Star Ratings\n(Real Yelp Review Data - 300 Test Samples)', 
                 fontsize=14, fontweight='bold', y=1.02)

    ax1 = axes[0, 0]
    scatter = ax1.scatter(all_true_reg, all_preds_reg, c=squared_errors, cmap='RdYlGn_r', 
                           s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.plot([1, 5], [1, 5], 'r--', lw=2, label='Perfect Prediction (y=x)')

    z = np.polyfit(all_true_reg, all_preds_reg, 1)
    p = np.poly1d(z)
    ax1.plot([1, 5], [p(1), p(5)], 'b-', lw=2, label=f'Regression: y={z[0]:.2f}x+{z[1]:.2f}')

    ax1.set_xlabel('Actual Star Rating', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Star Rating', fontsize=12, fontweight='bold')
    ax1.set_title(f'Predicted vs Actual\nMSE: {mse:.3f}, RMSE: {rmse:.3f}', fontsize=11, fontweight='bold')
    ax1.set_xlim([0.5, 5.5])
    ax1.set_ylim([0.5, 5.5])
    ax1.legend(loc='upper left')
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Squared Error', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 1b. Residual plot
    ax2 = axes[0, 1]
    ax2.scatter(all_preds_reg, errors, c=all_true_reg, cmap='viridis', s=50, alpha=0.7, 
                edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='red', linestyle='--', lw=2, label='Zero Error')
    ax2.axhline(y=errors.mean(), color='blue', linestyle='-', lw=2, 
                label=f'Mean Error: {errors.mean():.3f}')
    ax2.fill_between([0.5, 5.5], [-1, -1], [1, 1], alpha=0.2, color='green', label='±1 star')
    ax2.set_xlabel('Predicted Rating', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residual (Predicted - Actual)', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Plot\n(Error by Prediction)', fontsize=11, fontweight='bold')
    ax2.set_xlim([0.5, 5.5])
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(ax2.collections[0], ax=ax2)
    cbar2.set_label('Actual Rating', fontweight='bold')

    # 1c. Error distribution histogram
    ax3 = axes[1, 0]
    ax3.hist(errors, bins=30, color='#3498db', edgecolor='black', alpha=0.7, density=True)
    ax3.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
    ax3.axvline(x=errors.mean(), color='green', linestyle='-', lw=2, 
                label=f'Mean: {errors.mean():.3f}')
    ax3.axvline(x=errors.mean() + errors.std(), color='orange', linestyle=':', lw=2)
    ax3.axvline(x=errors.mean() - errors.std(), color='orange', linestyle=':', lw=2, 
                label=f'±1 Std: {errors.std():.3f}')

    x_norm = np.linspace(errors.min(), errors.max(), 100)
    ax3.plot(x_norm, stats.norm.pdf(x_norm, errors.mean(), errors.std()), 'r-', lw=2, label='Normal Fit')

    ax3.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax3.set_title('Error Distribution', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 1d. Squared error by actual rating
    ax4 = axes[1, 1]
    se_by_rating = [squared_errors[all_true_reg == r] for r in [1, 2, 3, 4, 5]]
    bp = ax4.boxplot(se_by_rating, labels=['1★', '2★', '3★', '4★', '5★'], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_rating):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    means = [np.mean(se) if len(se) > 0 else 0 for se in se_by_rating]
    ax4.scatter(range(1, 6), means, color='red', s=100, zorder=5, label='Mean SE', marker='D')

    ax4.set_xlabel('Actual Star Rating', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Squared Error', fontsize=12, fontweight='bold')
    ax4.set_title('Squared Error by Rating Category', fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('regression_mse_analysis_real.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: regression_mse_analysis_real.png")
    plt.close()

    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.suptitle('Mean Squared Error (MSE) Detailed Analysis\n(Real Yelp Review Data)', 
                 fontsize=14, fontweight='bold', y=1.02)

    ax1 = axes[0, 0]
    mse_by_rating = []
    counts_by_rating = []
    for r in [1, 2, 3, 4, 5]:
        mask = all_true_reg == r
        if mask.sum() > 0:
            mse_by_rating.append(squared_errors[mask].mean())
            counts_by_rating.append(mask.sum())
        else:
            mse_by_rating.append(0)
            counts_by_rating.append(0)

    bars = ax1.bar(range(1, 6), mse_by_rating, color=colors_rating, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=mse, color='red', linestyle='--', lw=2, label=f'Overall MSE: {mse:.3f}')
    ax1.set_xlabel('Actual Star Rating', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
    ax1.set_title('MSE by Rating Category', fontsize=11, fontweight='bold')
    ax1.set_xticks(range(1, 6))
    ax1.set_xticklabels(['1★', '2★', '3★', '4★', '5★'])
    ax1.legend()

    for i, (bar, count, m) in enumerate(zip(bars, counts_by_rating, mse_by_rating)):
        ax1.text(bar.get_x() + bar.get_width()/2, m + 0.1, f'n={count}\nMSE={m:.2f}', 
                 ha='center', fontsize=9, fontweight='bold')

    ax2 = axes[0, 1]
    mae_by_rating = []
    for r in [1, 2, 3, 4, 5]:
        mask = all_true_reg == r
        if mask.sum() > 0:
            mae_by_rating.append(absolute_errors[mask].mean())
        else:
            mae_by_rating.append(0)

    bars = ax2.bar(range(1, 6), mae_by_rating, color=colors_rating, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=mae, color='red', linestyle='--', lw=2, label=f'Overall MAE: {mae:.3f}')
    ax2.set_xlabel('Actual Star Rating', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Absolute Error', fontsize=12, fontweight='bold')
    ax2.set_title('MAE by Rating Category', fontsize=11, fontweight='bold')
    ax2.set_xticks(range(1, 6))
    ax2.set_xticklabels(['1★', '2★', '3★', '4★', '5★'])
    ax2.legend()

    for i, (bar, m) in enumerate(zip(bars, mae_by_rating)):
        ax2.text(bar.get_x() + bar.get_width()/2, m + 0.05, f'{m:.2f}', 
                 ha='center', fontsize=10, fontweight='bold')

    ax3 = axes[0, 2]
    for r in [1, 2, 3, 4, 5]:
        mask = all_true_reg == r
        if mask.sum() > 5:
            preds = all_preds_reg[mask]
            ax3.hist(preds, bins=15, alpha=0.5, label=f'{r}★ (n={mask.sum()})', 
                     color=colors_rating[r-1], edgecolor='black')

    ax3.set_xlabel('Predicted Rating', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('Prediction Distribution by Actual Rating', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 0]
    sorted_ae = np.sort(absolute_errors)
    cumulative = np.arange(1, len(sorted_ae) + 1) / len(sorted_ae)
    ax4.plot(sorted_ae, cumulative, 'b-', lw=2)
    ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=np.median(absolute_errors), color='red', linestyle='--', lw=2, 
                label=f'Median AE: {np.median(absolute_errors):.3f}')
    ax4.axvline(x=1.0, color='green', linestyle=':', lw=2, label='1-star threshold')
    ax4.fill_between(sorted_ae, 0, cumulative, where=(sorted_ae <= 1), alpha=0.3, color='green')

    pct_within_1 = (absolute_errors <= 1).mean() * 100
    ax4.set_xlabel('Absolute Error', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Cumulative Proportion', fontsize=12, fontweight='bold')
    ax4.set_title(f'Cumulative Error Distribution\n{pct_within_1:.1f}% within 1 star', 
                  fontsize=11, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 1]
    epochs_range = range(1, EPOCHS + 1)
    ax5.plot(epochs_range, train_mse_losses, 'b-', lw=2, marker='o', label='Training MSE')
    ax5.axhline(y=mse, color='red', linestyle='--', lw=2, label=f'Test MSE: {mse:.3f}')
    ax5.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax5.set_ylabel('MSE Loss', fontsize=12, fontweight='bold')
    ax5.set_title('Training MSE Over Epochs', fontsize=11, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xticks(epochs_range)

    ax6 = axes[1, 2]
    ax6.axis('off')


    ax6.text(0.02, 0.5, summary_text, fontsize=9, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1))

    plt.tight_layout()
    plt.savefig('regression_mse_detailed_real.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig3, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim([0, 16])
    ax.set_ylim([0, 10])
    ax.axis('off')
    ax.set_title('Mean Squared Error (MSE) Computation Pipeline\nWith Real Examples from Yelp Reviews', 
                 fontsize=14, fontweight='bold', pad=20)

    def draw_box(ax, x, y, w, h, text, color='lightblue', fontsize=10):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')

    # Pipeline boxes
    draw_box(ax, 0.5, 6, 2.5, 2, 'Input Text\n(Tokenized)', 'lightyellow')
    draw_box(ax, 3.5, 5.5, 2.5, 3, 'Multi-Kernel\nCNN + FC', 'lightgreen')
    draw_box(ax, 6.5, 6, 2, 2, 'Regression\nHead', 'lightcoral')
    draw_box(ax, 9, 6, 2, 2, 'Predicted\nRating', 'lightskyblue')
    draw_box(ax, 11.5, 6, 2.5, 2, 'MSE Loss\nCalculation', 'plum')

    # Arrows
    arrow_props = dict(arrowstyle='->', color='black', lw=2)
    ax.annotate('', xy=(3.4, 7), xytext=(3.0, 7), arrowprops=arrow_props)
    ax.annotate('', xy=(6.4, 7), xytext=(6.0, 7), arrowprops=arrow_props)
    ax.annotate('', xy=(8.9, 7), xytext=(8.5, 7), arrowprops=arrow_props)
    ax.annotate('', xy=(11.4, 7), xytext=(11.0, 7), arrowprops=arrow_props)

    # MSE Formula
    ax.text(8, 9.2, r'MSE = $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$', fontsize=14, style='italic', ha='center')
    ax.text(8, 8.6, 'Penalizes larger errors more heavily (quadratic)', fontsize=10, ha='center')

    # Real examples
    ax.text(0.5, 4.5, 'Real Examples from Test Set:', fontsize=12, fontweight='bold')

    sample_indices = [
        np.argmin(squared_errors),
        np.argmax(squared_errors),
        np.argsort(squared_errors)[len(squared_errors)//2],
    ]

    y_pos = 3.8
    for i, idx in enumerate(sample_indices[:3]):
        actual = all_true_reg[idx]
        pred = all_preds_reg[idx]
        error = errors[idx]
        se = squared_errors[idx]
        
        label = ['Best', 'Worst', 'Median'][i]
        color = '#27ae60' if se < 0.5 else '#f39c12' if se < 2 else '#e74c3c'
        
        text = f"{label}: Actual={actual:.0f}★, Predicted={pred:.2f}, Error={error:+.2f}, SE={se:.3f}"
        ax.text(0.5, y_pos, text, fontsize=10, color=color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        y_pos -= 0.6

    # Why square explanation
    ax.text(8, 3.5, 'Why Square the Error?', fontsize=11, fontweight='bold', ha='center')
    ax.text(8, 3.0, '• Removes negative signs (direction-agnostic)', fontsize=9, ha='center')
    ax.text(8, 2.6, '• Penalizes large errors disproportionately', fontsize=9, ha='center')
    ax.text(8, 2.2, '• Differentiable for gradient-based optimization', fontsize=9, ha='center')

    ax.text(12, 4.5, f'Test Set Results:', fontsize=11, fontweight='bold')
    ax.text(12, 4.0, f'MSE: {mse:.4f}', fontsize=10)
    ax.text(12, 3.6, f'RMSE: {rmse:.4f}', fontsize=10)
    ax.text(12, 3.2, f'MAE: {mae:.4f}', fontsize=10)

    plt.savefig('regression_mse_pipeline_real.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig4, axes = plt.subplots(2, 5, figsize=(18, 8))
    plt.suptitle('Sample Regression Predictions\n(Real Yelp Reviews - Showing Actual vs Predicted Ratings)', 
                 fontsize=14, fontweight='bold', y=1.02)

    sample_count = 0
    for rating in [1, 2, 3, 4, 5]:
        mask = all_true_reg == rating
        if mask.sum() > 0:
            indices = np.where(mask)[0]
            rating_errors = absolute_errors[mask]
            best_idx = indices[np.argmin(rating_errors)]
            worst_idx = indices[np.argmax(rating_errors)]
            
            for row, idx in enumerate([best_idx, worst_idx]):
                ax = axes[row, sample_count]
                actual = all_true_reg[idx]
                pred = all_preds_reg[idx]
                error = errors[idx]
                se = squared_errors[idx]
                
                categories = ['Actual', 'Predicted']
                values = [actual, pred]
                colors_bar = ['#3498db', '#e74c3c' if abs(error) > 1 else '#27ae60']
                
                bars = ax.bar(categories, values, color=colors_bar, edgecolor='black', linewidth=1.5)
                ax.axhline(y=actual, color='#3498db', linestyle='--', alpha=0.7)
                ax.set_ylim([0, 5.5])
                ax.set_ylabel('Rating' if sample_count == 0 else '')
                
                status = 'Good' if abs(error) <= 1 else 'Off'
                ax.set_title(f'{status}: Error={error:+.2f}\nSE={se:.3f}', 
                             fontsize=10, color='green' if abs(error) <= 1 else 'red')
                
                if row == 0:
                    ax.set_xlabel(f'Actual: {int(actual)}★', fontweight='bold')
        
        sample_count += 1

    plt.tight_layout()
    plt.savefig('regression_samples_real.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
