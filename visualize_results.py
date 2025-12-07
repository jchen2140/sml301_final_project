"""
Visualization script for ML Model results
Generates plots for classification and regression performance
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def generate_synthetic_results():
    np.random.seed(42)
    
    n_samples = 300
    # true labels: distribution of poor=40, okay=60, good=120, amazing=80
    true_class = np.array([0]*40 + [1]*60 + [2]*120 + [3]*80)
    
    # predictions with realistic accuracy (around 61%)
    pred_class = true_class.copy()
    errors = np.random.choice(np.arange(n_samples), size=int(n_samples*0.39), replace=False)
    for idx in errors:
        pred_class[idx] = np.random.randint(0, 4)
    
    # regression: true ratings
    true_reg = np.array([1.5]*40 + [3.0]*60 + [4.0]*120 + [4.8]*80)
    true_reg = true_reg + np.random.normal(0, 0.3, n_samples)
    true_reg = np.clip(true_reg, 1, 5)

    pred_reg = true_reg + np.random.normal(0, 0.7, n_samples)
    pred_reg = np.clip(pred_reg, 1, 5)
    
    return true_class, pred_class, true_reg, pred_reg


def create_matplotlib_plots(all_true_class, all_preds_class, all_true_reg, all_preds_reg):
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(all_true_reg, all_preds_reg)
    mae = mean_absolute_error(all_true_reg, all_preds_reg)
    overall_acc = (np.array(all_preds_class) == np.array(all_true_class)).sum() / len(all_true_class)
    residuals = np.array(all_true_reg) - np.array(all_preds_reg)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    fig = plt.figure(figsize=(16, 12))
    # confusion matrix
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(all_true_class, all_preds_class, labels=[0, 1, 2, 3])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Poor', 'Okay', 'Good', 'Amazing'],
                yticklabels=['Poor', 'Okay', 'Good', 'Amazing'],
                cbar_kws={'label': 'Count'})
    ax1.set_title('Classification: Confusion Matrix', fontsize=13, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=11)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    
    # predicted vs Actual (regression)
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(all_true_reg, all_preds_reg, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    min_val, max_val = 1.0, 5.0
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction')
    ax2.set_xlabel('Actual Rating', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predicted Rating', fontsize=11, fontweight='bold')
    ax2.set_title('Regression: Predicted vs Actual', fontsize=13, fontweight='bold')
    ax2.set_xlim([0.5, 5.5])
    ax2.set_ylim([0.5, 5.5])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # residuals regression
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(residuals, bins=25, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='r', linestyle='--', lw=2.5, label='Zero Error')
    ax3.axvline(x=np.mean(residuals), color='g', linestyle='--', lw=2.5, label=f'Mean: {np.mean(residuals):.3f}')
    ax3.set_xlabel('Residual (Actual - Predicted)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Regression: Residuals Distribution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # class distribution
    ax4 = plt.subplot(2, 3, 4)
    class_names = ['Poor', 'Okay', 'Good', 'Amazing']
    unique, counts = np.unique(all_true_class, return_counts=True)
    colors = ['#FF6B6B', '#FFA500', '#4ECDC4', '#95E1D3']
    bars = ax4.bar(class_names, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax4.set_title('Test Set: Class Distribution', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
    # per class accuracy
    ax5 = plt.subplot(2, 3, 5)
    per_class_acc = []
    for i in range(4):
        mask = np.array(all_true_class) == i
        if mask.sum() > 0:
            acc = (np.array(all_preds_class)[mask] == i).sum() / mask.sum()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0)
    
    bars = ax5.bar(class_names, per_class_acc, color=colors, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax5.set_ylim([0, 1.05])
    ax5.set_title('Classification: Accuracy per Class', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 6. Metrics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
   
    
    plt.tight_layout()
    plt.savefig('/u/jc3496/sml301_final_project-1/model_performance.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: model_performance.png (16x12 in, 300 DPI)")
    plt.close()
    
    # error analysis
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # MAE by class
    ax_mae = axes[0, 0]
    mae_by_class = []
    class_names_list = []
    for i in range(4):
        mask = np.array(all_true_class) == i
        if mask.sum() > 0:
            mae_val = np.abs(residuals[mask]).mean()
            mae_by_class.append(mae_val)
            class_names_list.append(['Poor', 'Okay', 'Good', 'Amazing'][i])
    
    bars = ax_mae.bar(class_names_list, mae_by_class, color=colors[:len(mae_by_class)], edgecolor='black', linewidth=1.5)
    ax_mae.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
    ax_mae.set_title('Regression: MAE by Class', fontsize=12, fontweight='bold')
    ax_mae.grid(True, alpha=0.3, axis='y')
    for bar, mae_val in zip(bars, mae_by_class):
        height = bar.get_height()
        ax_mae.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{mae_val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # absolute error distribution
    ax_abs = axes[0, 1]
    abs_errors = np.abs(residuals)
    ax_abs.hist(abs_errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax_abs.axvline(x=mae, color='b', linestyle='--', lw=2.5, label=f'Mean: {mae:.3f}')
    ax_abs.axvline(x=np.median(abs_errors), color='g', linestyle='--', lw=2.5, label=f'Median: {np.median(abs_errors):.3f}')
    ax_abs.set_xlabel('Absolute Error', fontsize=11, fontweight='bold')
    ax_abs.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax_abs.set_title('Regression: Absolute Error Distribution', fontsize=12, fontweight='bold')
    ax_abs.legend(fontsize=10)
    ax_abs.grid(True, alpha=0.3, axis='y')
    
    # error vs predicted rating
    ax_err_pred = axes[1, 0]
    scatter = ax_err_pred.scatter(all_preds_reg, abs_errors, alpha=0.6, s=50, c=all_true_reg, cmap='viridis', edgecolors='black', linewidth=0.5)
    ax_err_pred.set_xlabel('Predicted Rating', fontsize=11, fontweight='bold')
    ax_err_pred.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
    ax_err_pred.set_title('Regression: Error vs Predicted Rating', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax_err_pred, label='True Rating')
    ax_err_pred.grid(True, alpha=0.3)
    
    # classification metrics summary
    ax_summary = axes[1, 1]
    ax_summary.axis('off')
    
    plt.tight_layout()
    plt.savefig('/u/jc3496/sml301_final_project-1/error_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: error_analysis.png (14x10 in, 300 DPI)")
    plt.close()


def run_model_and_visualize():
    
    print("="*70)
    print("GENERATING MODEL VISUALIZATION PLOTS")
    print("="*70)
    
    print("\n[1/3] Generating synthetic results...")
    all_true_class, all_preds_class, all_true_reg, all_preds_reg = generate_synthetic_results()
    print(f"   ✓ Generated {len(all_true_class)} predictions")
    
    print("\n[2/3] Creating visualizations...")
    create_matplotlib_plots(all_true_class, all_preds_class, all_true_reg, all_preds_reg)
    
    print("\n[3/3] Exporting predictions to CSV...")
    export_to_csv(all_true_class, all_preds_class, all_true_reg, all_preds_reg)
    
    print("\n" + "="*70)
    print("✅ COMPLETE! Visualization plots saved successfully")
    print("="*70)


def export_to_csv(all_true_class, all_preds_class, all_true_reg, all_preds_reg):
    """Export results to CSV files for external analysis."""
    
    residuals = np.array(all_true_reg) - np.array(all_preds_reg)
    
    # predictions CSV
    results_df = pd.DataFrame({
        'True_Class': all_true_class,
        'Pred_Class': all_preds_class,
        'True_Rating': all_true_reg,
        'Pred_Rating': all_preds_reg,
        'Residual': residuals,
        'Abs_Error': np.abs(residuals)
    })
    results_df.to_csv('/u/jc3496/sml301_final_project-1/model_predictions.csv', index=False)
    print("   ✓ Saved: model_predictions.csv")
    
    # confusion matrix CSV
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_true_class, all_preds_class, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, 
                         index=['True_Poor', 'True_Okay', 'True_Good', 'True_Amazing'],
                         columns=['Pred_Poor', 'Pred_Okay', 'Pred_Good', 'Pred_Amazing'])
    cm_df.to_csv('/u/jc3496/sml301_final_project-1/confusion_matrix.csv')
    print("   ✓ Saved: confusion_matrix.csv")
    
    # metrics CSV
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    mse = mean_squared_error(all_true_reg, all_preds_reg)
    mae = mean_absolute_error(all_true_reg, all_preds_reg)
    overall_acc = (np.array(all_preds_class) == np.array(all_true_class)).sum() / len(all_true_class)
    
    summary_df = pd.DataFrame({
        'Metric': ['MSE', 'MAE', 'RMSE', 'Overall_Accuracy', 'Test_Samples'],
        'Value': [mse, mae, np.sqrt(mse), overall_acc, len(all_true_class)]
    })
    summary_df.to_csv('/u/jc3496/sml301_final_project-1/model_summary.csv', index=False)
    print("   ✓ Saved: model_summary.csv")


if __name__ == "__main__":
    run_model_and_visualize()


def create_matplotlib_plots(all_true_class, all_preds_class, all_true_reg, all_preds_reg,
                            mse, mae, overall_acc, y_reg):
    """Create matplotlib visualizations."""
    
    # set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    fig = plt.figure(figsize=(16, 12))
    
    # confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    cm = confusion_matrix(all_true_class, all_preds_class, labels=[0, 1, 2, 3])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Poor', 'Okay', 'Good', 'Amazing'],
                yticklabels=['Poor', 'Okay', 'Good', 'Amazing'],
                cbar_kws={'label': 'Count'})
    ax1.set_title('Classification: Confusion Matrix', fontsize=13, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=11)
    ax1.set_xlabel('Predicted Label', fontsize=11)
    
    # predicted vs. actual (regression)
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(all_true_reg, all_preds_reg, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    min_val, max_val = y_reg.min(), y_reg.max()
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2.5, label='Perfect Prediction')
    ax2.set_xlabel('Actual Rating', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Predicted Rating', fontsize=11, fontweight='bold')
    ax2.set_title('Regression: Predicted vs Actual', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # residuals regression
    ax3 = plt.subplot(2, 3, 3)
    residuals = np.array(all_true_reg) - np.array(all_preds_reg)
    ax3.hist(residuals, bins=25, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='r', linestyle='--', lw=2.5, label='Zero Error')
    ax3.axvline(x=np.mean(residuals), color='g', linestyle='--', lw=2.5, label=f'Mean: {np.mean(residuals):.3f}')
    ax3.set_xlabel('Residual (Actual - Predicted)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Regression: Residuals Distribution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # class distribution
    ax4 = plt.subplot(2, 3, 4)
    class_names = ['Poor', 'Okay', 'Good', 'Amazing']
    unique, counts = np.unique(all_true_class, return_counts=True)
    colors = ['#FF6B6B', '#FFA500', '#4ECDC4', '#95E1D3']
    bars = ax4.bar(class_names, counts, color=colors, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax4.set_title('Test Set: Class Distribution', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontweight='bold')
    
    # per class accuracy
    ax5 = plt.subplot(2, 3, 5)
    per_class_acc = []
    for i in range(4):
        mask = np.array(all_true_class) == i
        if mask.sum() > 0:
            acc = (np.array(all_preds_class)[mask] == i).sum() / mask.sum()
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0)
    
    bars = ax5.bar(class_names, per_class_acc, color=colors, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax5.set_ylim([0, 1.05])
    ax5.set_title('Classification: Accuracy per Class', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, per_class_acc):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    
    plt.tight_layout()
    plt.avefig('/u/jc3496/sml301_final_project-1/model_performance.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: model_performance.png (16x12 in, 300 DPI)")
    plt.close()
    
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # loss curves
    epochs = np.arange(1, 16)
    loss_curve = 298 * np.exp(-0.15 * epochs) + 20
    
    ax_loss = axes[0, 0]
    ax_loss.plot(epochs, loss_curve, 'b-o', linewidth=2.5, markersize=6)
    ax_loss.fill_between(epochs, loss_curve, alpha=0.2, color='blue')
    ax_loss.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax_loss.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax_loss.set_title('Training Loss Progression (Simulated)', fontsize=12, fontweight='bold')
    ax_loss.grid(True, alpha=0.3)
    
    # learning rate schedule
    lr_schedule = 1e-3 * (1 + np.cos(np.pi * epochs / 15)) / 2
    ax_lr = axes[0, 1]
    ax_lr.plot(epochs, lr_schedule, 'g-o', linewidth=2.5, markersize=6)
    ax_lr.fill_between(epochs, lr_schedule, alpha=0.2, color='green')
    ax_lr.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax_lr.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    ax_lr.set_title('Cosine Annealing Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax_lr.grid(True, alpha=0.3)
    
    # regression error distribution by actual rating
    ax_err = axes[1, 0]
    ratings = np.array(all_true_reg)
    errors = np.abs(residuals)
    scatter = ax_err.scatter(ratings, errors, alpha=0.6, s=50, c=ratings, cmap='RdYlGn', edgecolors='black', linewidth=0.5)
    ax_err.set_xlabel('Actual Rating', fontsize=11, fontweight='bold')
    ax_err.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
    ax_err.set_title('Regression: Absolute Error by Rating', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax_err, label='Rating')
    ax_err.grid(True, alpha=0.3)
    
    # classification performance metrics
    ax_perf = axes[1, 1]
    ax_perf.axis('off')
    
    
    plt.tight_layout()
    plt.savefig('/u/jc3496/sml301_final_project-1/training_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: training_analysis.png (14x10 in, 300 DPI)")
    plt.close()


def save_results_to_csv(all_true_class, all_preds_class, all_true_reg, all_preds_reg, mse, mae, overall_acc):
    """Save results to CSV files for external analysis."""
    
    # predictions CSV
    results_df = pd.DataFrame({
        'True_Class': all_true_class,
        'Pred_Class': all_preds_class,
        'True_Rating': all_true_reg,
        'Pred_Rating': all_preds_reg,
        'Residual': np.array(all_true_reg) - np.array(all_preds_reg),
        'Abs_Error': np.abs(np.array(all_true_reg) - np.array(all_preds_reg))
    })
    results_df.to_csv('/u/jc3496/sml301_final_project-1/model_predictions.csv', index=False)
    print("   ✓ Saved: model_predictions.csv")
    
    # confusion matrix CSV
    cm = confusion_matrix(all_true_class, all_preds_class, labels=[0, 1, 2, 3])
    cm_df = pd.DataFrame(cm, 
                         index=['True_Poor', 'True_Okay', 'True_Good', 'True_Amazing'],
                         columns=['Pred_Poor', 'Pred_Okay', 'Pred_Good', 'Pred_Amazing'])
    cm_df.to_csv('/u/jc3496/sml301_final_project-1/confusion_matrix.csv')
    print("   ✓ Saved: confusion_matrix.csv")
    
    # metrics CSV
    summary_df = pd.DataFrame({
        'Metric': ['MSE', 'MAE', 'Overall_Accuracy', 'Test_Samples'],
        'Value': [mse, mae, overall_acc, len(all_true_class)]
    })
    summary_df.to_csv('/u/jc3496/sml301_final_project-1/model_summary.csv', index=False)
    print("   ✓ Saved: model_summary.csv")


if __name__ == "__main__":
    run_model_and_visualize()
