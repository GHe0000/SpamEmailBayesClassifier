import numpy as np
import matplotlib.pyplot as plt

from BayesClassification import load_model, predict
from Data import load_dataset


# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, class_labels, title='Confusion Matrix'):
    num_classes = len(class_labels)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    class_to_idx = {cls: idx for idx, cls in enumerate(class_labels)}
    for true, pred in zip(y_true, y_pred):
        true_idx = class_to_idx[true]
        pred_idx = class_to_idx[pred]
        cm[true_idx, pred_idx] += 1
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Counts', rotation=-90, va="bottom")
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_labels,
        yticklabels=class_labels,
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2  # 文字颜色阈值
    for i in range(num_classes):
        for j in range(num_classes):
            total = cm[i].sum()
            percentage = cm[i, j] / total if total != 0 else 0
            text = f"{cm[i, j]}\n({percentage:.1%})"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.grid(False)
    plt.show()


if __name__ == "__main__":
    X_test, y_test = load_dataset('./Data/ProcessedData/', 500)
    model = load_model("./Model/bayes.npz")
    y_pred = predict(model, X_test)
    accuracy = np.mean(np.array(y_pred) == np.array(y_test))
    print("Accuracy:", accuracy)

    y_true_01 = np.array(y_test) == 'spam'
    y_pred_01 = np.array(y_pred) == 'spam'
    precision = y_true_01[y_pred_01].mean()
    recall = y_pred_01[y_true_01].mean()
    f1 = (2 * precision * recall / (precision + recall))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    plot_confusion_matrix(
        y_test,
        y_pred,
        class_labels=model['classes'],
        title=f"Classification Confusion Matrix (Acc={accuracy:.2%})"
    )
