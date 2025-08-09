import matplotlib.pyplot as plt

def plot_history(history):
    """
    input:
    {
        "epoch": [...],
        "bleu": [...],
        "rouge": [...],
        "bert_f1": [...],
        "val_loss": [...],
        "meteor": [...]
    }
    """
    epochs = history["epoch"]

    # ===== plot Validation Loss  =====
    plt.figure(figsize=(12, 5))
    plt.plot(epochs, history["val_loss"], marker='o', label='Validation Loss', color='red')
    for i, v in zip(epochs, history["val_loss"]):
        plt.text(i, v, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Validation Loss over Epochs')
    plt.grid(True)
    plt.legend()
    plt.show()

    # ===== plot BLEU、ROUGE-L、BERTScore-F1、Meteor =====
    plt.figure(figsize=(12, 6))
    for key, label in [("bleu", "BLEU"), ("rouge", "ROUGE-L"), ("bert_f1", "BERTScore-F1"), ("meteor", "Meteor")]:
        if key in history:
            plt.plot(epochs, history[key], marker='o', label=label)
            # tag the max value
            max_idx = max(range(len(history[key])), key=lambda i: history[key][i])
            plt.text(epochs[max_idx], history[key][max_idx], f"{history[key][max_idx]:.3f}",
                     ha='center', va='bottom', fontsize=8, color='black')

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()
