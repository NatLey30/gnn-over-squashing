import matplotlib.pyplot as plt
import os


def plot_training(history, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    epochs = range(len(history["train_loss"]))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # --- Loss plot ---
    axs[0].plot(epochs, history["train_loss"])
    axs[0].set_title("Training Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")

    # --- Accuracy plot ---
    axs[1].plot(epochs, history["train_acc"])
    axs[1].set_title("Training Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
