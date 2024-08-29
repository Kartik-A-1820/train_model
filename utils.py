import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_metrics(history, save_dir):
    # Determine the correct accuracy key
    accuracy_key = 'binary_accuracy' if 'binary_accuracy' in history.history else 'accuracy'
    val_accuracy_key = f'val_{accuracy_key}'

    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history[accuracy_key])
    plt.plot(history.history[val_accuracy_key])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'))
    plt.close()

    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    plt.close()

def save_confusion_matrix(cm, save_dir):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()
