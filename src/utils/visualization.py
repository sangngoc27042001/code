import os
import json
import matplotlib.pyplot as plt
import numpy as np

def visualize_training_history(history, y_true_train, y_pred_train, y_true_val, y_pred_val, result_name="results"):
    """
    Plots training history, saves plots as PDFs, and stores max accuracy in a JSON file.
    
    Args:
        history (keras.callbacks.History): The history object returned from model.fit().
        y_true_train (np.ndarray): True labels for the training set.
        y_pred_train (np.ndarray): Predicted labels for the training set.
        y_true_val (np.ndarray): True labels for the validation set.
        y_pred_val (np.ndarray): Predicted labels for the validation set.
    """
    print("\nPlotting training history and saving results...")
    
    # --- Create a unique run folder ---
    runs_dir = "runs"
    os.makedirs(runs_dir, exist_ok=True)
    index = len(os.listdir(runs_dir)) + 1
    folder_name = f"{result_name}_{index:02d}"
    run_path = os.path.join(runs_dir, folder_name)
    os.makedirs(run_path, exist_ok=True)
    
    # --- Save Plots as PDFs ---
    
    # Plot and save Loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_path, 'loss.pdf'))
    plt.close() # Close the plot to free up memory
    
    # Plot and save Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['exact_match_accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_exact_match_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_path, 'accuracy.pdf'))
    plt.close()
    
    # --- Save Confusion Matrices as PDFs ---
    
    # You'll need to define a function to plot the confusion matrix.
    # We can create a simple one here. You'll also need scikit-learn.
    # The `y_true` and `y_pred` arguments must be passed to this function.
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    def plot_confusion_matrix(y_true, y_pred, title, filename):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        disp.ax_.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(run_path, filename))
        plt.close()

    # Plot and save Training Confusion Matrix
    plot_confusion_matrix(y_true_train, y_pred_train, 'Training Confusion Matrix', 'confusion_matrix_train.pdf')
    
    # Plot and save Validation Confusion Matrix
    plot_confusion_matrix(y_true_val, y_pred_val, 'Validation Confusion Matrix', 'confusion_matrix_val.pdf')
    
    # --- Save Max Accuracy to JSON ---
    max_train_accuracy = max(history.history['exact_match_accuracy'])
    max_val_accuracy = max(history.history['val_exact_match_accuracy'])
    
    accuracy_data = {
        'max_training_accuracy': max_train_accuracy,
        'max_validation_accuracy': max_val_accuracy
    }
    
    json_path = os.path.join(run_path, 'max_accuracy.json')
    with open(json_path, 'w') as f:
        json.dump(accuracy_data, f, indent=4)
    
    json_path = os.path.join(run_path, 'history.json')
    with open(json_path, 'w') as f:
        json.dump(history.history, f, indent=4)
        
    print(f"Results saved in folder: '{run_path}'")