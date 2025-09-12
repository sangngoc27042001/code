import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.utils.hybrid_cnn_quantum_model import create_hybrid_model
from src.utils.preprocessing_cnn_model import test_cnn_extractor
import os

def convert_labels_to_binary_array(labels, n_classes=10):
    """Convert integer labels to binary array representation."""
    n_readout_qubits = int(np.ceil(np.log2(n_classes)))
    binary_array = np.zeros((len(labels), n_readout_qubits), dtype=np.float32)
    
    for i, label in enumerate(labels):
        binary_str = format(label, f'0{n_readout_qubits}b')
        binary_array[i] = [int(bit) for bit in binary_str]
    
    return binary_array

def load_and_preprocess_mnist(total_size=1000, val_split=0.2):
    """Load and preprocess MNIST dataset with balanced smaller subset for train and validation only."""
    from sklearn.model_selection import train_test_split
    
    print("Loading MNIST dataset...")
    
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Create smaller balanced subset using stratified sampling
    print(f"Creating balanced subset: {total_size} total samples")
    
    # Use stratified sampling to maintain class balance
    x_subset, _, y_subset, _ = train_test_split(
        x_train, y_train, 
        train_size=total_size, 
        stratify=y_train, 
        random_state=42
    )
    
    # Split into train and validation
    x_train_small, x_val_small, y_train_small, y_val_small = train_test_split(
        x_subset, y_subset,
        test_size=val_split,
        stratify=y_subset,
        random_state=42
    )
    
    # Normalize pixel values to [0, 1]
    x_train_small = x_train_small.astype('float32') / 255.0
    x_val_small = x_val_small.astype('float32') / 255.0
    
    # Add channel dimension (28, 28) -> (28, 28, 1)
    x_train_small = np.expand_dims(x_train_small, axis=-1)
    x_val_small = np.expand_dims(x_val_small, axis=-1)

    # preprocess labels to binary array:
    y_train_binary = convert_labels_to_binary_array(y_train_small, n_classes=10)
    y_val_binary = convert_labels_to_binary_array(y_val_small, n_classes=10)
    
    # Print class distribution to verify balance
    unique_train, counts_train = np.unique(y_train_small, return_counts=True)
    unique_val, counts_val = np.unique(y_val_small, return_counts=True)
    
    print(f"Training data shape: {x_train_small.shape}")
    print(f"Training labels shape: {y_train_binary.shape}")
    print(f"Validation data shape: {x_val_small.shape}")
    print(f"Validation labels shape: {y_val_binary.shape}")
    print(f"Label range: {y_train_binary.min()} to {y_train_binary.max()}")
    print(f"Training class distribution: {dict(zip(unique_train, counts_train))}")
    print(f"Validation class distribution: {dict(zip(unique_val, counts_val))}")
    
    return (x_train_small, y_train_binary), (x_val_small, y_val_binary)

def evaluate_model(model, x_test, y_test):
    """Evaluate the trained model."""
    print("\nEvaluating model...")
    
    # Get predictions
    predictions = model.predict(x_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Show some predictions
    print("\nSample predictions:")
    for i in range(min(10, len(y_test))):
        pred_class = predicted_classes[i]
        true_class = y_test[i]
        confidence = predictions[i][pred_class]
        print(f"Sample {i}: Predicted={pred_class}, True={true_class}, Confidence={confidence:.3f}")
    
    return accuracy, predictions


def visualize_training_history(history):
    """Plot training history."""
    print("\nPlotting training history...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history.history['exact_match_accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_exact_match_accuracy'], label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs('runs', exist_ok=True)
    index = len(os.listdir('runs')) + 1
    index = f"{index}".zfill(2)
    plt.savefig(f'runs/training_history_{index}.png', dpi=150, bbox_inches='tight')
    print("Training history saved as 'training_history.png'")


def test_components():
    """Test individual components before full training."""
    print("="*60)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("="*60)
    
    # Test CNN extractor
    print("\n1. Testing CNN Feature Extractor:")
    cnn_extractor, cnn_model = test_cnn_extractor()
    
    print("\n2. Testing Hybrid Model:")
    from src.utils.hybrid_cnn_quantum_model import test_hybrid_model
    hybrid_model, history = test_hybrid_model()
    
    return cnn_extractor, hybrid_model


def main():
    """Main training pipeline."""
    print("="*60)
    print("HYBRID CNN-QUANTUM MODEL TRAINING")
    print("="*60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Test components first
    print("\nStep 1: Testing Components")
    # test_components()
    
    # Load data with balanced train/validation split
    print("\nStep 2: Loading Data")
    (x_train, y_train), (x_val, y_val) = load_and_preprocess_mnist(total_size=1000, val_split=0.2)
    
    print(f"Final dataset sizes:")
    print(f"  Training: {len(x_train)}")
    print(f"  Validation: {len(x_val)}")
    
    # Create and train model
    print("\nStep 3: Creating Hybrid Model")
    model = create_hybrid_model(n_classes=10, n_layers=1)
    
    # Build model
    _ = model(x_train[:1])
    print(f"Model built with {model.count_params()} parameters")
    
    # Train model
    print("\nStep 4: Training Model")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=100,
        batch_size=16,
        verbose=1
    )
    
    # Get final validation accuracy
    final_val_accuracy = max(history.history['val_exact_match_accuracy'])
    
    # Visualize results
    print("\nStep 5: Visualizing Results")
    visualize_training_history(history)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")
    print("="*60)
    
    return model, history, final_val_accuracy


if __name__ == "__main__":
    model, history, accuracy = main()
