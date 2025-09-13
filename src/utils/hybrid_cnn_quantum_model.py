import pennylane as qml
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from src.utils.preprocessing_cnn_model import CNNFeatureExtractor
from src.utils.loss_and_metric import my_custom_loss, exact_match_accuracy, abrupt_sigmoid


class HybridCNNQuantumModel(tf.keras.Model):
    """
    Hybrid CNN-Quantum model for image classification.
    
    Architecture:
    1. CNN Feature Extractor: 28x28x1 -> 4 scalars
    2. Quantum Circuit: 4 input qubits + 4 readout qubits
    3. Final Classifier: 4 quantum measurements -> n_classes
    """
    
    def __init__(self, n_classes=10, n_layers=1, **kwargs):
        super(HybridCNNQuantumModel, self).__init__(**kwargs)
        
        self.n_classes = n_classes
        self.n_input_qubits = 4  # Number of CNN output features
        self.n_readout_qubits = math.ceil(math.log2(n_classes))  # ceil(log2(10)) = 4
        self.n_total_qubits = self.n_input_qubits + self.n_readout_qubits
        self.n_layers = n_layers
        
        print(f"Quantum circuit setup:")
        print(f"  Input qubits: {self.n_input_qubits}")
        print(f"  Readout qubits: {self.n_readout_qubits}")
        print(f"  Total qubits: {self.n_total_qubits}")
        
        # CNN feature extractor
        self.cnn_extractor = CNNFeatureExtractor()
        
        # Quantum device
        self.dev = qml.device("default.qubit", wires=self.n_total_qubits)
        
        # Create quantum node
        self.quantum_node = qml.QNode(self._quantum_circuit, self.dev, interface="tf")
        
        # Quantum weights for controlled rotations
        # Shape: (n_layers, n_input_qubits, n_readout_qubits)
        self.quantum_weights = self.add_weight(
            name='quantum_weights',
            shape=(n_layers, self.n_input_qubits, self.n_readout_qubits),
            # initializer='random_normal',
            initializer='glorot_uniform',
            trainable=True,
            dtype=tf.float32
        )
        
        # # Final classifier
        self.classifier = tf.keras.layers.Dense(
            self.n_readout_qubits, activation=abrupt_sigmoid(), dtype=tf.float32, name='classifier'
        )
    
    def _quantum_circuit(self, inputs, weights):
        """
        Quantum circuit with angle embedding and controlled rotations.
        
        Args:
            inputs: 4 scalar values from CNN (input features)
            weights: Quantum circuit parameters
            
        Returns:
            List of expectation values from readout qubits
        """
        # Ensure proper dtypes
        inputs = tf.cast(inputs, tf.float32)
        weights = tf.cast(weights, tf.float32)

        # Apply Hadamard gates to all qubits to create superposition
        for i in range(self.n_total_qubits):
            qml.Hadamard(wires=i)
    
        # 1. Angle embedding on input qubits (0-3)
        for i in range(self.n_input_qubits):
            qml.RY(inputs[i], wires=i)
        
        # 2. Parameterized quantum circuit with controlled rotations
        for layer in range(self.n_layers):
            # Each input qubit controls rotations on each readout qubit
            for i in range(self.n_input_qubits):  # input qubits (control)
                for j in range(self.n_readout_qubits):  # readout qubits (target)
                    control_qubit = i
                    target_qubit = self.n_input_qubits + j
                    qml.CRY(weights[layer, i, j], wires=[control_qubit, target_qubit])
            
            # # Add some entanglement between readout qubits
            # if layer < self.n_layers - 1:  # Don't add on last layer
            #     for j in range(self.n_readout_qubits - 1):
            #         qml.CNOT(wires=[self.n_input_qubits + j, self.n_input_qubits + j + 1])
        
        # 3. Measure readout qubits (4-7)
        measurements = []
        for j in range(self.n_readout_qubits):
            measurements.append(qml.expval(qml.PauliZ(self.n_input_qubits + j)))
        
        return measurements
    
    def draw_circuit(self, save_path="quantum_circuit.png"):
        """Draw and save the quantum circuit visualization."""
        dev = qml.device("default.qubit", wires=self.n_total_qubits)
        
        @qml.qnode(dev)
        def circuit():
            # Sample inputs and weights for visualization
            inputs = [0.5, 0.3, 0.8, 0.2]
            weights = np.random.random((self.n_layers, self.n_input_qubits, self.n_readout_qubits))
            
            return self._quantum_circuit(inputs, weights)
        
        fig, ax = qml.draw_mpl(circuit)()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Circuit saved to {save_path}")
        return save_path
    
    def call(self, inputs, training=None):
        """Forward pass through the hybrid model."""
        # Ensure inputs are float32
        inputs = tf.cast(inputs, tf.float32)
        
        # 1. CNN feature extraction: (batch, 28, 28, 1) -> (batch, 4)
        cnn_features = self.cnn_extractor(inputs)
        
        # 2. Quantum processing: (batch, 4) -> (batch, n_readout_qubits)
        def process_sample(sample):
            """Process a single sample through quantum circuit."""
            result = self.quantum_node(sample, self.quantum_weights)
            return tf.stack(result, axis=0)
        
        # Process batch through quantum circuit
        quantum_outputs = tf.vectorized_map(process_sample, cnn_features)
        quantum_outputs = tf.cast(quantum_outputs, tf.float32)
        
        # 3. Final classification: (batch, n_readout_qubits) -> (batch, n_classes)
        predictions = abrupt_sigmoid()(quantum_outputs)
        # predictions = self.classifier(quantum_outputs)
        
        return predictions
    
    def get_config(self):
        """Return model configuration for serialization."""
        config = super(HybridCNNQuantumModel, self).get_config()
        config.update({
            'n_classes': self.n_classes,
            'n_layers': self.n_layers,
        })
        return config


def create_hybrid_model(n_classes=10, n_layers=1):
    """
    Create a hybrid CNN-quantum model.
    
    Args:
        n_classes: Number of output classes
        n_layers: Number of quantum circuit layers
        
    Returns:
        HybridCNNQuantumModel: Compiled hybrid model
    """
    model = HybridCNNQuantumModel(n_classes=n_classes, n_layers=n_layers)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=my_custom_loss,
        metrics=[exact_match_accuracy]
    )
    
    return model

def test_hybrid_model():
    """Test the hybrid CNN-quantum model with sample data."""
    print("Testing Hybrid CNN-Quantum Model...")
    
    # Create sample data (MNIST-like)
    batch_size = 32  # Increased batch size for better training
    sample_images = np.random.randn(batch_size, 28, 28, 1).astype(np.float32)
    sample_labels = np.random.randint(0, 2, size=(batch_size, 4)).astype(np.float32)  # Multi-label for 10 classes
    
    # Create validation data
    val_batch_size = 16
    val_images = np.random.randn(val_batch_size, 28, 28, 1).astype(np.float32)
    val_labels = np.random.randint(0, 2, size=(val_batch_size, 4)).astype(np.float32)
    
    # Create model
    model = create_hybrid_model(n_classes=10, n_layers=1)
    
    # Build model by calling it once
    predictions = model(sample_images[:2])  # Use smaller batch for initial test
    
    print(f"Input shape: {sample_images.shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions shape: {predictions.shape}")
    print(f"Sample predictions (first sample): {predictions[0].numpy()}")
    print(f"Prediction sum (should be ~1.0): {tf.reduce_sum(predictions[0]):.3f}")
    
    # Count parameters
    total_params = model.count_params()
    print(f"Total trainable parameters: {total_params}")
    
    # Test training with model.fit() for 10 epochs
    print("\n" + "="*60)
    print("TRAINING WITH MODEL.FIT() FOR 10 EPOCHS")
    print("="*60)

    history = model.fit(
        sample_images,
        sample_labels,
        epochs=1,
        validation_data=(val_images, val_labels)
    )
    
    print("Training completed!")
    
    # Display final training summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    final_train_loss = history.history['loss'][-1]
    final_train_acc = history.history['exact_match_accuracy'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_val_acc = history.history['val_exact_match_accuracy'][-1]
    
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    
    return model, history


if __name__ == "__main__":
    model, history = test_hybrid_model()
