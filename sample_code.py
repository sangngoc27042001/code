import pennylane as qml
import tensorflow as tf
import numpy as np

# Configure TensorFlow to use float32 consistently
# tf.keras.backend.set_floatx('float32')

# Set up quantum device
n_qubits = 2  # Start with fewer qubits for stability
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf")
def quantum_circuit(inputs, weights):
    """Simple quantum circuit with proper dtype handling"""
    # Ensure inputs are float32
    inputs = tf.cast(inputs, tf.float32)
    weights = tf.cast(weights, tf.float32)
    
    # Data encoding
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    
    # Parameterized layer
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    
    # Entangling layer
    qml.CNOT(wires=[0, 1])
    
    # Return expectation values
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class SimpleQuantumLayer(tf.keras.layers.Layer):
    def __init__(self, n_qubits=2):
        super().__init__()
        self.n_qubits = n_qubits
        # Initialize quantum weights
        self.quantum_weights = self.add_weight(
            name='quantum_weights',
            shape=(n_qubits,),
            initializer='random_normal',
            trainable=True,
            dtype=tf.float32
        )
    
    def call(self, inputs):
        # Ensure inputs are float32
        inputs = tf.cast(inputs, tf.float32)
        
        # Process each sample in the batch
        def process_sample(sample):
            result = quantum_circuit(sample, self.quantum_weights)
            return tf.stack(result, axis=0)
        
        # Use vectorized_map for better performance
        outputs = tf.vectorized_map(process_sample, inputs)
        return tf.cast(outputs, tf.float32)

class QuantumNeuralNetwork(tf.keras.Model):
    def __init__(self, n_qubits=2):
        super().__init__()
        self.n_qubits = n_qubits
        
        # Classical preprocessing
        self.dense1 = tf.keras.layers.Dense(n_qubits, activation='tanh', dtype=tf.float32)
        
        # Quantum layer
        self.quantum_layer = SimpleQuantumLayer(n_qubits)
        
        # Classical postprocessing
        self.dense2 = tf.keras.layers.Dense(4, activation='relu', dtype=tf.float32)
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', dtype=tf.float32)
    
    def call(self, inputs):
        # Ensure inputs are float32
        inputs = tf.cast(inputs, tf.float32)
        
        # Classical preprocessing
        x = self.dense1(inputs)
        
        # Quantum processing
        x = self.quantum_layer(x)
        
        # Classical postprocessing
        x = self.dense2(x)
        return self.output_layer(x)

def create_sample_data():
    """Create sample binary classification data"""
    np.random.seed(42)
    X = np.random.randn(100, 4).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32).reshape(-1, 1)
    return X, y

def main():
    print("Creating quantum neural network...")
    
    # Create sample data
    X_train, y_train = create_sample_data()
    print(f"Data shapes: X={X_train.shape}, y={y_train.shape}")
    
    # Create model
    model = QuantumNeuralNetwork(n_qubits=2)
    
    # Compile with explicit dtype policy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Build the model by calling it once
    _ = model(X_train[:1])
    print("Model built successfully!")
    print(f"Model has {model.count_params()} trainable parameters")
    
    # Train the model
    print("\nTraining quantum neural network...")
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=5,
        validation_split=0.2,
        verbose=1
    )
    
    # Make predictions
    predictions = model.predict(X_train[:5])
    print(f"\nSample predictions: {predictions.flatten()}")
    print(f"Actual labels: {y_train[:5].flatten()}")
    
    # Test quantum circuit directly
    print(f"\nDirect quantum circuit test:")
    test_input = tf.constant([0.5, -0.3], dtype=tf.float32)
    test_weights = tf.constant([0.1, 0.2], dtype=tf.float32)
    result = quantum_circuit(test_input, test_weights)
    print(f"Quantum circuit output: {result}")

if __name__ == "__main__":
    main()