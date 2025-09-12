import tensorflow as tf
import numpy as np


class CNNFeatureExtractor(tf.keras.layers.Layer):
    """
    CNN feature extractor that reduces 28x28x1 images to 4 scalars.
    
    Architecture:
    Input (28, 28, 1) -> Conv2D -> MaxPool -> Conv2D -> MaxPool -> Conv2D -> MaxPool -> Flatten -> Dense -> Dense(4)
    """
    
    def __init__(self, **kwargs):
        super(CNNFeatureExtractor, self).__init__(**kwargs)
        
        # Convolutional layers
        self.conv1 = tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu', padding='same', dtype=tf.float32
        )
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2), dtype=tf.float32)
        
        self.conv2 = tf.keras.layers.Conv2D(
            64, (3, 3), activation='relu', padding='same', dtype=tf.float32
        )
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2), dtype=tf.float32)
        
        self.conv3 = tf.keras.layers.Conv2D(
            128, (3, 3), activation='relu', padding='same', dtype=tf.float32
        )
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2), dtype=tf.float32)
        
        # Dense layers
        self.flatten = tf.keras.layers.Flatten(dtype=tf.float32)
        self.dense1 = tf.keras.layers.Dense(16, activation='relu', dtype=tf.float32)
        self.dense2 = tf.keras.layers.Dense(4, activation='tanh', dtype=tf.float32)  # tanh to bound outputs
        
    def call(self, inputs):
        """Forward pass through CNN feature extractor."""
        # Ensure inputs are float32
        x = tf.cast(inputs, tf.float32)
        
        # First conv block: (28, 28, 1) -> (14, 14, 32)
        x = self.conv1(x)
        x = self.pool1(x)
        
        # Second conv block: (14, 14, 32) -> (7, 7, 64)
        x = self.conv2(x)
        x = self.pool2(x)
        
        # Third conv block: (7, 7, 64) -> (3, 3, 128)
        x = self.conv3(x)
        x = self.pool3(x)
        
        # Dense layers: (3, 3, 128) -> (1152,) -> (16,) -> (4,)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return x
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super(CNNFeatureExtractor, self).get_config()
        return config


def create_cnn_model(input_shape=(28, 28, 1)):
    """
    Create a standalone CNN model for feature extraction.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        
    Returns:
        tf.keras.Model: CNN model that outputs 4 features
    """
    inputs = tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)
    features = CNNFeatureExtractor()(inputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=features, name='cnn_feature_extractor')
    return model


def test_cnn_extractor():
    """Test the CNN feature extractor with sample data."""
    print("Testing CNN Feature Extractor...")
    
    # Create sample data
    batch_size = 4
    sample_images = np.random.randn(batch_size, 28, 28, 1).astype(np.float32)
    
    # Create and test the extractor
    extractor = CNNFeatureExtractor()
    features = extractor(sample_images)
    
    print(f"Input shape: {sample_images.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Output range: [{tf.reduce_min(features):.3f}, {tf.reduce_max(features):.3f}]")
    print(f"Sample features: {features[0].numpy()}")
    
    # Test standalone model
    model = create_cnn_model()
    model_features = model(sample_images)
    
    print(f"Model output shape: {model_features.shape}")
    print(f"Model parameters: {model.count_params()}")
    
    return extractor, model


if __name__ == "__main__":
    test_cnn_extractor()
