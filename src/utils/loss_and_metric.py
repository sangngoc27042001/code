import tensorflow as tf

bce = tf.keras.losses.BinaryCrossentropy()

def abrupt_sigmoid(steepness=7.0):
    def return_function(x):
        # Implements a steeper sigmoid function.
        return 1 / (1 + tf.math.exp(-steepness * x))
    return return_function

def my_custom_loss(y_true, y_pred):
    # Calculate the binary cross-entropy loss for each sample
    return bce(y_true, y_pred)

def exact_match_accuracy(y_true, y_pred):
    # Round predictions to 0 or 1 based on a threshold (e.g., 0.5)
    # The `tf.round` function rounds to the nearest integer.
    y_pred_binarized = tf.round(y_pred)

    # Compare the binarized predictions with the ground truth
    # `tf.cast` converts boolean tensor to float tensor (True -> 1.0, False -> 0.0)
    matches = tf.cast(tf.equal(y_true, y_pred_binarized), tf.float32)

    # Sum up the number of matching labels for each sample
    # `tf.reduce_sum` with `axis=1` sums across the labels for each sample in the batch
    correct_labels_per_sample = tf.reduce_sum(matches, axis=1)

    # The `tf.shape(y_true)[1]` gives the number of labels for each sample
    num_labels = tf.cast(tf.shape(y_true)[1], tf.float32)

    # Check if all labels for a sample are correct
    all_labels_correct = tf.equal(correct_labels_per_sample, num_labels)

    # Calculate the mean of correct samples in the batch
    # `tf.reduce_mean` calculates the average over the samples
    return tf.reduce_mean(tf.cast(all_labels_correct, tf.float32))