import tensorflow as tf
from tensorflow.keras import layers, models, datasets
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Loading and Preprocessing ---
print("Loading and preprocessing CIFAR-10 dataset...")

# Load the CIFAR-10 dataset
# The dataset is split into training and testing sets
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
# Image pixel values typically range from 0 to 255.
# Normalizing helps the neural network learn more effectively.
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# You can optionally visualize some images to understand the data
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels are arrays, so we need to flatten them to get the single class index
    plt.xlabel(class_names[train_labels[i][0]])
plt.suptitle("Sample CIFAR-10 Images")
plt.show()

# --- 2. Build the CNN Model ---
print("\nBuilding the CNN model architecture...")

# Create a Sequential model, which builds a linear stack of layers
model = models.Sequential()

# Add the first Convolutional layer (Conv2D)
# 32 filters, 3x3 kernel size (filter size)
# activation='relu' (Rectified Linear Unit) is a common activation function
# input_shape specifies the dimensions of the input images (32x32 pixels, 3 color channels - RGB)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
# Add a MaxPooling2D layer to reduce the spatial dimensions
# (2, 2) pool size means it takes the maximum value over a 2x2 window
model.add(layers.MaxPooling2D((2, 2)))

# Add a second set of Conv2D and MaxPooling2D layers
# Increase the number of filters to 64, as typically deeper layers learn more complex features
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Add a third set of Conv2D layers
# This layer doesn't have a MaxPooling layer immediately after it, which is also a common pattern
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# --- 3. Add Dense Layers (Classifier Head) ---
# After the convolutional and pooling layers, the data needs to be flattened
# into a 1D array to be fed into standard dense (fully connected) layers.
model.add(layers.Flatten())

# Add a Dense layer with 64 units
model.add(layers.Dense(64, activation='relu'))

# Add the output Dense layer
# 10 units for 10 classes in CIFAR-10
# activation='softmax' ensures that the output is a probability distribution over the classes (sums to 1)
model.add(layers.Dense(10, activation='softmax'))

# Display the model summary, showing the layers, output shapes, and number of parameters
model.summary()

# --- 4. Compile the Model ---
print("\nCompiling the model...")

# Configure the model for training
# optimizer='adam' is a popular optimization algorithm
# loss='sparse_categorical_crossentropy' is used because our labels are integers (0-9)
# If labels were one-hot encoded (e.g., [0,0,1,0,...]), we would use 'categorical_crossentropy'
# metrics specifies what to monitor during training, typically accuracy
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# --- 5. Train the Model ---
print("\nTraining the model (this may take a few minutes)...")

# Train the model
# epochs: Number of times to iterate over the entire training dataset
# validation_data: Data on which to evaluate the loss and any model metrics at the end of each epoch
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# --- 6. Evaluate the Model ---
print("\nEvaluating the model on the test set...")

# Evaluate the model's performance on the test set
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# --- 7. Visualize Training History ---
print("\nVisualizing training history...")

plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True)

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- 8. Make Predictions (Optional) ---
print("\nMaking predictions on a few test images...")

# Select a few images from the test set for prediction
num_predictions_to_show = 5
sample_indices = np.random.choice(len(test_images), num_predictions_to_show, replace=False)

plt.figure(figsize=(10, 8))
for i, idx in enumerate(sample_indices):
    image_to_predict = test_images[idx]
    true_label = test_labels[idx][0] # Flatten label for class_names lookup

    # Add batch dimension for prediction (model expects a batch of images)
    input_image = np.expand_dims(image_to_predict, axis=0)
    predictions = model.predict(input_image)
    predicted_label = np.argmax(predictions[0]) # Get the index of the highest probability

    plt.subplot(1, num_predictions_to_show, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image_to_predict)
    color = 'green' if predicted_label == true_label else 'red'
    plt.xlabel(f"Pred: {class_names[predicted_label]}\nTrue: {class_names[true_label]}", color=color)
plt.suptitle("Sample Predictions (Green: Correct, Red: Incorrect)")
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent suptitle overlap
plt.show()

print("\nCNN model building, training, evaluation, and prediction complete!")
