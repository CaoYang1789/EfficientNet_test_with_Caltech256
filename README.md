# EfficientNet Training with Caltech256 Dataset

This project demonstrates the process of downloading, preparing, and training various versions of EfficientNet on the **Caltech256** dataset. The process involves setting up the environment, downloading the dataset, training multiple EfficientNet models, and recording their performance.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Download the Caltech256 Dataset](#download-the-caltech256-dataset)
3. [Training EfficientNet Models](#training-efficientnet-models)
4. [Saving and Visualizing Results](#saving-and-visualizing-results)
5. [SSH and File Transfer Setup](#ssh-and-file-transfer-setup)
6. [Final Output](#final-output)

## Environment Setup

### 1. Verifying TensorFlow Installation

Before proceeding, verify that TensorFlow is correctly installed in your environment.

```bash
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

The expected output should show the installed TensorFlow version (e.g., `2.12.0`).

### 2. Installing Necessary Packages

Ensure all necessary packages like `kaggle`, `tensorflow`, and `pandas` are installed. If not, install them using `pip`.

```bash
pip install tensorflow keras pandas kaggle
```

## Download the Caltech256 Dataset

We will use the **Kaggle** API to download the **Caltech256** dataset.

### 1. Get Your Kaggle API Key

1. Go to your [Kaggle Account Settings](https://www.kaggle.com/account).
2. Scroll down to the "API" section and click **Create New API Token**.
3. A `kaggle.json` file will be downloaded. Place this file in the appropriate directory.

### 2. Move the API Key to the Right Location

```bash
mkdir -p ~/.kaggle
mv ~/path_to/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download the Dataset

Run the following command to download the Caltech-256 dataset:

```bash
kaggle datasets download -d jessicali9530/caltech256
```

### 4. Extract the Dataset

```bash
unzip caltech256.zip -d ./caltech256
```

## Training EfficientNet Models

### Python Script for Training

Below is the Python script used to train multiple EfficientNet models (B0, B1, B3, B5, B7) on the Caltech-256 dataset. The script loads the dataset, trains each model, and records the training and validation accuracy and loss for each epoch.

```python
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB3, EfficientNetB5, EfficientNetB7
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import preprocess_input
import pandas as pd

# Load and preprocess dataset
def load_datasets():
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        '256_ObjectCategories',
        label_mode='categorical',
        image_size=(224, 224),
        batch_size=32,
        validation_split=0.2,
        subset='training',
        seed=123
    )

    val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        '256_ObjectCategories',
        label_mode='categorical',
        image_size=(224, 224),
        batch_size=32,
        validation_split=0.2,
        subset='validation',
        seed=123
    )

    train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
    val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x), y))

    return train_dataset, val_dataset

# Train multiple EfficientNet models and record results
def train_multiple_models(models_list, model_names, train_dataset, val_dataset, epochs=5):
    results = []
    for model_class, model_name in zip(models_list, model_names):
        model = model_class(weights=None, input_shape=(224, 224, 3), classes=257)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
        
        # Record the accuracy and loss for each epoch
        for epoch in range(epochs):
            results.append({
                'model': model_name,
                'epoch': epoch + 1,
                'accuracy': history.history['accuracy'][epoch],
                'val_accuracy': history.history['val_accuracy'][epoch],
                'loss': history.history['loss'][epoch],
                'val_loss': history.history['val_loss'][epoch]
            })
    return results

# Main
if __name__ == "__main__":
    model_classes = [EfficientNetB0, EfficientNetB1, EfficientNetB3, EfficientNetB5, EfficientNetB7]
    model_names = ["EfficientNetB0", "EfficientNetB1", "EfficientNetB3", "EfficientNetB5", "EfficientNetB7"]

    train_dataset, val_dataset = load_datasets()

    # Train the models and collect the results
    results = train_multiple_models(model_classes, model_names, train_dataset, val_dataset, epochs=5)

    # Convert the results to a DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv("training_results.csv", index=False)
    print("Results saved to training_results.csv")
```

## Saving and Visualizing Results

All the training and validation results are saved in a CSV file named `training_results.csv`.

To visualize the results (accuracy, loss) across different EfficientNet models and epochs, you can read the CSV and plot the data using Python libraries like `matplotlib` or `seaborn`.

Example:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
data = pd.read_csv('training_results.csv')

# Get all the different model names
models = data['Model'].unique()

# Plot the figures
plt.figure(figsize=(12, 5))

# Plot the accuracy curves
plt.subplot(1, 2, 1)
for model in models:
    model_data = data[data['Model'] == model]
    plt.plot(model_data['Epoch'], model_data['Train Accuracy'], label=f'{model} - Train', marker='o')
    plt.plot(model_data['Epoch'], model_data['Validation Accuracy'], label=f'{model} - Validation', marker='o')

plt.title('Training and Validation Accuracy for Different Models')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot the loss curves
plt.subplot(1, 2, 2)
for model in models:
    model_data = data[data['Model'] == model]
    plt.plot(model_data['Epoch'], model_data['Train Loss'], label=f'{model} - Train', marker='o')
    plt.plot(model_data['Epoch'], model_data['Validation Loss'], label=f'{model} - Validation', marker='o')

plt.title('Training and Validation Loss for Different Models')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()
```


## Final Output
![image](https://github.com/user-attachments/assets/be319460-bcc8-42d3-bbc7-497bad948a00)

## 1. Training and Validation Accuracy

The left figure shows the trend of training and validation accuracy across different EfficientNet models as the number of epochs increases:

- **Training Accuracy**: As can be seen, the training accuracy of all models improves with the increase of epochs. The EfficientNetB0 model starts with an accuracy of around 0.72, gradually increasing to about 0.96, while more complex models like EfficientNetB7 show similar improvement, starting slightly lower but eventually reaching near 0.97 accuracy.
  
- **Validation Accuracy**: The validation accuracy for all models starts relatively high and rapidly improves after the first epoch. Most models’ validation accuracy tends to stabilize between 0.85 and 0.88, indicating good performance after training. However, some models, like EfficientNetB0, have a validation accuracy slightly lower than their training accuracy, which suggests slight overfitting.

## 2. Training and Validation Loss

The right figure shows the changes in training and validation loss for different EfficientNet models:

- **Training Loss**: The training loss is generally high during the first epoch, with the initial loss for EfficientNetB0 around 1.5, while other models start with an initial loss of around 1.4. This indicates that the prediction error is quite large at the beginning. As training progresses, the loss rapidly decreases and tends to stabilize, eventually converging between 0.4 and 0.6.

- **Validation Loss**: The validation loss follows a similar trend as the training loss. After the first epoch, the loss drops significantly and then stabilizes. The validation loss for EfficientNetB0 is slightly higher than for the other models. It is worth noting that the validation loss is close to the training loss, indicating good generalization ability of the model, with no significant overfitting.

## 3. Why the initial loss is greater than 1
### Loss Calculation Code Example
```python
# Calculate loss, which includes softmax cross entropy and L2 regularization.
cross_entropy = tf.losses.softmax_cross_entropy(
    logits=logits,
    onehot_labels=labels,
    label_smoothing=FLAGS.label_smoothing)

# Add weight decay to the loss for non-batch-normalization variables.
loss = cross_entropy + FLAGS.weight_decay * tf.add_n(
    [tf.nn.l2_loss(v) for v in tf.trainable_variables()
     if 'batch_normalization' not in v.name])

The loss function calculation uses Softmax cross-entropy loss combined with L2 regularization. Now we will use these two to answer the question.
```

### How Softmax Cross-Entropy Works:

- **logits** are the unnormalized outputs of the model.
- **onehot_labels** are the target labels in the one-hot encoding format.
- **label_smoothing** is a hyperparameter for label smoothing, which is used to reduce the model's overconfidence in predictions and prevent the output probabilities from being too concentrated on a single class.

### The Cross-Entropy Loss Calculation Formula:

\[
\text{Loss} = - \sum_i y_i \log(\hat{y}_i)
\]

Where:

- \( y_i \) is the true class label (one-hot encoded).
- \( \hat{y}_i \) is the predicted probability computed by Softmax.

When the model’s predictions are inaccurate, particularly when the predicted probability is close to 0, \( \log(\hat{y}_i) \) becomes very large (approaching negative infinity), resulting in a large cross-entropy loss. This is why the first **loss** during training can be greater than 1.

### Key Points:

- **High Initial Loss**: Since the model's predictions are inaccurate during early training (for example, misclassifications, and the output probability distribution deviates from the true distribution), the cross-entropy loss is very large, and it is normal for the loss value to be greater than 1.
- **Loss Decreases with Training**: As training progresses, the model’s predicted probabilities become closer to the true labels, and the cross-entropy loss gradually decreases.

### How L2 Regularization Works:

L2 regularization is a way to prevent overfitting by adding a penalty term based on the square of the parameters. It adds a weight decay term to the loss function:

\[
\text{L2 Loss} = \lambda \sum_i w_i^2
\]

Where:

- \( w_i \) are the model's trainable parameters (weights).
- \( \lambda \) is the regularization coefficient (i.e., **FLAGS.weight_decay**), which controls the weight of the regularization term.

### Key Points:

- **Regularization Increases Loss**: The L2 regularization term imposes an additional penalty on the trainable parameters of non-Batch Normalization layers. Therefore, the total **loss** is the sum of the **cross_entropy** and the L2 regularization term. Although this regularization usually accounts for only a small portion of the total loss, it still affects the final loss value.

---

### Answer: Why the First **loss** is Greater Than 1

- **Characteristics of Cross-Entropy Loss**: Cross-entropy loss is very large when the model predictions are inaccurate, especially in the early stages of training. The predicted probabilities for the target class can be much lower than the true value, so the loss function value will be greater than 1.
- **L2 Regularization**: The L2 regularization term in the loss function increases the overall **loss**, which further explains why the **loss** can be higher than just the **cross_entropy**.

Therefore, it is normal for the initial **loss** to be greater than 1. As training progresses and the model learns the data distribution, the loss value will gradually decrease, converging towards a lower value.
