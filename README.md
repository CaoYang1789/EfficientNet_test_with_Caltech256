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
import seaborn as sns
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv("training_results.csv")

# Plot accuracy
sns.lineplot(data=df, x="epoch", y="accuracy", hue="model")
plt.title("Training Accuracy across EfficientNet Models")
plt.show()

# Plot validation accuracy
sns.lineplot(data=df, x="epoch", y="val_accuracy", hue="model")
plt.title("Validation Accuracy across EfficientNet Models")
plt.show()
```

## SSH and File Transfer Setup

If you need to transfer files from your virtual machine to your local machine, follow the steps below:

1. **Generate SSH keys** (if not done already):
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

2. **Copy the public key to your VM**:
   ```bash
   ssh-copy-id yc4535@your_vm_ip_address
   ```

3. **Transfer files using scp**:
   ```bash
   scp -i ~/.ssh/id_ed25519 yc4535@your_vm_ip_address:/home/yc4535/test1/training_results.csv "C:/local_directory/"
   ```

## Final Output

After completing the training, the final results (accuracy, loss, validation accuracy, validation loss) are saved in the `training_results.csv` file for further analysis.

