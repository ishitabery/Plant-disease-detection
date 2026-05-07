# Plant Disease Recognition using Transfer Learning

## Project Overview
This project demonstrates how to build and train a deep learning model to classify plant diseases from leaf images. It leverages transfer learning with a pre-trained EfficientNetB4 model, fine-tuned on a custom dataset of plant leaf images.

## Dataset
The dataset used in this project is sourced from Mendeley Data, specifically the "Plant leave diseases dataset with augmentation". It contains a wide variety of plant leaf images, categorized by plant type and disease (or healthy status).

**Data Source**: `https://data.mendeley.com/datasets/tywbtsjrjv/1`

### Data Preparation
1.  **Download**: The dataset (a `.zip` file) is downloaded directly into the Colab environment.
2.  **Extraction**: The zip file is extracted, creating a directory structure of `Plant_leave_diseases_dataset_with_augmentation` containing subdirectories for each plant disease class.
3.  **Splitting**: The `split-folders` library is used to divide the dataset into training, validation, and testing sets with a ratio of 80% train, 10% validation, and 10% test.
4.  **Loading**: TensorFlow's `image_dataset_from_directory` is used to load images, automatically handling labels based on directory names. Images are resized to 160x160 pixels and batched into groups of 32.
5.  **Preprocessing**: Image data is preprocessed using `tf.keras.applications.efficientnet.preprocess_input` which scales pixel values to the range [-1, 1], suitable for the EfficientNet model.

## Model Architecture
The model is built using transfer learning with a pre-trained EfficientNetB4 convolutional base from `tf.keras.applications`.

1.  **Base Model**: `EfficientNetB4` (pre-trained on ImageNet) is used without its top (classification) layer (`include_top=False`). The convolutional base is initially frozen (`base_model.trainable = False`) to allow for quick training of the new classification head.
2.  **Classification Head**: A custom classification head is added on top of the base model:
    *   `GlobalAveragePooling2D`: Reduces the spatial dimensions of the feature maps.
    *   `Dropout(0.2)`: A dropout layer for regularization.
    *   `Dense` layer: A final dense layer with `sigmoid` activation, matching the number of classes in the dataset.

## Training
The training process involves two main phases:

1.  **Feature Extraction (Initial Training)**:
    *   The base model's layers are frozen, and only the newly added classification head is trained.
    *   **Optimizer**: `tf.keras.optimizers.Adam()`
    *   **Loss Function**: `tf.keras.losses.SparseCategoricalCrossentropy()`
    *   **Metrics**: `tf.keras.metrics.SparseCategoricalAccuracy()`
    *   **Epochs**: Trained for `initial_epochs` (e.g., 6 epochs).
    *   This phase helps the new classification head learn to interpret the features extracted by the pre-trained EfficientNetB4.

2.  **Fine-tuning**: 
    *   After initial training, some of the higher-level layers of the base model are unfrozen (`base_model.trainable = True`), allowing their weights to be adjusted during further training.
    *   `fine_tune_at = 100`: Layers before this index are kept frozen, while layers from this index onwards are made trainable.
    *   **Optimizer**: `tf.keras.optimizers.Adam()` is re-compiled for a potentially lower learning rate (though not explicitly set lower in this example, it's good practice).
    *   **Epochs**: Trained for an additional `fine_tune_epochs` (e.g., 10 epochs), continuing from the initial epochs.
    *   Fine-tuning allows the model to adapt the pre-trained features more specifically to the target dataset.

## Results
The model's performance is evaluated on the test dataset after both training phases.

*   **Initial Training Accuracy (Validation)**: Approximately 94.80% after 6 epochs.
*   **Fine-tuning Accuracy (Validation)**: The validation accuracy fluctuates during fine-tuning, reaching peaks (e.g., 97.91%), but it's important to monitor for overfitting.
*   **Final Test Accuracy**: The model achieved a test accuracy of approximately 91.05%.

Training and validation accuracy/loss plots are generated to visualize the model's learning progress throughout the epochs.
