# 📄 Rock-Paper-Scissors (RPS) Image Classification with Custom CNN

An end-to-end Computer Vision pipeline built with **TensorFlow / Keras** to classify images into three distinct categories: **Rock**, **Paper**, and **Scissors**.

This repository contains a complete deep learning workflow, including automated dataset extraction, robust data augmentation, custom class weighting to handle inter-class bias, label smoothing loss, dynamic learning rate adjustments, model checkpoints, and final evaluation on an independent test set.

---

## 📌 Key Features

* **Automated Data Handling**: Automatically detects and extracts training and test archives (`.zip`), handling nested directory structures gracefully.
* **Data Augmentation Pipeline**: Leverages `ImageDataGenerator` with rotation, translation, zoom, and horizontal flipping to enhance generalization.
* **Class Weighting Strategy**: Implements custom class weights during training to correct model bias towards overrepresented or overly dominant classes (e.g., preventing *Rock* from becoming a fallback class).
* **Custom CNN Architecture**: Features 4 convolutional blocks with L2 regularization, Max Pooling, dense feature extraction, and Dropout for regularization.
* **Advanced Training Setup**:
  * **Label Smoothing**: Reduces overconfidence in classification targets.
  * **Callbacks**: `ModelCheckpoint` (saving top weights), `ReduceLROnPlateau` (dynamic LR adjustment), and `EarlyStopping`.
* **Complete Evaluation & Visualization**: Outputs test metrics, classification reports, confusion matrices, and exports dual-panel loss/accuracy plots (`training_performance.png`).

---

## 🛠️ Tech Stack & Requirements

* **Python**: `3.10+`
* **TensorFlow / Keras**: `2.x`
* **Scikit-Learn**: For metrics (`classification_report`, `confusion_matrix`)
* **NumPy**: Numerical operations
* **Matplotlib**: Performance plotting

- Link for download -

TRAIN_AND_VAL_URL = "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip"
TEST_URL = "https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip"

### Installation

Clone the repository and install required dependencies:

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
pip install tensorflow scikit-learn numpy matplotlib

## 📂 Project Structure

.
├── progetto_paper_rock_scissor.ipynb   # Main notebook: data extraction, CNN model training, and evaluation
├── test_con_immagini_esterne.ipynb     # Notebook for inference and testing on custom external images
├── README.md                           # Detailed project documentation
├── training_performance.png            # Saved Loss & Accuracy evaluation curves
├── .gitignore                          # Git exclusion rules for large files
└── .gitattributes                      # Git configuration

- CNN Architecture OverviewThe -

network follows a sequential 4-block Convolutional architecture: 
Layer (type)            Output Shape             Details / Regularization
Input                   (150, 150, 3)            RGB Images
Conv2D(32, 3x3)         (150, 150, 32)           ReLU, padding='same'
MaxPooling2D            (75, 75, 32)             (2, 2)
Conv2D(64, 3x3)         (75, 75, 64)             ReLU, L2(0.001)
MaxPooling2D            (37, 37, 64)             (2, 2)
Conv2D(128, 3x3)        (37, 37, 128)            ReLU, padding='same'
MaxPooling2D            (18, 18, 128)            (2, 2)
Conv2D(256, 3x3)        (18, 18, 256)            ReLU, padding='same'
MaxPooling2D            (9, 9, 256)              (2, 2)
Flatten                 (20736)                  -
Dense(512)              (512)                    ReLU
Dropout(0.5)            (512)                    50% Dropout
Dense(3 - Output)       (3)                      Softmax


- Training Strategy & Hyperparameters -

Preprocessing & Augmentation:
Rescaling: 1/255
Random Rotation Range: 30°
Width / Height Shift: 15%
Zoom Range: 20%
Horizontal Flip: True
Validation Split: 15% of training data

Custom Class Weights:
To prevent the model from leaning heavily towards Rock as a high-confidence shortcut, specific loss weights were assigned:
Paper (0): 1.4 (Upweighted)
Rock (1): 0.8 (Downweighted)
Scissors (2): 1.0 (Standard)

Optimization & Loss:
Optimizer: Adam(learning_rate=3e-4)
Loss Function: CategoricalCrossentropy(label_smoothing=0.1)

Callbacks:
EarlyStopping: patience=3 monitoring val_loss.
ReduceLROnPlateau: Drops learning rate by 50% after 2 epochs of plateau on val_loss.
ModelCheckpoint: Saves top-performing weights based on val_accuracy to best_rps_model.keras.

- Training Performance & Visualizations -

The execution script automatically tracks training dynamics and exports dual-panel high-resolution plots (training_performance.png):
Loss Trend: Compares Categorical Crossentropy loss across training and validation splits.
Accuracy Trend: Tracks overall performance and checks for overfitting across epochs.


- Results & Evaluation -

The model is evaluated on the completely unseen rps-test-set:

Classification Metrics & Confusion Matrix
A full per-class evaluation is computed using scikit-learn:

--- CLASSIFICATION REPORT ---

              precision    recall  f1-score   support

       paper       1.00      0.94      0.97       124
        rock       0.94      1.00      0.97       124
    scissors       1.00      1.00      1.00       124

    accuracy                           0.98       372
   macro avg       0.98      0.98      0.98       372
weighted avg       0.98      0.98      0.98       372


--- CONFUSION MATRIX ---
[[116   8   0]
 [  0 124   0]
 [  0   0 124]]


- How to Run -

Place rps.zip and rps-test-set.zip in the root directory.

Execute the pipeline:

Bash
python main.py
The script will automatically unpack datasets, train the CNN, export training_performance.png, save the models (best_rps_model.keras and rps_custom_model.keras), and print evaluation metrics.