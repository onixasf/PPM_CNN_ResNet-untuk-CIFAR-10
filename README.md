# ResNet-50 on CIFAR-10 Classification

This project implements a deep learning model using **ResNet-50** architecture to classify images from the **CIFAR-10** dataset. The model is trained and evaluated using PyTorch and Google Colab, with exploratory analysis and visualizations to support understanding of the data and training performance.

## 📁 Contents

- `ResNet_50_CIFAR_10.ipynb`: Main notebook containing all steps from data loading, preprocessing, model setup, training, evaluation, and visualization.

## 🚀 Objective

To build and evaluate an image classifier using **ResNet-50**, a deep convolutional neural network, trained on the CIFAR-10 dataset which contains 10 classes of 32x32 images.

## 📦 Dataset

- **CIFAR-10**: Contains 60,000 32x32 color images in 10 classes, with 6,000 images per class.
- Automatically loaded from `torchvision.datasets`.

## 🔧 Features & Workflow

1. **Mount Google Drive**  
   Mounts Google Drive to store and load datasets or models.

2. **Library Installation**  
   Installs required libraries like `seaborn`, `torchvision`, and others.

3. **Data Preprocessing**  
   - Normalize image data
   - Apply transformations (e.g., resize, normalization)

4. **Model Definition**  
   - Load pre-trained `ResNet-50` model
   - Modify the final fully connected layer to match CIFAR-10 classes (10 outputs)

5. **Training**  
   - Uses Cross Entropy Loss and Adam optimizer
   - Runs for several epochs with training loss and accuracy tracking

6. **Evaluation**  
   - Calculates accuracy on test data
   - Visualizes confusion matrix
   - Shows correctly and incorrectly classified images

7. **Model Saving**  
   - Saves trained model to Google Drive for future use

## 📊 Visualization

- Training & Validation Accuracy and Loss curves
- Confusion Matrix using `seaborn`
- Sample predictions with actual vs predicted labels

## 🛠️ Technologies Used

- Python
- PyTorch
- Google Colab
- Matplotlib & Seaborn

## 📈 Results

- The model achieves decent accuracy on test data with visible improvement through epochs.
- Insights into class-wise performance using the confusion matrix.

## 📚 How to Use

1. Open `ResNet_50_CIFAR_10.ipynb` in Google Colab
2. Mount your Google Drive
3. Run all cells sequentially
4. View the performance and download the model if needed
