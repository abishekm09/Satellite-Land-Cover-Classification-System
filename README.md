# Satellite-Land-Cover-Classification-System
Satellite Image Classifier using ResNet50 and Transfer Learning. This model distinguishes between agricultural and non-agricultural land with good accuracy. Built with TensorFlow/Keras, it features automated data pipelines, a custom dense head for precision, and an inference script for real-time predictions.

📌 Project Overview
This project implements a deep learning pipeline to automate the identification of land use from satellite imagery. By leveraging a pre-trained model, the system achieves near-perfect accuracy with minimal training time.

🧠 Model Architecture
Base Model: ResNet50 (pre-trained on ImageNet) with frozen weights to preserve learned features.

Input Dimensions: Images are resized to 224x224 pixels with 3 color channels (RGB).

Custom Layers: Includes a Flatten layer, a Dense layer (128 units, ReLU activation), and a Dropout layer (20%) to prevent overfitting.

Final Layer: Softmax activation for multi-class probability output.

📊 Dataset & Training
Data Source: 6,000 satellite images categorized into class_0_non_agri and class_1_agri.

Optimization: Compiled with the Adam optimizer and Sparse Categorical Crossentropy loss.

Results:

Training Accuracy: 99.40%.

Validation Accuracy: 98.33%.

Training Time: 5 Epochs.

🛠️ Installation & Usage
Clone the repository:
Bash
git clone https://github.com/abishekm09/Satellite-Land-Cover-Classification-System

Install dependencies:
Bash
pip install tensorflow numpy matplotlib requests
Run the Notebook: Open trained_model.ipynb in Jupyter or Google Colab to execute the training and inference pipeline.

🔮 Inference
To test a new image, use the prediction block in the notebook. It converts the image to a numerical array, runs it through the model, and returns the predicted class with a confidence percentage.
