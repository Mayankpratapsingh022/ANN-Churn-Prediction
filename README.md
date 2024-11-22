# ANN-Based Customer Churn Prediction

This repository contains an Artificial Neural Network (ANN)-based solution for predicting customer churn using the **Churn_Modelling.csv** dataset. The project demonstrates the end-to-end implementation of a machine learning pipeline, including data preprocessing, model training, evaluation, and deployment through a Streamlit web app.

---

![image](https://github.com/user-attachments/assets/a742a76a-f51f-4584-bfbb-b49cdbeceb96)
[Click to check the deployed app](https://ann-churn-prediction-mayank.streamlit.app/)


## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Pipeline Steps](#pipeline-steps)
4. [Model Architecture](#model-architecture)
5. [Streamlit App](#streamlit-app)
6. [Dependencies](#dependencies)
7. [How to Run](#how-to-run)
8. [Results](#results)
9. [Folder Structure](#folder-structure)


---

## Project Overview

Customer churn prediction is a crucial problem in industries aiming to retain customers. This project leverages a deep learning approach to predict whether a customer will leave a service provider based on various features such as age, gender, credit score, and geography.

Key features of this project include:

- Data preprocessing with label and one-hot encoding.
- ANN model training and evaluation.
- Early stopping and TensorBoard for monitoring training.
- Deployment of the trained model using a Streamlit web app.

---

## Dataset Description

The dataset used is `Churn_Modelling.csv`, which contains 10,000 rows and 13 columns, including:

- **Independent Features:** Credit Score, Geography, Gender, Age, Tenure, Balance, etc.
- **Target Feature:** `Exited` (1 indicates churn, 0 indicates no churn).

---

## Pipeline Steps

1. **Data Preprocessing:**
   - Dropped irrelevant columns (e.g., `RowNumber`, `CustomerId`, `Surname`).
   - Encoded categorical variables using:
     - Label encoding for `Gender`.
     - One-hot encoding for `Geography`.
   - Scaled the features using `StandardScaler`.

2. **Model Training:**
   - Designed a sequential ANN model with:
     - Input Layer: Matches the number of features.
     - Hidden Layers: Two layers with ReLU activation.
     - Output Layer: Single neuron with sigmoid activation.
   - Binary cross-entropy as the loss function.
   - Adam optimizer with a learning rate of 0.01.
   - Early stopping to avoid overfitting.

3. **Model Evaluation:**
   - Achieved ~86% validation accuracy.
   - Monitored training and validation performance using TensorBoard.

4. **Deployment:**
   - Created an interactive Streamlit app for real-time predictions.

---

## Model Architecture

| Layer Type      | Output Shape | Number of Parameters |
|------------------|--------------|-----------------------|
| Dense (Hidden 1) | 64           | 832                   |
| Dense (Hidden 2) | 32           | 2080                  |
| Dense (Output)   | 1            | 33                    |

Total trainable parameters: **2,945**

---

## Streamlit App

The Streamlit app provides an easy-to-use interface for making predictions based on the trained ANN model. Users can input values for features like `CreditScore`, `Age`, `Gender`, etc., and receive a prediction on whether the customer is likely to churn.

### App Features:
- Interactive sliders and dropdowns for user inputs.
- Real-time predictions using the trained `model.h5`.
- Encoders and scaler are loaded from saved `.pkl` files.

---

## Dependencies

The following Python libraries are required to run the project:

- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `streamlit`
- `pickle`

The full list of dependencies can be found in `requirements.txt`.

---

## How to Run

### Local Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/Mayankpratapsingh022/Churn-Prediction.git
   cd Churn-Prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser (default: http://localhost:8501).

---

## Results

- **Validation Accuracy:** 86%
- **Loss Reduction:** Consistent decrease with early stopping.
- **Streamlit App:** Interactive and user-friendly.

---

## Folder Structure

```
├── Churn_Modelling.csv        # Dataset file
├── app.py                     # Streamlit app script
├── experiments.ipynb          # Jupyter notebook with preprocessing and model training
├── model.h5                   # Trained ANN model
├── label_encoder_gender.pkl   # Saved label encoder for gender
├── onehot_encoder_geo.pkl     # Saved one-hot encoder for geography
├── scaler.pkl                 # Saved scaler for feature normalization
├── requirements.txt           # List of dependencies
├── README.md                  # Project README file
```

---

