# Parkinsons Disease Prediction Using SVM

This project focuses on predicting Parkinson's disease using Support Vector Machine (SVM). The goal is to provide an early diagnosis of Parkinson's disease based on relevant features extracted from voice recordings. This helps in early detection and effective treatment of the disease.

## Getting Started

### Prerequisites

- Python 3
- Libraries: pandas, numpy, scikit-learn

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/charvijain12/Parkinsons-Disease-Prediction.git
   ```

2. Install the required dependencies if not already installed:
   ```
   pip install pandas numpy scikit-learn
   ```

## Data Collection & Analysis

- The dataset is loaded from a CSV file containing voice recordings.
- The first 5 rows of the dataset are displayed.
- Basic information about the dataset is provided.

## Data Pre-Processing

- Data features and the target variable are separated.
- Data is split into training and testing sets.
- Data standardization is performed to scale the features.

## Model Training

- A Support Vector Machine (SVM) with a linear kernel is chosen as the classification model.
- The SVM model is trained using the training data.

## Model Evaluation

- Accuracy scores for the training and testing data are calculated.

## Building a Predictive System

- The project demonstrates how to make predictions using the trained SVM model.
- It provides an example input, standardizes it, and predicts the presence of Parkinson's disease.

## License

This project is licensed under the MIT License.

### Acknowledgments

- Dataset source: UCI Machine Learning Repository
