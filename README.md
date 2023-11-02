# ParkinSVM: Parkinson's Disease Prediction Using SVM

ParkinSVM is a project focused on predicting Parkinson's disease using Support Vector Machine (SVM). Its goal is to provide an early diagnosis of Parkinson's disease based on relevant features extracted from voice recordings, enabling early detection and effective treatment.

## Getting Started

### Prerequisites

Before you get started, ensure you have the following prerequisites:

- Python 3
- Libraries: pandas, numpy, scikit-learn

### Installation

Follow these steps to set up and run the project:

1. Clone the repository:

   ```bash
   git clone https://github.com/charvijain12/ParkinSVM.git
   ```

2. Install the required dependencies if not already installed:

   ```bash
   pip install pandas numpy scikit-learn
   ```

## Data Collection & Analysis

- The project loads the dataset from a CSV file containing voice recordings.
- It displays the first 5 rows of the dataset.
- Basic information about the dataset is provided.

## Data Pre-Processing

The project performs the following data pre-processing steps:

- Separates data features and the target variable.
- Splits the data into training and testing sets.
- Standardizes the data to scale the features.

## Model Training

- A Support Vector Machine (SVM) with a linear kernel is chosen as the classification model.
- The project trains the SVM model using the training data.

## Model Evaluation

- Accuracy scores for the training and testing data are calculated.

## Building a Predictive System

The project demonstrates how to build a predictive system using the trained SVM model:

- It provides an example input, standardizes it, and predicts the presence of Parkinson's disease.

## License

This project is licensed under the MIT License.

### Acknowledgments

- Dataset source: UCI Machine Learning Repository

Feel free to explore, contribute, and use this project to support early diagnosis of Parkinson's disease using SVM.
