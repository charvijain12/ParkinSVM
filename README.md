# Parkinson's Disease Prediction
Parkinson's Disease Prediction is a machine learning project that aims to predict the presence of Parkinson's disease in individuals based on voice recordings. Parkinson's disease is a neurodegenerative disorder that affects movement and can lead to tremors, rigidity, and difficulty with speech. Early detection of Parkinson's disease is crucial for effective treatment, and this project aims to provide a tool for early diagnosis.

The project uses a dataset of voice recordings from individuals with and without Parkinson's disease, and applies machine learning algorithms to classify the recordings as either Parkinson's positive or negative. The algorithms used include K-Nearest Neighbors, Random Forest, and Support Vector Machine. The dataset is preprocessed to extract relevant features from the recordings, such as mean frequency and standard deviation.

The project is implemented in Python using the scikit-learn library for machine learning and pandas for data manipulation. The project also includes data visualization using matplotlib and seaborn, and performance evaluation using metrics such as accuracy, precision, recall, and F1 score.

## How to use
To use this project, you can clone the repository and run the `Parkinsons_Disease_Prediction.ipynb` file in Jupyter Notebook or Google Colab. The notebook contains all the code and documentation needed to run the project, including data loading, preprocessing, model training, and performance evaluation. You can modify the code and parameters to experiment with different models and features, or use the existing code as a basis for your own Parkinson's disease prediction project.

## Dataset
The dataset used in this project is the `Parkinsons Telemonitoring Data Set` from the UCI Machine Learning Repository. The dataset contains voice recordings from individuals with Parkinson's disease and healthy individuals, collected by a telemonitoring device. The recordings include sustained vowel phonations as well as speech tasks, and are processed to extract various acoustic features. The dataset includes 5875 recordings from 42 individuals, and is split into training and testing sets.

## Project structure
The project repository has the following structure:

1. `Parkinsons_Disease_Prediction.ipynb`: Jupyter Notebook file containing the project code and documentation.

2. `parkinsons.data`: CSV file containing the training data.

3. `parkinsons.names`: Text file containing the dataset description and attribute information.

## Dependencies
1. Python 3.6+

2. pandas 1.0+

3. scikit-learn 0.23+

4. matplotlib 3.2+

5. seaborn 0.10+


## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or feedback

## License
This project is licensed under the `MIT License`.
