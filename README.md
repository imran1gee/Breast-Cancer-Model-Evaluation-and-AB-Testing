# Breast-Cancer-Model-Evaluation-and-AB-Testing

This project evaluates the performance of different machine learning models for breast cancer classification using a dataset. The models compared include:

- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Random Forest**

Additionally, the project includes **A/B testing** to statistically compare the performance of the models and determine which performs better in terms of classification accuracy. The goal of this project is to predict whether a tumor is malignant or benign based on various features extracted from breast cancer biopsy data. The dataset is pre-processed, and three machine learning models are applied. The models are then evaluated using performance metrics such as accuracy, confusion matrix, and classification report. 

### Key Steps

1. **Data Loading and Exploration**
   - The dataset is loaded and basic exploratory data analysis is performed.
   - Missing values, summary statistics, and data types are checked.

2. **Data Cleaning**
   - Unnecessary columns are dropped, and the dataset is cleaned for further analysis.

3. **Target Variable Encoding**
   - The target variable, `diagnosis`, is encoded from categorical values ('M' and 'B') to numerical values (1 and 0).

4. **Data Splitting**
   - The dataset is split into training and testing sets to train and evaluate the models.

5. **Feature Scaling**
   - StandardScaler is applied to scale the features to improve model performance.

6. **Model Training and Evaluation**
   - Three different models (Logistic Regression, SVM, and Random Forest) are trained on the scaled data.
   - The models are evaluated on accuracy, classification report, and confusion matrix.

7. **Statistical Analysis (Paired t-test)**
   - A paired t-test is performed to assess whether there are significant performance differences between the models.

## Models Evaluated

### 1. Logistic Regression
   - **Logistic Regression** is a simple yet powerful model used for binary classification. It is evaluated using accuracy, precision, recall, and F1-score.

### 2. Support Vector Machine (SVM)
   - **Support Vector Machine (SVM)** is a powerful classification technique that finds the hyperplane that best separates the classes. It is tuned and evaluated for optimal performance.

### 3. Random Forest Classifier
   - **Random Forest Classifier** is an ensemble learning model that combines multiple decision trees to improve prediction accuracy. It is also evaluated and compared to the other models.

## Results

- **Model Accuracies**: The accuracies for each model are reported and compared.
- **Confusion Matrix**: The confusion matrix is used to further assess the model performance.
- **Classification Report**: Precision, recall, F1-score, and support are calculated for each model.
- **Paired t-test**: The t-test results reveal whether the differences in performance between the models are statistically significant.

## Technologies Used

- **Python**: Programming language used for implementing machine learning models and statistical analysis.
- **Pandas**: For data manipulation and cleaning.
- **Scikit-learn**: For machine learning models, data preprocessing, and evaluation.
- **SciPy**: For statistical analysis (paired t-test).
- **Jupyter Notebook**: For interactive code execution and visualization.

## Dataset

The dataset used in this project is the **Breast Cancer Wisconsin dataset** (available publicly). The dataset contains features of cell nuclei present in breast cancer biopsies, and the target variable is the diagnosis of the tumor (malignant or benign).

- **Source**: [Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic))

## How to Run the Code

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/Breast-Cancer-Classification.git
