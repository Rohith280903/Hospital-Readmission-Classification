# Hospital-Readmission-Classification

## Project Overview
This project implements machine learning techniques to predict hospital readmission rates. Hospital readmission is a critical healthcare metric that affects patient outcomes and healthcare costs. By predicting the likelihood of patient readmission, healthcare providers can take preventive measures and improve patient care quality.

## Table of Contents
- [Dataset](#dataset)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Concepts Explained](#concepts-explained)
- [Implementation Details](#implementation-details)
- [Libraries Used](#libraries-used)
- [How to Run](#how-to-run)

## Dataset
The project uses the **Hospital Readmission Dataset** (`hospital_readmissions.csv`) which contains:
- Patient demographic information (age, gender, etc.)
- Medical history and diagnoses
- Treatment and medication data
- Previous admission records
- Hospital stay duration
- **Target variable**: `readmitted` (whether patient was readmitted)

## Machine Learning Workflow

### 1. Data Preprocessing
**Code Location**: Cell 2 in `rohithCA2DS.ipynb`

#### Steps Implemented:
- **Data Loading**: Read CSV file using pandas
- **Duplicate Removal**: `data.drop_duplicates()` eliminates redundant records
- **Missing Value Handling**: `data.dropna()` removes rows with missing values
- **Label Encoding**: Converts categorical variables to numerical format using `LabelEncoder()`
  - Categorical features (like gender, diagnosis codes) are transformed to integers
  - This is necessary because machine learning algorithms work with numerical data
- **Feature-Target Separation**:
  - `X`: All features except the target variable
  - `y`: Target variable ('readmitted')

**Concept**: Data preprocessing is crucial for model performance. Clean, properly formatted data ensures accurate predictions and prevents errors during training.

### 2. Exploratory Data Analysis (EDA)
**Code Location**: Cell 3 in `rohithCA2DS.ipynb`

#### Visualizations Created:

**a) Target Variable Distribution (Bar Chart)**
- Shows the count of readmitted vs non-readmitted patients
- Helps identify class imbalance in the dataset
- **Concept**: Understanding class distribution is crucial for selecting appropriate evaluation metrics and handling imbalanced datasets

**b) Age Distribution (Histogram)**
- Displays the frequency distribution of patient ages
- Uses 10 bins to group age ranges
- Helps identify age patterns related to readmission
- **Concept**: Histograms reveal the shape of data distribution (normal, skewed, bimodal, etc.)

**c) Correlation Matrix (Heatmap)**
- Visualizes relationships between all numerical features
- Color intensity indicates correlation strength (coolwarm colormap)
- Helps identify:
  - Highly correlated features (potential multicollinearity)
  - Features strongly correlated with the target variable
- **Concept**: Correlation analysis guides feature selection and engineering decisions

### 3. Data Splitting and Scaling
**Code Location**: Cell 4 in `rohithCA2DS.ipynb`

#### Train-Test Split
```python
train_test_split(X, y, test_size=0.2, random_state=42)
```
- **80% Training Data**: Used to train the model
- **20% Test Data**: Used to evaluate model performance
- **random_state=42**: Ensures reproducibility of results

**Concept**: Splitting data prevents overfitting and provides unbiased evaluation. The model never sees test data during training, ensuring honest performance assessment.

#### Feature Scaling (StandardScaler)
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Why Scaling is Important**:
- Standardizes features to have mean=0 and standard deviation=1
- Prevents features with larger values from dominating the model
- Essential for algorithms sensitive to feature magnitude:
  - Logistic Regression
  - Neural Networks
  - K-Nearest Neighbors
  - Support Vector Machines

**Note**: Only fit on training data, then transform both train and test to prevent data leakage.

## Concepts Explained

### Classification
- **Type**: Supervised learning task
- **Goal**: Predict categorical outcomes (readmitted: Yes/No)
- **Algorithms Available in Code**:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - AdaBoost Classifier
  - Gradient Boosting Classifier
  - Gaussian Naive Bayes
  - K-Nearest Neighbors
  - Multi-Layer Perceptron (Neural Network)

### Cross-Validation
**Imported but not yet implemented in visible cells**:
- `KFold`: Divides data into K folds for robust performance estimation
- `LeaveOneOut`: Uses each sample as validation once
- `cross_val_score`: Evaluates model using cross-validation

**Purpose**: Provides more reliable performance estimates than a single train-test split.

### Ensemble Methods
Multiple ensemble algorithms are imported:
- **Random Forest**: Combines multiple decision trees
- **AdaBoost**: Sequentially trains weak learners focusing on mistakes
- **Gradient Boosting**: Builds trees sequentially to correct previous errors

**Concept**: Ensemble methods often outperform individual models by combining predictions.

### Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Reduces feature dimensions while preserving variance
- **Use Case**: Handles high-dimensional data, speeds up training, removes noise

### Clustering
- **KMeans**: Unsupervised algorithm to group similar patients
- **Potential Use**: Patient segmentation, pattern discovery

### Model Evaluation Metrics
Imported metrics for comprehensive evaluation:
- **Accuracy Score**: Overall correctness
- **Classification Report**: Precision, recall, F1-score per class
- **Confusion Matrix**: True positives, false positives, true negatives, false negatives
- **ROC Curve & AUC Score**: Model's ability to discriminate between classes

## Libraries Used

### Data Manipulation
- **pandas**: Data loading and manipulation
- **numpy**: Numerical computations

### Visualization
- **matplotlib**: Creating plots and charts

### Machine Learning
- **scikit-learn**: Complete ML pipeline
  - Model selection and evaluation
  - Preprocessing and feature engineering
  - Classification algorithms
  - Clustering and dimensionality reduction
  - Performance metrics

## Implementation Details

### Current Implementation Status:
✅ **Completed**:
1. Library imports
2. Data loading and preprocessing
3. Exploratory data analysis with visualizations
4. Train-test split
5. Feature scaling

📋 **Next Steps** (not in visible code):
1. Model training with multiple algorithms
2. Cross-validation implementation
3. Hyperparameter tuning
4. Model comparison and selection
5. Final model evaluation on test set
6. ROC curve and AUC analysis
7. Feature importance analysis

## How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib scikit-learn
```

### Steps
1. Clone this repository
2. Ensure `hospital_readmissions.csv` is in the correct path
3. Open `rohithCA2DS.ipynb` in Jupyter Notebook or JupyterLab
4. Run all cells sequentially

### Update File Path
Update the CSV path in Cell 2 to match your system:
```python
data = pd.read_csv(r"your_path_here/hospital_readmissions.csv")
```

## Key Takeaways

1. **Preprocessing is Critical**: Clean data = Better models
2. **Visualization Guides Decisions**: EDA reveals patterns and issues
3. **Proper Scaling Matters**: Essential for many ML algorithms
4. **Train-Test Split Prevents Overfitting**: Never test on training data
5. **Multiple Algorithms Available**: Different algorithms suit different problems
6. **Ensemble Methods**: Often provide superior performance
7. **Comprehensive Evaluation**: Use multiple metrics beyond accuracy

## Project Significance

This project demonstrates:
- **Healthcare Impact**: Predicting readmissions can save lives and reduce costs
- **End-to-End ML Pipeline**: From data loading to model evaluation
- **Best Practices**: Proper preprocessing, scaling, and validation
- **Multiple Techniques**: Classification, ensemble methods, clustering, dimensionality reduction

## Future Enhancements
- Implement model training and comparison
- Add cross-validation results
- Feature importance analysis
- Hyperparameter optimization
- Deploy model as a web service
- Handle class imbalance (SMOTE, class weights)
- Add more advanced visualizations

## Author
Rohith280903

## License
This project is available for educational purposes.

---

**Note**: This README provides detailed explanations of the machine learning concepts and workflow implemented in the `rohithCA2DS.ipynb` notebook. The code demonstrates foundational steps in building a hospital readmission prediction system.
