# Travel Abroad Prediction - Machine Learning Project

## Overview
This project implements two machine learning classifiers (Decision Tree and Random Forest) to predict whether a person is likely to travel abroad based on their age and income. The project includes data preprocessing, model training, evaluation, and visualization of decision boundaries.

## Project Structure
```
TravelAbroad-Prediction/
├── TravelInfo.csv           # Dataset containing age, income, and travel information
├── TravelPrediction.py      # Main Python script with implementation
└── README.md                # This file
```

## Requirements
- Python 3.x
- Libraries:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn

Install requirements with:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Code Explanation

### 1. Data Loading and Preparation
```python
dataset = pd.read_csv(r"C:\Users\basil\OneDrive\Desktop\Base\Undergraduate\Semester 4\F78DS - Data Science Life Cycle\TravelInfo.csv")
X = dataset[["Age", "Income"]]
y = dataset["TravelAbroad"]
```
- Loads the dataset from CSV file
- Selects features (Age and Income) and target variable (TravelAbroad)

### 2. Data Splitting
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
```
- Splits data into 75% training and 25% testing sets
- `random_state=0` ensures reproducible results

### 3. Feature Scaling
```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
- Standardizes features by removing mean and scaling to unit variance
- Important for algorithms sensitive to feature scales

### 4. Model Training and Evaluation Function
The `train_and_evaluate` function:
1. Trains the model on training data
2. Makes predictions on test data
3. Prints confusion matrix
4. Visualizes decision boundaries

### 5. Decision Tree Classifier
```python
decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
```
- Uses entropy as the splitting criterion
- Visualizes non-linear decision boundaries

### 6. Random Forest Classifier
```python
random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
```
- Ensemble of 100 decision trees
- Uses entropy for splitting
- Typically more accurate than single decision tree

## How to Use
1. Clone the repository
2. Ensure you have the required Python libraries installed
3. Place your dataset (named `TravelInfo.csv`) in the project directory
4. Run the script:
```bash
python TravelPrediction.py
```

## Output Interpretation
- **Confusion Matrix**: Shows true positives, true negatives, false positives, and false negatives
- **Decision Boundary Plots**: Visualize how the model separates the two classes (Travel/No Travel)
  - Red region: Predicted "No Travel"
  - Green region: Predicted "Travel"
  - Dots: Actual test data points

## Key Concepts
- **Decision Tree**: A flowchart-like structure that makes decisions based on feature thresholds
- **Random Forest**: An ensemble method that combines multiple decision trees to improve accuracy
- **Feature Scaling**: Standardizing features to have mean=0 and variance=1
- **Decision Boundary**: The surface that separates different predicted classes in feature space

## Potential Improvements
- Try different classification algorithms (SVM, Logistic Regression, etc.)
- Perform hyperparameter tuning
- Add more relevant features if available
- Implement cross-validation for more reliable performance estimates

This project demonstrates fundamental machine learning concepts and provides a clear visualization of how classification models make decisions based on input features.
