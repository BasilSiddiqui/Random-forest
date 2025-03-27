import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

# Load dataset
dataset = pd.read_csv(r"C:\Users\basil\OneDrive\Desktop\Base\Undergraduate\Semester 4\F78DS - Data Science Life Cycle\TravelInfo.csv")

# Select features (Age, Income) and target (TravelAbroad)
X = dataset[["Age", "Income"]]
y = dataset["TravelAbroad"]  # Converted to a Series

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Function to train and evaluate models
def train_and_evaluate(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nConfusion Matrix for {model_name}:\n", cm)
    
    # Visualizing decision boundary
    X_set, y_set = X_test, y_test.values  # Ensure y_set is a NumPy array
    X1, X2 = np.meshgrid(
        np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
        np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01)
    )

    plt.contourf(
        X1, X2,
        model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
        alpha=0.75, cmap=ListedColormap(('red', 'green'))
    )

    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())

    # Fix scatter plot indexing
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(
            X_set[y_set == j, 0], X_set[y_set == j, 1],
            c=ListedColormap(('red', 'green'))(i), label=j
        )

    plt.title(f'{model_name} Classification (Test set)')
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.legend()
    plt.show()

# Train and evaluate Decision Tree
decision_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
train_and_evaluate(decision_tree, "Decision Tree")

# Train and evaluate Random Forest
random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)  # Increased trees to 100
train_and_evaluate(random_forest, "Random Forest")