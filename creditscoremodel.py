#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install --upgrade pandas')


# In[1]:


get_ipython().system('pip install --upgrade numpy')


# In[2]:


get_ipython().system('pip install --upgrade pandas')


# In[5]:


get_ipython().system('pip install numpy==1.24.4')


# In[6]:


import numpy as np
import pandas as pd

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)


# In[4]:


import pandas as pd

# Load the dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Replace with your actual file path
credit_data = pd.read_csv(dataset_path)

# Display the first few rows
print(credit_data)


# In[5]:


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with the correct path
credit_data = pd.read_csv(dataset_path)

# Display the first few rows to inspect
print(credit_data.head())

# Create a binary target column based on CreditScore (Good = 1, Bad = 0)
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)

# Drop the original 'CreditScore' column as it's now replaced by the 'Target' column
credit_data = credit_data.drop(columns=['CreditScore'])

# Inspect the cleaned data
print(credit_data.head())


# In[6]:


# Define features (X) and target (y)
X = credit_data.drop(columns=['Target'])
y = credit_data['Target']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting splits
print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")


# In[7]:


# Standardize the features (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[8]:


# Initialize the Logistic Regression model
logreg = LogisticRegression()

# Train the model
logreg.fit(X_train_scaled, y_train)


# In[9]:


# Predict on the test set
y_pred = logreg.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)


# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with the correct path
credit_data = pd.read_csv(dataset_path)

# Create binary target column based on CreditScore
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features and target
X = credit_data.drop(columns=['Target'])
y = credit_data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = logreg.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test_scaled)[:,1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Feature Importance (Coefficients)
coefficients = logreg.coef_[0]
plt.figure(figsize=(10, 6))
plt.bar(X.columns, coefficients)
plt.title('Feature Importance (Logistic Regression Coefficients)')
plt.xlabel('Feature')
plt.ylabel('Coefficient Value')
plt.xticks(rotation=45)
plt.show()


# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with the correct path
credit_data = pd.read_csv(dataset_path)

# Create binary target column based on CreditScore
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features and target
X = credit_data.drop(columns=['Target'])
y = credit_data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = decision_tree.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, decision_tree.predict_proba(X_test_scaled)[:,1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Visualizing the Decision Tree (optional)
from sklearn.tree import plot_tree
plt.figure(figsize=(15, 10))
plot_tree(decision_tree, filled=True, feature_names=X.columns, class_names=["Bad", "Good"], rounded=True, fontsize=10)
plt.title("Decision Tree Visualization")
plt.show()


# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with the correct path
credit_data = pd.read_csv(dataset_path)

# Create binary target column based on CreditScore
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features and target
X = credit_data.drop(columns=['Target'])
y = credit_data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, 10, None],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'criterion': ['gini', 'entropy'],  # Criterion for splitting (Gini or Entropy)
    'max_features': [None, 'auto', 'sqrt', 'log2']  # Number of features to consider when looking for the best split
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best parameters found by GridSearchCV
print("Best Parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Get the best model
best_tree = grid_search.best_estimator_

# Predict on the test set using the best model
y_pred_best = best_tree.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_best)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_best))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_tree.predict_proba(X_test_scaled)[:,1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Visualizing the Decision Tree (best model)
plt.figure(figsize=(15, 10))
plot_tree(best_tree, filled=True, feature_names=X.columns, class_names=["Bad", "Good"], rounded=True, fontsize=10)
plt.title("Best Decision Tree Visualization (After Hyperparameter Tuning)")
plt.show()


# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with the correct path
credit_data = pd.read_csv(dataset_path)

# Create binary target column based on CreditScore
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features and target
X = credit_data.drop(columns=['Target'])
y = credit_data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7, 10, None],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'criterion': ['gini', 'entropy'],  # Criterion for splitting (Gini or Entropy)
    'max_features': [None, 'auto', 'sqrt', 'log2']  # Number of features to consider when looking for the best split
}

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Best parameters found by GridSearchCV
print("Best Parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Get the best model
best_tree = grid_search.best_estimator_

# Predict on the test set using the best model
y_pred_best = best_tree.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_best)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad", "Good"], yticklabels=["Bad", "Good"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_best))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_tree.predict_proba(X_test_scaled)[:,1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()

# Visualizing the Decision Tree (best model)
plt.figure(figsize=(15, 10))
plot_tree(best_tree, filled=True, feature_names=X.columns, class_names=["Bad", "Good"], rounded=True, fontsize=10)
plt.title("Best Decision Tree Visualization (After Hyperparameter Tuning)")
plt.show()

# Scatter Plot
# Choose two features to visualize in the scatter plot
feature1 = 'AnnualIncome'  # Replace with actual feature names from your dataset
feature2 = 'LoanAmount'  # Replace with actual feature names from your dataset

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[feature1], y=X_test[feature2], hue=y_test, palette='coolwarm', style=y_test, markers=['o', 's'])
plt.title(f'Scatter Plot of {feature1} vs {feature2} with Classification')
plt.xlabel(feature1)
plt.ylabel(feature2)
plt.legend(title='Class', labels=["Bad", "Good"])
plt.show()


# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with the correct path
credit_data = pd.read_csv(dataset_path)

# Create binary target column based on CreditScore
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features and target
X = credit_data[['AnnualIncome', 'LoanAmount']]  # Select only the two features for scatter plot
y = credit_data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SVM model with a linear kernel
svm_model = SVC(kernel='linear', random_state=42)

# Train the model
svm_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test_scaled)

# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Create a mesh grid to plot decision boundary
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict class for each point in the mesh grid
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')

# Plot the actual points (test set) on top of the decision boundary
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=100, marker='o')
plt.title('SVM Decision Boundary and Scatter Plot')
plt.xlabel('Standardized AnnualIncome')
plt.ylabel('Standardized LoanAmount')
plt.colorbar(label='Class')
plt.show()


# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with the correct path
credit_data = pd.read_csv(dataset_path)

# Create binary target column based on CreditScore
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features and target
X = credit_data[['AnnualIncome', 'LoanAmount']]  # Select only the two features for scatter plot
y = credit_data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the SVM model with RBF kernel
svm_model_rbf = SVC(kernel='rbf', random_state=42)

# Train the model
svm_model_rbf.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_rbf = svm_model_rbf.predict(X_test_scaled)

# Accuracy of the model
accuracy_rbf = accuracy_score(y_test, y_pred_rbf)
print(f"Accuracy (RBF Kernel): {accuracy_rbf:.4f}")

# Create a mesh grid to plot decision boundary
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict class for each point in the mesh grid
Z = svm_model_rbf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')

# Plot the actual points (test set) on top of the decision boundary
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=100, marker='o')
plt.title('SVM (RBF Kernel) Decision Boundary and Scatter Plot')
plt.xlabel('Standardized AnnualIncome')
plt.ylabel('Standardized LoanAmount')
plt.colorbar(label='Class')
plt.show()


# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with the correct path
credit_data = pd.read_csv(dataset_path)

# Create binary target column based on CreditScore
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features and target
X = credit_data[['AnnualIncome', 'LoanAmount']]  # Select only the two features for scatter plot
y = credit_data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN model
knn_model = KNeighborsClassifier(n_neighbors=2)  # n_neighbors is the number of neighbors to consider

# Train the model
knn_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_knn = knn_model.predict(X_test_scaled)

# Accuracy of the model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy (KNN): {accuracy_knn:.4f}")

# Create a mesh grid to plot decision boundary
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict class for each point in the mesh grid
Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')

# Plot the actual points (test set) on top of the decision boundary
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=100, marker='o')
plt.title('KNN Decision Boundary and Scatter Plot')
plt.xlabel('Standardized AnnualIncome')
plt.ylabel('Standardized LoanAmount')
plt.colorbar(label='Class')
plt.show()


# In[38]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with your file path
credit_data = pd.read_csv(dataset_path)

# Check for missing values and handle them (if any)
credit_data.isnull().sum()  # Checking for missing values in the dataset

# Create binary target column based on CreditScore
# If CreditScore > 600, assign 1 (Good Credit), otherwise 0 (Bad Credit)
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)

# Drop the original 'CreditScore' column as we already used it to create the target
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features (let's use 'AnnualIncome' and 'LoanAmount' for this example)
X = credit_data[['AnnualIncome', 'LoanAmount']]  # Features
y = credit_data['Target']  # Target (binary classification)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Naive Bayes model
nb_model = GaussianNB()

# Train the model
nb_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_nb = nb_model.predict(X_test_scaled)

# Accuracy of the model
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Accuracy (Naive Bayes - Binary Classification): {accuracy_nb:.4f}")

# Create a mesh grid to plot the decision boundary (use only the two features)
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict class for each point in the mesh grid
Z = nb_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')

# Plot the actual points (test set) on top of the decision boundary
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=100, marker='o')
plt.title('Naive Bayes Binary Classification Decision Boundary and Scatter Plot')
plt.xlabel('Standardized AnnualIncome')
plt.ylabel('Standardized LoanAmount')
plt.colorbar(label='Class')
plt.show()


# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with your file path
credit_data = pd.read_csv(dataset_path)

# Check for missing values and handle them (if any)
credit_data.isnull().sum()  # Checking for missing values in the dataset

# Create binary target column based on CreditScore
# If CreditScore > 600, assign 1 (Good Credit), otherwise 0 (Bad Credit)
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)

# Drop the original 'CreditScore' column as we already used it to create the target
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features (let's use 'AnnualIncome' and 'LoanAmount' for this example)
X = credit_data[['AnnualIncome', 'LoanAmount']]  # Features
y = credit_data['Target']  # Target (binary classification)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Gradient Boosting Classifier model
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Train the model
gb_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_gb = gb_model.predict(X_test_scaled)

# Accuracy of the model
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print(f"Accuracy (Gradient Boosting - Binary Classification): {accuracy_gb:.4f}")

# Create a mesh grid to plot the decision boundary (use only the two features)
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict class for each point in the mesh grid
Z = gb_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')

# Plot the actual points (test set) on top of the decision boundary
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=100, marker='o')
plt.title('Gradient Boosting Binary Classification Decision Boundary and Scatter Plot')
plt.xlabel('Standardized AnnualIncome')
plt.ylabel('Standardized LoanAmount')
plt.colorbar(label='Class')
plt.show()


# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# Load dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with your file path
credit_data = pd.read_csv(dataset_path)

# Check for missing values and handle them (if any)
credit_data.isnull().sum()  # Checking for missing values in the dataset

# Create binary target column based on CreditScore
# If CreditScore > 600, assign 1 (Good Credit), otherwise 0 (Bad Credit)
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)

# Drop the original 'CreditScore' column as we already used it to create the target
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features (using 'AnnualIncome' and 'LoanAmount' for this example)
X = credit_data[['AnnualIncome', 'LoanAmount']]  # Features
y = credit_data['Target']  # Target (binary classification)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Linear Discriminant Analysis (LDA) model
lda_model = LinearDiscriminantAnalysis()

# Train the model
lda_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_lda = lda_model.predict(X_test_scaled)

# Accuracy of the model
accuracy_lda = accuracy_score(y_test, y_pred_lda)
print(f"Accuracy (LDA - Binary Classification): {accuracy_lda:.4f}")

# Create a mesh grid to plot the decision boundary (use only the two features)
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict class for each point in the mesh grid
Z = lda_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')

# Plot the actual points (test set) on top of the decision boundary
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=100, marker='o')
plt.title('LDA Binary Classification Decision Boundary and Scatter Plot')
plt.xlabel('Standardized AnnualIncome')
plt.ylabel('Standardized LoanAmount')
plt.colorbar(label='Class')
plt.show()


# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with your file path
credit_data = pd.read_csv(dataset_path)

# Check for missing values and handle them (if any)
credit_data.isnull().sum()  # Checking for missing values in the dataset

# Create binary target column based on CreditScore
# If CreditScore > 600, assign 1 (Good Credit), otherwise 0 (Bad Credit)
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)

# Drop the original 'CreditScore' column as we already used it to create the target
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features (using 'AnnualIncome' and 'LoanAmount' for this example)
X = credit_data[['AnnualIncome', 'LoanAmount']]  # Features
y = credit_data['Target']  # Target (binary classification)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest model
rf_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test_scaled)

# Accuracy of the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy (Random Forest): {accuracy_rf:.4f}")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Feature Importance from Random Forest
feature_importances = rf_model.feature_importances_
print("Feature Importances:", feature_importances)

# Plot feature importance
plt.bar(X.columns, feature_importances)
plt.title('Random Forest Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Optional: Create a mesh grid to visualize decision boundary (if 2D features)
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Predict class for each point in the mesh grid
Z = rf_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')

# Plot the actual points (test set) on top of the decision boundary
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', s=100, marker='o')
plt.title('Random Forest Binary Classification Decision Boundary and Scatter Plot')
plt.xlabel('Standardized AnnualIncome')
plt.ylabel('Standardized LoanAmount')
plt.colorbar(label='Class')
plt.show()


# In[44]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with your file path
credit_data = pd.read_csv(dataset_path)

# Create binary target column based on CreditScore
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)

# Drop the original 'CreditScore' column as we already used it to create the target
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features (using 'AnnualIncome' and 'LoanAmount' for this example)
X = credit_data[['AnnualIncome', 'LoanAmount']].values  # Features (as NumPy array)
y = credit_data['Target'].values  # Target (binary classification)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize AdaBoost model with DecisionTreeClassifier as the base estimator
base_estimator = DecisionTreeClassifier(max_depth=1)  # Weak learner (decision stump)
adaboost_model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)

# Train the model
adaboost_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_adaboost = adaboost_model.predict(X_test_scaled)

# Accuracy of the model
accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
print(f"Accuracy (AdaBoost): {accuracy_adaboost:.4f}")

# Classification Report
print("Classification Report (AdaBoost):")
print(classification_report(y_test, y_pred_adaboost))

# Create scatter plot with decision boundary
plt.figure(figsize=(10, 6))

# Plot the training points
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', marker='o', label="Training data")
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='coolwarm', edgecolors='k', marker='x', label="Test data")

# Create mesh grid to plot decision boundary
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Use AdaBoost to predict for the mesh grid points
Z = adaboost_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')

# Labels and title
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Loan Amount (scaled)')
plt.title('AdaBoost: Decision Boundary and Data Points')
plt.legend()
plt.show()


# In[45]:


get_ipython().system('pip install catboost')


# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

# Load dataset
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with your file path
credit_data = pd.read_csv(dataset_path)

# Create binary target column based on CreditScore
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)

# Drop the original 'CreditScore' column as we already used it to create the target
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features (using 'AnnualIncome' and 'LoanAmount' for this example)
X = credit_data[['AnnualIncome', 'LoanAmount']].values  # Features (as NumPy array)
y = credit_data['Target'].values  # Target (binary classification)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the CatBoost model
catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, random_state=42, verbose=0)
catboost_model.fit(X_train_scaled, y_train)

# Create scatter plot with decision boundary
plt.figure(figsize=(10, 6))

# Plot the training points
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', edgecolors='k', marker='o', label="Training data")
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='coolwarm', edgecolors='o', marker='x', label="Test data")

# Create mesh grid to plot decision boundary
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Use CatBoost to predict for the mesh grid points
Z = catboost_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')

# Labels and title
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Loan Amount (scaled)')
plt.title('CatBoost: Decision Boundary and Data Points')
plt.legend()
plt.show()


# In[49]:


get_ipython().system('pip install scikit-learn')


# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset (replace the path with your actual file path)
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with your actual file path
credit_data = pd.read_csv(dataset_path)

# Create binary target column based on CreditScore
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)

# Drop the original 'CreditScore' column
credit_data = credit_data.drop(columns=['CreditScore'])

# Define features (use 'AnnualIncome' and 'LoanAmount' for this example)
X = credit_data[['AnnualIncome', 'LoanAmount']].values  # Features (NumPy array)
y = credit_data['Target'].values  # Target (binary classification)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the MLP model
mlp_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
mlp_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = mlp_model.predict(X_test_scaled)

# Model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print performance metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Visualizing the MLP Model Decision Boundary
plt.figure(figsize=(10, 6))

# Plot the training points
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', marker='o', label="Training data")
plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='coolwarm', marker='x', label="Test data")

# Create mesh grid to plot decision boundary
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Use the MLP model to predict for the mesh grid points
Z = mlp_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')

# Labels and title
plt.xlabel('Annual Income (scaled)')
plt.ylabel('Loan Amount (scaled)')
plt.title('MLP: Decision Boundary and Data Points')
plt.legend()
plt.show()


# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset (replace with your actual file path)
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with your actual file path
credit_data = pd.read_csv(dataset_path)

# Preprocess the data (create binary target column)
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)
credit_data = credit_data.drop(columns=['CreditScore'])  # Drop original CreditScore

# Select features for classification
X = credit_data[['AnnualIncome', 'LoanAmount']].values  # Features (NumPy array)
y = credit_data['Target'].values  # Target (binary classification)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a function to evaluate the models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    return accuracy, conf_matrix, class_report

# List of models to evaluate (without ANN)
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'CatBoost': CatBoostClassifier(learning_rate=0.1, iterations=500, depth=6, cat_features=[], verbose=False),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300),
}

# Create an empty dictionary to store results
results = {}

# Evaluate all models
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    accuracy, conf_matrix, class_report = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    results[model_name] = {
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'class_report': class_report
    }
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    print("-" * 80)

# Displaying the results
for model_name, result in results.items():
    print(f"Results for {model_name}:")
    print(f"Accuracy: {result['accuracy']:.2f}")
    print(f"Confusion Matrix:\n{result['conf_matrix']}")
    print(f"Classification Report:\n{result['class_report']}")
    print("-" * 80)
    
# Optionally, visualize confusion matrices
fig, axes = plt.subplots(4, 3, figsize=(15, 15))
axes = axes.flatten()
for idx, (model_name, result) in enumerate(results.items()):
    ax = axes[idx]
    ax.matshow(result['conf_matrix'], cmap='Blues', alpha=0.7)
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels(['0', '1'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
plt.tight_layout()
plt.show()


# In[54]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset (replace with your actual file path)
dataset_path = r"C:\Users\kurle\OneDrive\Desktop\credit_score_dataset.csv"  # Update with your actual file path
credit_data = pd.read_csv(dataset_path)

# Preprocess the data (create binary target column)
credit_data['Target'] = credit_data['CreditScore'].apply(lambda x: 1 if x > 600 else 0)
credit_data = credit_data.drop(columns=['CreditScore'])  # Drop original CreditScore

# Select features for classification
X = credit_data[['AnnualIncome', 'LoanAmount']].values  # Features (NumPy array)
y = credit_data['Target'].values  # Target (binary classification)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a function to evaluate the models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Fit the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    return accuracy, conf_matrix, class_report

# List of models to evaluate (without ANN)
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'CatBoost': CatBoostClassifier(learning_rate=0.1, iterations=500, depth=6, cat_features=[], verbose=False),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300),
}

# Create an empty dictionary to store results
results = {}

# Evaluate all models
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    accuracy, conf_matrix, class_report = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    results[model_name] = {
        'accuracy': accuracy,
        'conf_matrix': conf_matrix,
        'class_report': class_report
    }
    print(f"{model_name} Accuracy: {accuracy:.2f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")
    print("-" * 80)

# Displaying the confusion matrix plots
fig, axes = plt.subplots(4, 3, figsize=(15, 15))
axes = axes.flatten()
for idx, (model_name, result) in enumerate(results.items()):
    ax = axes[idx]
    cm = result['conf_matrix']
    
    # Plot the confusion matrix
    ax.matshow(cm, cmap='Blues', alpha=0.7)
    ax.set_title(f"{model_name} Confusion Matrix")
    ax.set_xticklabels(['0', '1'], fontsize=10)
    ax.set_yticklabels(['0', '1'], fontsize=10)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    
    # Display text inside the matrix
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', fontsize=12, color='black')

plt.tight_layout()
plt.show()


# In[ ]:




