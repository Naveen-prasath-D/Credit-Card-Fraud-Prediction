# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    matthews_corrcoef, confusion_matrix


# Define the load_data function to read the data from a CSV file
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


file_path = "ENTER THE FILE PATH"
data = load_data(file_path)

# Quick peek at the data
print("Shape of the dataset:", data.shape)
print("Description of the dataset:")
print(data.describe())

# Check the distribution of fraud cases
fraud_count = len(data[data['Class'] == 1])
valid_count = len(data[data['Class'] == 0])
outlier_fraction = fraud_count / valid_count
print("Fraud Cases:", fraud_count)
print("Valid Transactions:", valid_count)
print("Outlier Fraction:", outlier_fraction)

# Summary statistics of transaction amounts
print("Amount details of the fraudulent transactions:")
print(data[data['Class'] == 1]['Amount'].describe())
print("Amount details of the valid transactions:")
print(data[data['Class'] == 0]['Amount'].describe())

# Correlation matrix visualization
corrmat = data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()

# Split the data into features (X) and target variable (Y)
X = data.drop(['Class'], axis=1)
Y = data['Class']

# Split data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.2, random_state=42)

# Build and train the Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(xTrain, yTrain)

# Make predictions
yPred = rfc.predict(xTest)

# Evaluate the classifier
print("Random Forest Classifier Metrics:")
print("Accuracy:", accuracy_score(yTest, yPred))
print("Precision:", precision_score(yTest, yPred))
print("Recall:", recall_score(yTest, yPred))
print("F1-Score:", f1_score(yTest, yPred))
print("Matthews Correlation Coefficient:", matthews_corrcoef(yTest, yPred))

# Confusion matrix visualization
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()
