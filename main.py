# Importing required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve

# Loading the dataset

df = pd.read_csv('creditcard.csv')

# Data preprocessing

df.drop(['Time'], axis=1, inplace=True) # Dropping the 'Time' column
scaler = StandardScaler() # Scaling the dataset
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1,1))

# Feature selection

X = df.drop(['Class'], axis=1)
y = df['Class']

# Splitting the dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model selection and training

lr = LogisticRegression() # Initializing the logistic regression model
lr.fit(X_train, y_train) # Training the model

# Model evaluation

y_pred = lr.predict(X_test) # Predicting the test data
conf_matrix = confusion_matrix(y_test, y_pred) # Creating a confusion matrix
print(conf_matrix)

# Printing the classification report

class_report = classification_report(y_test, y_pred)
print(class_report)

# Printing the accuracy score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# ROC curve and AUC score

y_pred_proba = lr.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
