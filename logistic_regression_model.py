import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

#!git clone https://github.com/anjaliv-22/-Making-Lending-Decisions-Understandable.git
#LC = pd.read_csv("-Making-Lending-Decisions-Understandable/loan_data.csv")

LC = pd.read_csv("loan_data.csv")

LC.columns = LC.columns.str.strip()
LC['int.rate'] = LC['int.rate'].astype(str).str.rstrip('%').astype(float) / 100


X = LC[['fico', 'int.rate', 'installment', 'log.annual.inc']]
y = LC['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Logistic Regression")
plt.show()
