import pandas as pd

# Load the dataset
LC = pd.read_csv("loan_data.csv")

# Preview the data
print("First 5 rows:")
print(LC.head())

# Basic analysis
print("\nAverage FICO score:")
print(LC["fico"].mean())

print("\nLoan purpose counts:")
print(LC["purpose"].value_counts())

print("\nPercentage of loans not fully paid:")
print(LC["not.fully.paid"].mean() * 100)

import matplotlib.pyplot as plt

LC.boxplot(by = 'purpose', column = 'fico')

sns.countplot(data=LC, x='not.fully.paid', palette='coolwarm')

'''# Scatter plot: FICO vs Interest Rate
plt.scatter(LC["fico"], LC["int.rate"])
plt.xlabel("FICO Score")
plt.ylabel("Interest Rate (%)")
plt.title("FICO Score vs Interest Rate")

plt.show()'''
