import pandas as pd

# Load the dataset
df = pd.read_csv("data/loans.csv")

# Preview the data
print("First 5 rows:")
print(df.head())

# Basic analysis
print("\nAverage FICO score:")
print(df["fico"].mean())

print("\nLoan purpose counts:")
print(df["purpose"].value_counts())

print("\nPercentage of loans not fully paid:")
print(df["not.fully.paid"].mean() * 100)
