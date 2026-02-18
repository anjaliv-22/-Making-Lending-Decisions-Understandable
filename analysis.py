import pandas as pd        
import matplotlib.pyplot as plt
import seaborn as sns

#!git clone https://github.com/anjaliv-22/-Making-Lending-Decisions-Understandable.git
#LC = pd.read_csv("-Making-Lending-Decisions-Understandable/loan_data.csv")
LC = pd.read_csv("loan_data.csv")

# Display first 5 rows
print("First 5 rows:")
LC.head()

print("\nDataset Info:")
LC.info()

#sum of rows with missing data for each column
missing_data = LC.isnull().sum().sort_values(ascending= False)
missing_data

print("\nStatistical Summary:")
LC.describe()

#Basic Analysis
print("\nAverage FICO score:")
print(LC["fico"].mean())

# Count how many loans fall into each purpose category
print("\nLoan purpose counts:")
print(LC["purpose"].value_counts())

print("\nPercentage of loans not fully paid:")
print(LC["not.fully.paid"].mean() * 100)

# Boxplot: Compare FICO score distributions across loan purposes
plt.figure(figsize=(10,6))
LC.boxplot(by='purpose', column='fico')
plt.xticks(rotation=45)
plt.title("FICO Score Distribution by Loan Purpose")
plt.suptitle("") 
plt.ylabel("FICO Score")
plt.show()

# Countplot: Show counts of loans that were fully paid vs not fully paid
plt.figure(figsize=(6,4))
sns.countplot(data=LC, x='not.fully.paid', hue='not.fully.paid', palette='coolwarm', legend=False)
plt.title("Loan Repayment Status")
plt.xlabel("Not Fully Paid (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()

# Boxplot: Compare FICO score distributions by Repayment Status

plt.figure(figsize=(8,5))
sns.boxplot(data=LC, x='not.fully.paid', y='fico')
plt.title("FICO Score vs Loan Repayment")
plt.show()

# Remove '%' and convert to float
LC['int.rate'] = LC['int.rate'].astype(str).str.rstrip('%').astype(float) / 100

# Histogram: Distribution of Interest Rates
#The KDE curve (pink line) overlays a smooth estimate of the density, highlighting the most common interest rates among borrowers.
plt.figure(figsize=(8,5))
sns.histplot(LC['int.rate'], bins=20, kde=True, color='pink')
plt.title("Distribution of Interest Rates")
plt.xlabel("Interest Rate (decimal)")
plt.show()

# Installment vs Default
# Analyze default risk by installment amount: quantile ranking, pivot table, line and scatter plots

'''Line plot: Shows how the default rate changes across increasing installment amounts (20 quantiles). Helps identify if bigger installments are riskier.

Scatter plot: Shows the direct relationship between average installment and default rate. Can visualize trends more clearly.'''

LC.columns = LC.columns.str.strip()

LC['installment.Rank'] = pd.qcut(LC['installment'], 20, labels=False)

installment_summary = pd.pivot_table(
    data=LC,
    index='installment.Rank',
    values=['installment', 'not.fully.paid'],
    aggfunc='mean'
)

# Line plot showing default rate by installment rank
plt.figure(figsize=(8,5))
sns.lineplot(data=installment_summary['not.fully.paid'], marker='o', color='purple')
plt.title("Loan Default Rate vs Installment Amount")
plt.xlabel("Installment Rank (Quantile)")
plt.ylabel("Mean Default Rate")
plt.grid(True)
plt.show()

# Scatter plot showing relationship between installment and default rate
plt.figure(figsize=(8,5))
installment_summary.plot.scatter(x='installment', y='not.fully.paid', color='green')
plt.title("Loan Default Rate vs Average Installment")
plt.xlabel("Average Installment ($)")
plt.ylabel("Mean Default Rate")
plt.grid(True)
plt.show()

# Interest Rate vs Default
# Analyze default risk by interest rate: quantile ranking, pivot table, line and scatter plots

'''Line plot: Higher interest rates usually lead to higher default rates. Quantile ranking makes the trend easier to see.

Scatter plot: Shows the direct correlation between average interest rate and default rate per group.'''

LC['int.rate.Rank'] = pd.qcut(LC['int.rate'], 20, labels=False, duplicates='drop')

int_summary = pd.pivot_table(
    data=LC,
    index='int.rate.Rank',
    values=['int.rate', 'not.fully.paid'],
    aggfunc='mean'
)

# Line plot for default rate by interest rate rank
plt.figure(figsize=(8,5))
sns.lineplot(data=int_summary['not.fully.paid'], marker='o', color='deeppink')
plt.title("Loan Default Rate vs Interest Rate")
plt.xlabel("Interest Rate Rank (Quantile)")
plt.ylabel("Mean Default Rate")
plt.grid(True)
plt.show()

# Scatter plot for interest rate vs default
plt.figure(figsize=(8,5))
int_summary.plot.scatter(x='int.rate', y='not.fully.paid', color='orange')
plt.title("Loan Default Rate vs Average Interest Rate")
plt.xlabel("Average Interest Rate (Decimal)")
plt.ylabel("Mean Default Rate")
plt.grid(True)
plt.show()

#FICO Score vs Default
# Analyze default risk by FICO score: quantile ranking, pivot table, line and scatter plots
'''Line plot: Clearly shows that lower FICO scores are associated with higher default rates. Quantiles smooth out the curve.

Scatter plot: Shows the negative correlation between FICO score and default rate.'''

LC['fico.Rank'] = pd.qcut(LC['fico'], 20, labels=False)

fico_summary = pd.pivot_table(
    data=LC,
    index='fico.Rank',
    values=['fico', 'not.fully.paid'],
    aggfunc='mean'
)

# Line plot for default rate by FICO score rank
plt.figure(figsize=(8,5))
sns.lineplot(data=fico_summary['not.fully.paid'], marker='o', color='teal')
plt.title("Loan Default Rate vs FICO Score")
plt.xlabel("FICO Score Rank (Quantile)")
plt.ylabel("Mean Default Rate")
plt.grid(True)
plt.show()

# Scatter plot for FICO vs default
plt.figure(figsize=(8,5))
fico_summary.plot.scatter(x='fico', y='not.fully.paid', color='blue')
plt.title("Loan Default Rate vs Average FICO Score")
plt.xlabel("Average FICO Score")
plt.ylabel("Mean Default Rate")
plt.grid(True)
plt.show()
