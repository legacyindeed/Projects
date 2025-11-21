# Credit Risk EDA: Analyze default behavior across income, utilization, and customer segments.

#Section 1 - Load Data
import pandas as pd

full_data = pd.read_csv("credit_risk_dataset.csv")
print(full_data.head())
print(full_data.shape)


#Section2 - Basic Exploration

#Data Types
full_data.info()

#Summary of Variables
print(full_data.describe(include='object'))

#Checking for Nulls
print(full_data.isnull().sum())


#SECTION 3 - Clean Data

#Handle Missing Data
full_data = full_data.dropna()
print(full_data.isnull().sum())

#Remove Duplicates
full_data = full_data.drop_duplicates()


#SECTION 4 - Exploratory Analysis

import matplotlib.pyplot as plt

#Age Distribution
full_data["age"].hist(bins = 30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Counts")
plt.savefig('plots/age_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Income distribution
full_data['income'].hist(bins=30)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.savefig('plots/income_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Utilization Distribution
full_data['utilization_rate'].hist(bins=30)
plt.title("Utilization Rate Distribution")
plt.xlabel("Utilization")
plt.ylabel("Count")
plt.savefig('plots/utilization_distribution.png', dpi=300, bbox_inches='tight')
plt.show()


# Average income by education level
print(full_data.groupby("education_level")["income"].mean())

# Average income by spend_segment
print(full_data.groupby("spend_segment")["income"].mean())

# Correlation Heatmap
import seaborn as sns
plt.figure(figsize=(14, 12))
sns.heatmap(full_data.corr(numeric_only=True), cmap="coolwarm",vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.show()


# Section 5 Default Rate Analysis by Segments

# Default Rate by Income
full_data['income_bin'] = pd.qcut(full_data['income'], 5)
print(full_data.groupby('income_bin')['default_12m'].mean())

# Default Rate by Utilization
full_data['util_bin'] = pd.cut(full_data['utilization_rate'], bins=[0,0.2,0.4,0.6,0.8,1])
print(full_data.groupby('util_bin')['default_12m'].mean())

#Default Rate by Spend Segment
print(full_data.groupby('spend_segment')['default_12m'].mean())

#Analyzing default rates across income segments (Low, Mid, Upper Mid, High)
full_data['income_segment']=pd.qcut(full_data['income'], q=4, labels=['Low','Mid','Upper Mid','High'])
print(full_data.groupby('income_segment')['default_12m'].mean())



# SECTION 6
#What % of customers default
print(float(full_data['default_12m'].mean()))

#Difference in Features between defaulters and non-defaulters
print(full_data.groupby('default_12m')[['income','utilization_rate','delinquency_12m','inquiries_6m']].mean())



# SECTION 7 – Key Insights
# 1. Default rates drop significantly as income increases.
# 2. Customers with utilization >80% show ~20% default rate.
# 3. Defaulters have lower income, higher utilization, more delinquencies, and more credit inquiries.
# 4. Spend segments and education levels show moderate differences in income and risk.
