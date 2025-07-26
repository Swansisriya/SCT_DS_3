import pandas as pd
import numpy as np
import seaborn  as  sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("bank_data.csv")  # Replace with your actual file path

# Display basic info
print("📊 Dataset Overview:")
print(df.info())
print("\n🧾 Sample Data:")
print(df.head())

# Replace 'unknown' values with NaN for easier analysis
df.replace('unknown', np.nan, inplace=True)

# Check for missing values
print("\n🔍 Missing Values:")
print(df.isnull().sum())

# Handle missing values: either drop or impute
df.dropna(inplace=True)  # Or use df.fillna(method='ffill'), based on strategy

# Encode categorical variables
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Correlation matrix
plt.figure(figsize=(12,8))
sns.heatmap(df_encoded.corr(), cmap='coolwarm', annot=False)
plt.title("🔗 Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# Class distribution for target
sns.countplot(data=df, x='y')
plt.title("🎯 Term Deposit Subscription Distribution")
plt.show()

# Save clean dataset
df_encoded.to_csv('bank_clean.csv', index=False)
print("✅ Cleaned dataset saved as 'bank_clean.csv'.")
