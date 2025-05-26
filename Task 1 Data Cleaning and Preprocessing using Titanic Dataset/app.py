import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Setting up the environment for visualizations
sns.set(style="whitegrid")

# 1. Importing and exploring the dataset
# Loading the Titanic dataset
df = pd.read_csv('Titanic-Dataset.csv')

# Displaying basic information about the dataset
print("Dataset Info:")
print(df.info())
print("\nNull Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

# Saving initial exploration plots
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.savefig('missing_values_heatmap.png')
plt.close()

# 2. Handling missing values
# Dropping 'Cabin' due to high percentage of missing values
df = df.drop(columns=['Cabin'])

# Imputing 'Age' with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Imputing 'Embarked' with mode
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Imputing 'Fare' with median
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Verifying no missing values remain
print("\nNull Values After Imputation:")
print(df.isnull().sum())

# 3. Converting categorical features to numerical
# Dropping non-informative columns
df = df.drop(columns=['PassengerId', 'Name', 'Ticket'])

# Label encoding for 'Sex'
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encoding for 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked')

# Converting boolean columns to integers
for col in ['Embarked_C', 'Embarked_Q', 'Embarked_S']:
    df[col] = df[col].astype(int)

# 4. Visualizing outliers using boxplots
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig('outliers_before_removal.png')
plt.close()

# Removing outliers using IQR method
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

# Visualizing data after outlier removal
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col} After Outlier Removal')
plt.tight_layout()
plt.savefig('outliers_after_removal.png')
plt.close()

# 5. Normalizing/standardizing numerical features
# Standardizing 'Age' and 'Fare'
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Normalizing 'SibSp' and 'Parch'
minmax_scaler = MinMaxScaler()
df[['SibSp', 'Parch']] = minmax_scaler.fit_transform(df[['SibSp', 'Parch']])

# Displaying final dataset info
print("\nFinal Dataset Info:")
print(df.info())
print("\nFinal Dataset Head:")
print(df.head())

# Saving the cleaned dataset
df.to_csv('Titanic_Dataset_Cleaned.csv', index=False)