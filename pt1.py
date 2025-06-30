import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset from seaborn (no need to download)
df = sns.load_dataset('titanic')

# Display basic info
print("Dataset Head:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())

# Drop columns with too many missing values
df = df.drop(['deck'], axis=1)

# Fill missing age with median
df['age'].fillna(df['age'].median(), inplace=True)

# Drop rows with missing 'embarked'
df.dropna(subset=['embarked'], inplace=True)

# Basic EDA Plots
sns.countplot(x='sex', data=df)
plt.title('Gender Distribution')
plt.show()

sns.histplot(df['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

sns.countplot(x='class', hue='sex', data=df)
plt.title('Passenger Class by Gender')
plt.show()

sns.boxplot(x='pclass', y='age', data=df)
plt.title('Age Distribution by Passenger Class')
plt.show()
