import pandas as pd
import matplotlib.pyplot as plt

# Sample categorical data
data = {'Gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male']}
df = pd.DataFrame(data)

# Count frequency
gender_counts = df['Gender'].value_counts()

# Plot bar chart
gender_counts.plot(kind='bar', color=['skyblue', 'pink'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
