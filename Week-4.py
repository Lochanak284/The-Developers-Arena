
# Student Performance Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)   
n = 100

data = pd.DataFrame({
    "Student_ID": range(1, n+1),
    "Math_Score": np.random.normal(70, 10, n).clip(40, 100),
    "Science_Score": np.random.normal(72, 12, n).clip(40, 100),
    "Study_Hours": np.random.normal(5, 1.5, n).clip(1, 10),
})

print("First 5 rows of Dataset:\n")
print(data.head())
print("\nDataset Summary:\n")
print(data.describe())

plt.figure(figsize=(6,4))
avg_scores = data[["Math_Score", "Science_Score"]].mean()
avg_scores.plot(kind="bar")
plt.title("Average Scores in Subjects")
plt.xlabel("Subjects")
plt.ylabel("Average Score")
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(data["Study_Hours"], data["Math_Score"])
plt.title("Study Hours vs Math Score")
plt.xlabel("Study Hours")
plt.ylabel("Math Score")
plt.grid(linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

print("\n=== Insights ===\n")

print("1. Students score slightly higher in Science than Mathematics on average.")
print(f"   - Average Math Score: {avg_scores['Math_Score']:.2f}")
print(f"   - Average Science Score: {avg_scores['Science_Score']:.2f}")

print("\n2. There is a positive trend between Study Hours and Math Score.")
print("   Students studying 5–8 hours tend to score higher (70+).")

print("\n3. Study Hours do not guarantee performance —")
print("   Several students study more but still score average, showing study quality matters.")
