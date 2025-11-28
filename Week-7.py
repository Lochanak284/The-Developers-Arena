
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import scipy.stats as stats # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

data = {
    "Month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
    "Sales": [54000, 58000, 60000, 63000, 64000, 68000, 70000, 72000, 75000, 77000, 80000, 82000],
    "Marketing_Spend": [15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000]
}

df = pd.DataFrame(data)

print("\n=== DATASET CREATED IN CODE ===")
print(df)

print("\n=== SUMMARY STATISTICS ===")
print(df.describe())

corr_value = df['Marketing_Spend'].corr(df['Sales'])
print("\n=== CORRELATION BETWEEN MARKETING SPEND & SALES ===")
print(f"Correlation: {corr_value:.4f}")


benchmark_sales = 60000
t_stat, p_val = stats.ttest_1samp(df['Sales'], benchmark_sales)

print("\n=== HYPOTHESIS TEST (One-sample t-test) ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_val:.6f}")

if p_val < 0.05:
    print("❌ Reject H0: Sales significantly differ from benchmark.")
else:
    print("✔ Fail to reject H0: No significant difference.")

X = df[['Marketing_Spend']]
y = df['Sales']

model = LinearRegression()
model.fit(X, y)

print("\n=== LINEAR REGRESSION MODEL ===")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Slope: {model.coef_[0]:.4f}")

pred = model.predict([[30000]])
print(f"Predicted sales for 30000 marketing spend: {pred[0]:.2f}")

plt.scatter(df['Marketing_Spend'], df['Sales'], label="Actual Sales")
plt.plot(df['Marketing_Spend'], model.predict(X), label="Trendline")
plt.xlabel("Marketing Spend")
plt.ylabel("Sales")
plt.title("Marketing Spend vs Sales")
plt.legend()
plt.show()

plt.hist(df['Sales'], bins=6)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.show()
