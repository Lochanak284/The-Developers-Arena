import pandas as pd


data = pd.read_csv("sales_data.csv")  

print("First 5 Rows of Dataset:")
print(data.head())
print("-" * 50)

print("📗 Dataset Information:")
print(data.info())
print("-" * 50)

data['total_sales'] = data['units_sold'] * data['price']


total_sales_value = data['total_sales'].sum()


best_product = data.loc[data['total_sales'].idxmax(), 'product']

print("📊 SALES REPORT 📊")
print(f"Total Sales (All Products): ₹{total_sales_value}")
print(f"Best-Selling Product: {best_product}")
print("\nDetailed Sales Table:")
print(data)
