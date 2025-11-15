
import pandas as pd # type: ignore
import plotly.express as px # type: ignore
from dash import Dash, html, dcc # type: ignore

# ----------------------------
# 1️⃣ Load or Create Sample Data
# ----------------------------

# You can replace this with your own CSV file if you have real data
data = {
    'CustomerID': [101, 102, 103, 101, 104, 105, 102, 106, 107, 101, 103, 102],
    'CustomerName': ['Aarav', 'Diya', 'Karan', 'Aarav', 'Riya', 'Mohan', 'Diya', 'Sneha', 'Rahul', 'Aarav', 'Karan', 'Diya'],
    'Product': ['Laptop', 'Headphones', 'Keyboard', 'Mouse', 'Monitor', 'Laptop', 'Mouse', 'Keyboard', 'Laptop', 'Headphones', 'Monitor', 'Laptop'],
    'Quantity': [1, 2, 1, 3, 1, 1, 2, 1, 1, 1, 2, 1],
    'UnitPrice': [60000, 2000, 1500, 800, 12000, 60000, 800, 1500, 60000, 2000, 12000, 60000],
    'Date': pd.to_datetime([
        '2025-01-05', '2025-01-06', '2025-01-07', '2025-01-10', '2025-01-12',
        '2025-01-15', '2025-01-18', '2025-01-19', '2025-01-21', '2025-01-25', '2025-01-28', '2025-01-29'
    ])
}

df = pd.DataFrame(data)

# Calculate Total Sales per order
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# ----------------------------
# 2️⃣ Data Analysis
# ----------------------------

# Total sales by customer
customer_sales = df.groupby('CustomerName')['TotalAmount'].sum().reset_index().sort_values(by='TotalAmount', ascending=False)

# Top 5 customers
top_customers = customer_sales.head(5)

# Monthly sales trend
df['Month'] = df['Date'].dt.to_period('M').astype(str)
monthly_sales = df.groupby('Month')['TotalAmount'].sum().reset_index()

# Product-wise sales
product_sales = df.groupby('Product')['TotalAmount'].sum().reset_index()

# ----------------------------
# 3️⃣ Dash App for Dashboard
# ----------------------------

app = Dash(__name__)
app.title = "Customer Sales Analysis Dashboard"

app.layout = html.Div([
    html.H1("📊 Customer Sales Analysis Dashboard", style={'textAlign': 'center'}),

    html.H3("Top 5 Customers"),
    dcc.Graph(
        figure=px.bar(top_customers,
                      x='CustomerName', y='TotalAmount',
                      color='CustomerName',
                      title="Top 5 Customers by Sales",
                      text_auto=True)
    ),

    html.H3("Monthly Sales Trend"),
    dcc.Graph(
        figure=px.line(monthly_sales,
                       x='Month', y='TotalAmount',
                       title="Sales Trend Over Time",
                       markers=True)
    ),

    html.H3("Product-wise Sales"),
    dcc.Graph(
        figure=px.pie(product_sales,
                      names='Product', values='TotalAmount',
                      title="Sales Distribution by Product")
    ),

    html.H3("Customer Purchase Patterns"),
    dcc.Graph(
        figure=px.scatter(df,
                          x='Date', y='TotalAmount',
                          color='CustomerName',
                          size='TotalAmount',
                          hover_data=['Product'],
                          title="Customer Purchase Patterns Over Time")
    )
])

# ----------------------------
# 4️⃣ Run the Dashboard
# ----------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
