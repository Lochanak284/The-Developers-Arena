import os
from datetime import datetime, timedelta
import numpy as np # type: ignore
import pandas as pd # type: ignore

import dash # type: ignore
from dash import dcc, html, Input, Output, State # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore

DATA_CSV = "sales_data.csv"  

def generate_sample_data(n_customers=500, n_products=30, start_date="2024-01-01", end_date=None, n_orders=5000):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    customers = [f"CUST_{i:04d}" for i in range(1, n_customers+1)]
    products = [f"P{p:03d}" for p in range(1, n_products+1)]
    product_names = [f"Product {p}" for p in range(1, n_products+1)]
    product_catalog = pd.DataFrame({"product_id": products, "product_name": product_names})
    regions = ["North", "South", "East", "West"]
    segments = ["Consumer", "Corporate", "Home Office", "Small Business"]

    rng = np.random.default_rng(42)
    order_dates = pd.to_datetime(rng.integers(int(start.timestamp()), int(end.timestamp()), size=n_orders), unit='s')
    order_dates = pd.to_datetime(order_dates.date) 

    df = pd.DataFrame({
        "order_id": [f"ORD_{i:06d}" for i in range(1, n_orders+1)],
        "order_date": order_dates,
        "customer_id": rng.choice(customers, size=n_orders),
        "product_id": rng.choice(products, size=n_orders, p=None),
        "quantity": rng.integers(1, 6, size=n_orders),
        "unit_price": np.round(rng.uniform(5, 500, size=n_orders), 2),
        "region": rng.choice(regions, size=n_orders),
        "segment": rng.choice(segments, size=n_orders, p=[0.6,0.15,0.15,0.1])
    })
    df = df.merge(product_catalog, on="product_id", how="left")
    df["revenue"] = df["quantity"] * df["unit_price"]
    return df.sort_values("order_date").reset_index(drop=True)

def load_data():
    if os.path.exists(DATA_CSV):
        try:
            df = pd.read_csv(DATA_CSV, parse_dates=["order_date"])
            required = {"order_id","order_date","customer_id","product_id","product_name","quantity","unit_price","region","segment"}
            if not required.issubset(set(df.columns)):
                raise ValueError("CSV missing required columns. Using generated sample data.")
            df["revenue"] = df["quantity"] * df["unit_price"]
            return df
        except Exception as e:
            print("Failed to load CSV, generating sample data. Error:", e)
    
    return generate_sample_data()

df = load_data()

def compute_rfm(df, reference_date=None):
    if reference_date is None:
        reference_date = df["order_date"].max() + pd.Timedelta(days=1)
    agg = df.groupby("customer_id").agg({
        "order_date": lambda x: (reference_date - x.max()).days,
        "order_id": "nunique",
        "revenue": "sum"
    }).rename(columns={"order_date":"recency", "order_id":"frequency", "revenue":"monetary"}).reset_index()
    return agg

rfm = compute_rfm(df)

app = dash.Dash(__name__)
app.title = "Interactive Sales Dashboard"

min_date = df["order_date"].min()
max_date = df["order_date"].max()
product_options = [
    {"label": f"{row['product_name']} ({row['product_id']})", "value": row['product_id']}
    for row in df[["product_id","product_name"]].drop_duplicates().to_dict("records")
]
product_options = [
    {"label": f"{row['product_name']} ({row['product_id']})", "value": row['product_id']}
    for row in df[["product_id","product_name"]].drop_duplicates().to_dict("records")
]
region_options = [{"label": r, "value": r} for r in sorted(df["region"].unique())]
segment_options = [{"label": s, "value": s} for s in sorted(df["segment"].unique())]

app.layout = html.Div(style={"fontFamily":"Arial, sans-serif", "margin":"12px"}, children=[
    html.H2("Interactive Sales Dashboard"),
    html.Div([
        html.Div([
            html.Label("Date Range"),
            dcc.DatePickerRange(
                id="date-range",
                start_date=min_date,
                end_date=max_date,
                display_format="YYYY-MM-DD"
            ),
        ], style={"display":"inline-block", "marginRight":"20px", "verticalAlign":"top"}),
        html.Div([
            html.Label("Products (multi)"),
            dcc.Dropdown(id="product-dropdown", options=product_options, multi=True, placeholder="Select products"),
        ], style={"display":"inline-block", "width":"380px", "marginRight":"20px", "verticalAlign":"top"}),
        html.Div([
            html.Label("Region"),
            dcc.Dropdown(id="region-dropdown", options=[{"label":"All","value":"ALL"}]+region_options, value="ALL"),
        ], style={"display":"inline-block", "width":"180px", "marginRight":"20px", "verticalAlign":"top"}),
        html.Div([
            html.Label("Segment"),
            dcc.Dropdown(id="segment-dropdown", options=[{"label":"All","value":"ALL"}]+segment_options, value="ALL"),
        ], style={"display":"inline-block", "width":"200px", "verticalAlign":"top"}),
    ], style={"marginBottom":"18px"}),

    html.Div([
        html.Div([
            dcc.Graph(id="sales-trend"),
            html.Div(id="trend-stats", style={"padding":"6px", "fontSize":"13px"})
        ], style={"width":"65%", "display":"inline-block", "verticalAlign":"top", "paddingRight":"12px"}),

        html.Div([
            dcc.Graph(id="top-products"),
            html.Div([
                html.Label("Top N products"),
                dcc.Slider(id="top-n", min=3, max=20, step=1, value=8,
                           marks={3:"3",8:"8",12:"12",16:"16",20:"20"})
            ], style={"padding":"6px"})
        ], style={"width":"34%", "display":"inline-block", "verticalAlign":"top"}),
    ]),

    html.Div([
        html.Div([
            dcc.Graph(id="customer-segmentation"),
        ], style={"width":"48%", "display":"inline-block", "verticalAlign":"top"}),

        html.Div([
            dcc.Graph(id="rfm-scatter"),
            html.Div("R: Recency (days) — lower is better | F: Frequency (orders) | M: Monetary (revenue)", style={"fontSize":"12px", "color":"#666"})
        ], style={"width":"50%", "display":"inline-block", "verticalAlign":"top", "paddingLeft":"12px"}),
    ], style={"marginTop":"18px"}),

    html.Hr(),
    html.Div([
        html.Label("Upload CSV (optional) — must contain columns: order_id, order_date, customer_id, product_id, product_name, quantity, unit_price, region, segment"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={
                'width': '60%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
            },
            multiple=False
        ),
        html.Div(id='upload-status', style={"marginTop":"8px", "color":"green"})
    ])
])

def filter_df(df, start_date, end_date, products, region, segment):
    dff = df.copy()
    if start_date is not None:
        dff = dff[dff["order_date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        dff = dff[dff["order_date"] <= pd.to_datetime(end_date)]
    if products and len(products) > 0:
        dff = dff[dff["product_id"].isin(products)]
    if region and region != "ALL":
        dff = dff[dff["region"] == region]
    if segment and segment != "ALL":
        dff = dff[dff["segment"] == segment]
    return dff

@app.callback(
    Output("sales-trend", "figure"),
    Output("trend-stats", "children"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("product-dropdown", "value"),
    Input("region-dropdown", "value"),
    Input("segment-dropdown", "value")
)
def update_sales_trend(start_date, end_date, products, region, segment):
    dff = filter_df(df, start_date, end_date, products, region, segment)
    if dff.empty:
        fig = go.Figure()
        fig.update_layout(title="No data for selected filters")
        return fig, "No data"
    daily = dff.groupby("order_date").agg({"revenue":"sum", "order_id":"nunique"}).rename(columns={"order_id":"orders"}).reset_index()
    daily = daily.sort_values("order_date")
    daily["ma7"] = daily["revenue"].rolling(7, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["order_date"], y=daily["revenue"], mode="lines+markers", name="Daily Revenue"))
    fig.add_trace(go.Scatter(x=daily["order_date"], y=daily["ma7"], mode="lines", name="7-day MA"))
    fig.update_layout(title="Sales Trend (Revenue)", xaxis_title="Date", yaxis_title="Revenue", hovermode="x unified")
    total_revenue = dff["revenue"].sum()
    total_orders = dff["order_id"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders>0 else 0
    stats = f"Total Revenue: ₹{total_revenue:,.2f}  |  Orders: {total_orders}  |  Avg Order Value: ₹{avg_order_value:,.2f}"
    return fig, stats

@app.callback(
    Output("top-products", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("region-dropdown", "value"),
    Input("segment-dropdown", "value"),
    Input("top-n", "value"),
)
def update_top_products(start_date, end_date, region, segment, top_n):
    dff = filter_df(df, start_date, end_date, None, region, segment)
    if dff.empty:
        fig = go.Figure()
        fig.update_layout(title="No data")
        return fig
    prod = dff.groupby(["product_id", "product_name"]).agg({"revenue":"sum", "quantity":"sum"}).reset_index()
    prod = prod.sort_values("revenue", ascending=False).head(top_n)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=prod["product_name"], y=prod["revenue"], name="Revenue"))
    fig.add_trace(go.Bar(x=prod["product_name"], y=prod["quantity"], name="Quantity", yaxis="y2"))
    fig.update_layout(
        title=f"Top {top_n} Products (by revenue)",
        xaxis_tickangle=-45,
        yaxis=dict(title="Revenue"),
        yaxis2=dict(title="Quantity", overlaying="y", side="right"),
        legend=dict(x=0.01, y=0.99)
    )
    return fig

@app.callback(
    Output("customer-segmentation", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("product-dropdown", "value"),
    Input("region-dropdown", "value"),
)
def update_customer_seg(start_date, end_date, products, region):
    dff = filter_df(df, start_date, end_date, products, region, None)
    if dff.empty:
        fig = go.Figure()
        fig.update_layout(title="No data")
        return fig
    dff["product_category"] = dff["product_name"].str.split().str[0]
    agg = dff.groupby(["region","segment","product_category"]).agg({"customer_id":"nunique","revenue":"sum"}).reset_index().rename(columns={"customer_id":"unique_customers"})
    fig = px.sunburst(agg, path=["region","segment","product_category"], values="revenue",
                      hover_data=["unique_customers"], title="Revenue distribution by Region → Segment → Product Category")
    return fig

@app.callback(
    Output("rfm-scatter", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("region-dropdown", "value"),
    Input("segment-dropdown", "value"),
)
def update_rfm(start_date, end_date, region, segment):
    dff = filter_df(df, start_date, end_date, None, region, segment)
    if dff.empty:
        fig = go.Figure()
        fig.update_layout(title="No data")
        return fig
    rfm_local = compute_rfm(dff, reference_date=dff["order_date"].max()+pd.Timedelta(days=1))
    customer_seg = dff.groupby("customer_id")["segment"].agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]).reset_index()
    rfm_local = rfm_local.merge(customer_seg, on="customer_id", how="left")
    fig = px.scatter(rfm_local, x="recency", y="frequency", size="monetary", color="segment",
                     hover_data=["customer_id","monetary"], title="Customer RFM Scatter (Size=Monetary)")
    fig.update_layout(xaxis_title="Recency (days)", yaxis_title="Frequency (orders)")
    return fig

import base64
import io

@app.callback(
    Output("upload-status", "children"),
    Output("product-dropdown", "options"),
    Output("region-dropdown", "options"),
    Output("segment-dropdown", "options"),
    Input("upload-data", "contents"),
    State("upload-data", "filename")
)
def handle_upload(contents, filename):
    global df
    if contents is None:
        return "Using built-in dataset. Place a CSV named 'sales_data.csv' in the app folder or upload one here.", product_options, [{"label":"All","value":"ALL"}]+region_options, [{"label":"All","value":"ALL"}]+segment_options
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        uploaded = pd.read_csv(io.StringIO(decoded.decode('utf-8')), parse_dates=["order_date"])
        required = {"order_id","order_date","customer_id","product_id","product_name","quantity","unit_price","region","segment"}
        if not required.issubset(set(uploaded.columns)):
            raise ValueError("CSV missing required columns.")
        uploaded["revenue"] = uploaded["quantity"] * uploaded["unit_price"]
        df = uploaded.sort_values("order_date").reset_index(drop=True)
        prod_opts = [
            {"label": f"{row['product_name']} ({row['product_id']})", "value": row['product_id']}
            for row in df[["product_id","product_name"]].drop_duplicates().to_dict("records")
        ]
        reg_opts = [{"label": r, "value": r} for r in sorted(df["region"].unique())]
        seg_opts = [{"label": s, "value": s} for s in sorted(df["segment"].unique())]
        return f"Uploaded '{filename}' successfully. Dataset loaded with {len(df):,} rows.", prod_opts, [{"label":"All","value":"ALL"}]+reg_opts, [{"label":"All","value":"ALL"}]+seg_opts
    except Exception as e:
        print("Upload error:", e)
        return f"Failed to parse uploaded file: {e}", product_options, [{"label":"All","value":"ALL"}]+region_options, [{"label":"All","value":"ALL"}]+segment_options

if __name__ == "__main__":
    app.run(debug=True)

