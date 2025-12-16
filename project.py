import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.templates.default = "plotly_dark"
figures_html = {} 

# load S&P 500
sp500 = pd.read_csv('SP500.csv')
sp500['SP500'] = pd.to_numeric(sp500['SP500'], errors='coerce')
sp500.dropna(inplace=True)
sp500['observation_date'] = pd.to_datetime(sp500['observation_date'])
sp500['Year'] = sp500['observation_date'].dt.year
sp500['Daily_Return'] = sp500['SP500'].pct_change()
sp500_annual = sp500.groupby('Year').agg(
    Year_Close=('SP500', 'last'),
    Daily_Std=('Daily_Return', 'std')
).reset_index()
sp500_annual['SP500_Return'] = sp500_annual['Year_Close'].pct_change() * 100
sp500_annual['SP500_Volatility'] = sp500_annual['Daily_Std'] * np.sqrt(252) * 100

# load economic data
gdp = pd.read_csv('GDP (1).csv')
gdp['observation_date'] = pd.to_datetime(gdp['observation_date'])
gdp['Year'] = gdp['observation_date'].dt.year
gdp_annual = gdp.groupby('Year')['GDP'].mean().reset_index()
gdp_annual['GDP_Growth'] = gdp_annual['GDP'].pct_change() * 100

cpi = pd.read_csv('FPCPITOTLZGUSA (1).csv')
cpi['observation_date'] = pd.to_datetime(cpi['observation_date'])
cpi['Year'] = cpi['observation_date'].dt.year
cpi = cpi.rename(columns={'FPCPITOTLZGUSA': 'Inflation'})
cpi_annual = cpi[['Year', 'Inflation']]

fed = pd.read_csv('FEDFUNDS (1).csv')
fed['observation_date'] = pd.to_datetime(fed['observation_date'])
fed['Year'] = fed['observation_date'].dt.year
fed_annual = fed.groupby('Year')['FEDFUNDS'].mean().reset_index()
fed_annual = fed_annual.rename(columns={'FEDFUNDS': 'Fed_Funds_Rate'})

unemp = pd.read_csv('UNRATE (1).csv')
unemp['observation_date'] = pd.to_datetime(unemp['observation_date'])
unemp['Year'] = unemp['observation_date'].dt.year
unemp_annual = unemp.groupby('Year')['UNRATE'].mean().reset_index()
unemp_annual = unemp_annual.rename(columns={'UNRATE': 'Unemployment_Rate'})

# f. stocks
stocks = pd.read_csv('all_stocks_5yr.csv')
stocks['date'] = pd.to_datetime(stocks['date'])
stocks['Year'] = stocks['date'].dt.year
stock_annual = stocks.groupby(['Name', 'Year']).agg(
    Annual_Return=('close', lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) > 0 else np.nan)
).reset_index()

df = sp500_annual.merge(gdp_annual, on='Year', how='inner')
df = df.merge(cpi_annual, on='Year', how='inner')
df = df.merge(fed_annual, on='Year', how='inner')
df = df.merge(unemp_annual, on='Year', how='inner')
df['Economy_Status'] = df['GDP_Growth'].apply(lambda x: 'Recession' if x < 0 else 'Expansion')
df['Unemployment_Change'] = df['Unemployment_Rate'].diff()
df['Unemp_Direction'] = df['Unemployment_Change'].apply(lambda x: 'Unemployment Rose' if x > 0 else 'Unemployment Dropped')
df['Lagged_GDP_Growth'] = df['GDP_Growth'].shift(1)
df['Rate_Category'] = df['Fed_Funds_Rate'].apply(lambda x: 'High Interest Rates' if x > df['Fed_Funds_Rate'].median() else 'Low Interest Rates')
df_clean = df.dropna().reset_index(drop=True)

print("Data processing complete. Generating figures...")

# function to convert fig to HTML div string
def fig_to_html(fig):
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)

# Viz 1
fig1 = px.line(df_clean, x='GDP_Growth', y='SP500_Return', text='Year', markers=True, title="1. GDP Growth vs S&P 500 Returns")
fig1.update_traces(textposition="bottom right")
figures_html['fig1'] = fig_to_html(fig1)

# Viz 2
fig2 = px.box(df_clean, x='Economy_Status', y='SP500_Return', color='Economy_Status', points="all", color_discrete_sequence=['#FF4136', '#2ECC40'], title="2. Recession vs Expansion")
figures_html['fig2'] = fig_to_html(fig2)

# Viz 3
fig3 = px.scatter(df_clean, x='Inflation', y='SP500_Return', trendline="ols", color="Fed_Funds_Rate", color_continuous_scale="Viridis", title="3. Inflation vs Returns")
figures_html['fig3'] = fig_to_html(fig3)

# Viz 4
sp500_monthly = sp500.set_index('observation_date').resample('M')['Daily_Return'].std().reset_index()
sp500_monthly['Monthly_Volatility'] = sp500_monthly['Daily_Return'] * np.sqrt(252) * 100
sp500_monthly['YearMonth'] = sp500_monthly['observation_date'].dt.to_period('M').astype(str)
unemp_monthly = unemp.copy()
unemp_monthly['YearMonth'] = unemp_monthly['observation_date'].dt.to_period('M').astype(str)
cpi_monthly = cpi.set_index('observation_date').resample('M').asfreq().interpolate(method='linear').reset_index()
cpi_monthly['YearMonth'] = cpi_monthly['observation_date'].dt.to_period('M').astype(str)

# Viz 5 - 14
fig5 = px.box(df_clean, x='Rate_Category', y='SP500_Return', color='Rate_Category', points="all", title="5. High vs Low Interest Rates")
figures_html['fig5'] = fig_to_html(fig5)

fig6 = px.scatter(df_clean, x='Fed_Funds_Rate', y='SP500_Volatility', color="Year", trendline="lowess", title="6. Fed Funds vs Volatility")
figures_html['fig6'] = fig_to_html(fig6)

fig7 = px.scatter(df_clean, x='Unemployment_Change', y='SP500_Return', color="Unemployment_Change", color_continuous_scale="RdBu_r", size_max=15, title="7. Unemployment Change vs Returns")
figures_html['fig7'] = fig_to_html(fig7)

fig8 = px.violin(df_clean, x='Unemp_Direction', y='SP500_Return', color='Unemp_Direction', box=True, points="all", title="8. Unemployment Direction")
figures_html['fig8'] = fig_to_html(fig8)

fig9 = px.scatter(df_clean, x='Lagged_GDP_Growth', y='SP500_Return', trendline="ols", color="Year", title="9. Predictive Power (Lagged GDP)")
figures_html['fig9'] = fig_to_html(fig9)

cols_to_corr = ['GDP_Growth', 'Inflation', 'Fed_Funds_Rate', 'Unemployment_Rate', 'SP500_Return', 'SP500_Volatility']
corr_matrix = df_clean[cols_to_corr].corr()
fig10 = px.imshow(corr_matrix, text_auto='.2f', aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1, title="10. Correlation Matrix")
figures_html['fig10'] = fig_to_html(fig10)

fig11 = px.scatter_3d(df_clean, x='GDP_Growth', y='Inflation', z='SP500_Return', color='Year', size='SP500_Volatility', title="11. The Economic Cube")
figures_html['fig11'] = fig_to_html(fig11)

years_overlap = list(set(df_clean['Year']) & set(stock_annual['Year']))
stock_viz_data = stock_annual[stock_annual['Year'].isin(years_overlap)]
market_viz_data = df_clean[df_clean['Year'].isin(years_overlap)]
fig12 = go.Figure()
fig12.add_trace(go.Box(x=stock_viz_data['Year'], y=stock_viz_data['Annual_Return'], name='Individual Stocks', marker_color='#00F0FF', boxpoints='outliers'))
fig12.add_trace(go.Scatter(x=market_viz_data['Year'], y=market_viz_data['SP500_Return'], mode='lines+markers', name='S&P 500 Index', line=dict(color='#FF00FF', width=4)))
fig12.update_layout(title="12. Market Breadth")
figures_html['fig12'] = fig_to_html(fig12)

sp500_monthly_candles = sp500.set_index('observation_date').resample('M')['SP500'].agg(['first', 'max', 'min', 'last'])
sp500_monthly_candles.columns = ['Open', 'High', 'Low', 'Close']
fig13 = go.Figure(data=[go.Candlestick(x=sp500_monthly_candles.index, open=sp500_monthly_candles['Open'], high=sp500_monthly_candles['High'], low=sp500_monthly_candles['Low'], close=sp500_monthly_candles['Close'])])
fig13.update_layout(title='13. S&P 500 Monthly Price Action', yaxis_title='S&P 500', xaxis_rangeslider_visible=False)
figures_html['fig13'] = fig_to_html(fig13)

selected_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'JPM']
selected_stocks = stocks[stocks['Name'].isin(selected_tickers)][['date', 'Name', 'close']]
stocks_wide = selected_stocks.pivot(index='date', columns='Name', values='close').reset_index()
sp500_daily = sp500[['observation_date', 'SP500']].rename(columns={'observation_date': 'date'})
comparison_df = pd.merge(stocks_wide, sp500_daily, on='date', how='inner').sort_values('date')
normalized_df = comparison_df.set_index('date')
normalized_df = (normalized_df / normalized_df.iloc[0]) * 100
corr_with_sp500 = normalized_df.corr()['SP500'].drop('SP500')
fig14 = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.1, subplot_titles=("Trend Comparison (Base 100)", "Correlation with S&P 500"))
fig14.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df['SP500'], name='S&P 500', line=dict(color='#FF00FF', width=4)), row=1, col=1)
colors = ['cyan', 'orange', 'yellow', 'lime', 'white']
for i, ticker in enumerate(selected_tickers):
    if ticker in normalized_df.columns:
        fig14.add_trace(go.Scatter(x=normalized_df.index, y=normalized_df[ticker], name=ticker, line=dict(width=1, color=colors[i % len(colors)])), row=1, col=1)
fig14.add_trace(go.Bar(x=corr_with_sp500.index, y=corr_with_sp500.values, marker_color=corr_with_sp500.values, marker_colorscale='Viridis', text=np.round(corr_with_sp500.values, 2), textposition='auto', name='Correlation'), row=2, col=1)
fig14.update_layout(title="14. S&P 500 Proxy Analysis")
figures_html['fig14'] = fig_to_html(fig14)

print("Constructing HTML...")

html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Economic Indicators Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            background-color: #111111;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max_width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            font-size: 3rem;
            color: #00F0FF;
            margin-bottom: 10px;
        }}
        p.subtitle {{
            text-align: center;
            color: #aaaaaa;
            font-size: 1.2rem;
            margin-bottom: 40px;
        }}
        .chart-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .chart-full {{
            margin-bottom: 30px;
        }}
        .card {{
            background-color: #1e1e1e;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        /* Responsive: stack columns on mobile */
        @media (max-width: 900px) {{
            .chart-row {{ grid-template-columns: 1fr; }}
        }}
        .section-title {{
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
            margin-top: 50px;
            margin-bottom: 20px;
            font-size: 1.8rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>US Economic Health Monitor</h1>
        <p class="subtitle">Historical analysis of Market Returns vs GDP, Inflation, and Rates</p>
        
        <div class="section-title">1. The Macro View</div>
        <div class="chart-row">
            <div class="card">{figures_html['fig1']}</div>
            <div class="card">{figures_html['fig2']}</div>
        </div>

        <div class="section-title">2. Inflation & Interest Rates</div>
        <div class="chart-row">
            <div class="card">{figures_html['fig3']}</div>
            <div class="card">{figures_html['fig6']}</div>
        </div>

        <div class="chart-row">
            <div class="card">{figures_html['fig5']}</div>
            <div class="card">{figures_html['fig11']}</div>
        </div>

        <div class="section-title">3. Market Breadth & Correlations</div>
        <div class="chart-full card">{figures_html['fig10']}</div>
        
        <div class="chart-row">
            <div class="card">{figures_html['fig9']}</div>
            <div class="card">{figures_html['fig7']}</div>
        </div>
        
        <div class="chart-full card">{figures_html['fig12']}</div>
        <div class="chart-full card">{figures_html['fig13']}</div>
        <div class="chart-full card">{figures_html['fig14']}</div>

        <p style="text-align:center; margin-top:50px; color:#555;">Generated with Python & Plotly</p>
    </div>
</body>
</html>
"""

with open("dashboard.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("Success! Open 'dashboard.html' in your browser.")