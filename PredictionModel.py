import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
import ta
import pandas_datareader as pdr
from datetime import datetime

# Initialize the Dash app with a dark Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# App layout with dark theme and loading spinner
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Stock Price Prediction Dashboard", className="text-center mb-4", style={'color': 'white'})),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("Enter Ticker Symbol:", style={'color': 'white'}),
            dbc.Input(id='ticker-symbol', type='text', value='AAPL', className="mb-3"),
            dbc.Button("Submit", id='submit-button', n_clicks=0, color="primary", className="mb-4"),
            dcc.Loading(
                id="loading-output",
                type="circle",
                children=html.Div(id='output-container', style={'color': 'white', 'white-space': 'pre-line'})
            )
        ], width=4),
        dbc.Col([
            dcc.Loading(
                id="loading-graph",
                type="circle",
                children=dcc.Graph(id='price-chart')
            )
        ], width=8)
    ]),
], fluid=True, style={'background-color': '#2c2c2c'})

# Callback to update the output based on ticker symbol input and submit button click
@app.callback(
    [Output('output-container', 'children'),
     Output('price-chart', 'figure')],
    [Input('submit-button', 'n_clicks')],
    [State('ticker-symbol', 'value')]
)
def update_dashboard(n_clicks, ticker):
    if n_clicks > 0:
        # Fetch stock data
        start_date = "2015-01-01"
        end_date = datetime.today().strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Incorporate macroeconomic data
        macro_data = pdr.get_data_fred(['T10Y2Y', 'UNRATE', 'CPIAUCSL', 'DGS10', 'MORTGAGE30US'], start=start_date, end=end_date)
        data = data.join(macro_data, how='inner')
        data.fillna(method='ffill', inplace=True)
        
        # Add technical indicators
        data['SMA_5'] = data['Close'].rolling(window=5).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
        data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
        data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)
        data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        
        # Prepare feature matrix and target variable
        data['Close_Lag1'] = data['Close'].shift(1)
        data['Return'] = data['Close'].pct_change()
        data.dropna(inplace=True)
        
        features = data[['SMA_5', 'SMA_20', 'EMA_10', 'RSI', 'ATR', 'OBV', 'T10Y2Y', 'UNRATE', 'CPIAUCSL', 'DGS10', 'MORTGAGE30US', 'Close_Lag1', 'Return']].values
        target = data['Close'].values
        
        # Split data into training and testing sets
        train_size = int(len(features) * 0.8)
        X_train, X_test = features[:train_size], features[train_size:]
        y_train, y_test = target[:train_size], target[train_size:]
        
        # Scale the data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        # Train ElasticNet model
        param_grid_en = {'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]}
        elastic_net = ElasticNet(random_state=42, max_iter=10000)
        grid_search_en = GridSearchCV(estimator=elastic_net, param_grid=param_grid_en, n_jobs=-1, cv=5)
        grid_search_en.fit(X_train_scaled, y_train_scaled)
        best_en_model = grid_search_en.best_estimator_
        
        # Predict the next two trading days
        last_features_scaled = scaler_X.transform(features[-1].reshape(1, -1))
        predicted_day1_price_en_scaled = best_en_model.predict(last_features_scaled)
        predicted_day1_price = scaler_y.inverse_transform(predicted_day1_price_en_scaled.reshape(-1, 1))[0, 0]
        
        # Use the prediction for the first trading day to help predict the second trading day
        next_day_features = features[-1].copy()
        next_day_features[-2] = predicted_day1_price  # Update 'Close_Lag1' with the predicted price
        next_day_features_scaled = scaler_X.transform(next_day_features.reshape(1, -1))
        
        predicted_day2_price_en_scaled = best_en_model.predict(next_day_features_scaled)
        predicted_day2_price = scaler_y.inverse_transform(predicted_day2_price_en_scaled.reshape(-1, 1))[0, 0]
        
        # Calculate next two trading day dates
        last_trading_date = pd.to_datetime(data.index[-1])
        next_trading_date_1 = last_trading_date + pd.offsets.BDay(1)
        next_trading_date_2 = last_trading_date + pd.offsets.BDay(2)
        
        # Create output text (each piece of information on a separate line, including next two trading day dates)
        output_text = f"""
        Best R-Squared: {grid_search_en.best_score_:.4f}\n
        Predicted Closing Price for {ticker} on {next_trading_date_1.strftime('%Y-%m-%d')}: ${predicted_day1_price:.2f}\n
        Predicted Closing Price for {ticker} on {next_trading_date_2.strftime('%Y-%m-%d')}: ${predicted_day2_price:.2f}
        """
        
        # Plot historical data with predictions for the next two days using dark background
        price_chart = go.Figure()
        price_chart.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Close Prices', line=dict(color='lightblue')))
        price_chart.add_trace(go.Scatter(x=[data.index[-1], next_trading_date_1], y=[data['Close'][-1], predicted_day1_price], mode='lines', name='Predicted Price Day 1', line=dict(color='yellow')))
        price_chart.add_trace(go.Scatter(x=[next_trading_date_1, next_trading_date_2], y=[predicted_day1_price, predicted_day2_price], mode='lines', name='Predicted Price Day 2', line=dict(color='orange')))
        price_chart.update_layout(
            title=f"{ticker} Historical Prices and Predictions",
            xaxis_title="Date",
            yaxis_title="Price",
            paper_bgcolor='#2c2c2c',
            plot_bgcolor='#2c2c2c',
            font=dict(color='white')
        )
        
        return output_text, price_chart
    
    # Default empty chart if no submit
    return "", go.Figure()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080)

