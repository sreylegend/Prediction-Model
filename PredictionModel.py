import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit
import optuna
import ta
from datetime import datetime
from sklearn.metrics import r2_score
import numpy as np

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

# Function to fetch and prepare data
def get_data(ticker):
    start_date = "2015-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=end_date)

    # Add technical indicators, including MACD
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=14)
    data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])

    # MACD components
    data['MACD'], data['Signal_Line'], data['MACD_Hist'] = ta.trend.macd(data['Close']), ta.trend.macd_signal(data['Close']), ta.trend.macd_diff(data['Close'])

    # Prepare feature matrix and target variable
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Return'] = data['Close'].pct_change()
    data.dropna(inplace=True)

    features = data[['SMA_5', 'SMA_20', 'EMA_10', 'RSI', 'ATR', 'OBV', 'MACD', 'Signal_Line', 'MACD_Hist', 'Close_Lag1', 'Return']].values
    target = data['Close'].values

    return features, target, data

# Function to detect market regimes
def detect_regime(features, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    regimes = kmeans.fit_predict(features)
    return regimes

# Function to train a model for each regime
def train_models_by_regime(X_train_scaled, y_train_scaled, regimes):
    unique_regimes = np.unique(regimes)
    regime_models = {}
    
    for regime in unique_regimes:
        X_train_regime = X_train_scaled[regimes == regime]
        y_train_regime = y_train_scaled[regimes == regime]

        # Train a model for this regime using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train_regime, y_train_regime), n_trials=50)

        # Best parameters for this regime
        best_alpha = study.best_params['alpha']
        best_l1_ratio = study.best_params['l1_ratio']

        # Train the ElasticNet model for the regime
        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000, random_state=42)
        model.fit(X_train_regime, y_train_regime)
        regime_models[regime] = model
    
    return regime_models

# Function to train separate models for multi-step direct forecasting
def train_multi_step_models(X_train_scaled, y_train_scaled, steps):
    models = {}
    for step in range(1, steps + 1):
        # Prepare shifted target data
        y_train_step = np.roll(y_train_scaled, -step)[:len(y_train_scaled) - step]

        # Train a separate model for each step using Optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train_scaled[:-step], y_train_step), n_trials=50)

        # Best parameters for this step
        best_alpha = study.best_params['alpha']
        best_l1_ratio = study.best_params['l1_ratio']

        # Train the ElasticNet model for this step
        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000, random_state=42)
        model.fit(X_train_scaled[:-step], y_train_step)
        models[step] = model
    
    return models

# Optuna objective function
def objective(trial, X_train_scaled, y_train_scaled):
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e1)
    l1_ratio = trial.suggest_uniform('l1_ratio', 0.0, 1.0)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for train_index, test_index in tscv.split(X_train_scaled):
        X_train_fold, X_test_fold = X_train_scaled[train_index], X_train_scaled[test_index]
        y_train_fold, y_test_fold = y_train_scaled[train_index], y_train_scaled[test_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)
        scores.append(r2_score(y_test_fold, y_pred_fold))

    return sum(scores) / len(scores)

# Callback to update the output and graph
@app.callback(
    [Output('output-container', 'children'),
     Output('price-chart', 'figure')],
    [Input('submit-button', 'n_clicks')],
    [State('ticker-symbol', 'value')]
)
def update_dashboard(n_clicks, ticker):
    if n_clicks > 0:
        # Fetch data
        X, y, data = get_data(ticker)

        # Train/Test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Detect regimes
        regimes = detect_regime(X_train)

        # Scale the data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        # Train models by regime
        regime_models = train_models_by_regime(X_train_scaled, y_train_scaled, regimes)

        # Train multi-step direct forecasting models
        steps = 2  # Forecasting for 2 future days
        multi_step_models = train_multi_step_models(X_train_scaled, y_train_scaled, steps)

        # Predict on test set to calculate R-Squared (using the regime-specific model)
        last_regime = detect_regime(X_test)[-1]  # Detect regime of last training sample
        model = regime_models.get(last_regime, list(regime_models.values())[0])  # Use appropriate model for the regime
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        r_squared = r2_score(y_test, y_pred)

        # Multi-step direct forecasting
        forecasted_prices = []
        for step in range(1, steps + 1):
            model_step = multi_step_models[step]
            X_last_scaled = scaler_X.transform(X[-1].reshape(1, -1))
            pred_scaled = model_step.predict(X_last_scaled)
            forecasted_price = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            forecasted_prices.append(forecasted_price)

        # Get the dates for the next two trading days
        last_trading_date = pd.to_datetime(data.index[-1])
        future_dates = [last_trading_date + pd.offsets.BDay(i + 1) for i in range(steps)]

        # Display the predictions and R-Squared
        output_text = f"""
        R-Squared for {ticker}: {r_squared:.4f}\n
        Predicted Closing Prices for {ticker}:\n
        {future_dates[0].strftime('%Y-%m-%d')}: ${forecasted_prices[0]:.2f}\n
        {future_dates[1].strftime('%Y-%m-%d')}: ${forecasted_prices[1]:.2f}
        """

        # Create the plot with predictions
        price_chart = go.Figure()
        price_chart.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Close Prices', line=dict(color='lightblue')))
        price_chart.add_trace(go.Scatter(x=[data.index[-1]] + future_dates, y=[data['Close'][-1]] + forecasted_prices, mode='lines', name='Predicted Prices', line=dict(color='orange')))
        price_chart.update_layout(
            title=f"{ticker} Historical Prices and Predictions",
            xaxis_title="Date",
            yaxis_title="Price",
            paper_bgcolor='#2c2c2c',
            plot_bgcolor='#2c2c2c',
            font=dict(color='white')
        )

        return output_text, price_chart

    return "", go.Figure()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
