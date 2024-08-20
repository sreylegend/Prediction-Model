import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import TimeSeriesSplit
import pandas_datareader as pdr
import optuna
import ta  # Importing the ta library for technical analysis indicators
from datetime import datetime
from sklearn.metrics import r2_score

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

    return features, target, data

# Optuna objective function
def objective(trial, X_train_scaled, y_train_scaled):
    # Hyperparameters to tune
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e1)
    l1_ratio = trial.suggest_uniform('l1_ratio', 0.0, 1.0)

    # ElasticNet model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, random_state=42)

    # Cross-validation
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

        # Scale the data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

        # Run Optuna optimization specific to the selected ticker
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train_scaled, y_train_scaled), n_trials=50)

        # Best parameters from Optuna
        best_alpha = study.best_params['alpha']
        best_l1_ratio = study.best_params['l1_ratio']

        # Train the model using the best parameters
        model = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000, random_state=42)
        model.fit(X_train_scaled, y_train_scaled)

        # Predict on test set to calculate R-Squared
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        r_squared = r2_score(y_test, y_pred)

        # Predict the next two trading days
        last_features_scaled = scaler_X.transform(X[-1].reshape(1, -1))
        predicted_day1_price_scaled = model.predict(last_features_scaled)
        predicted_day1_price = scaler_y.inverse_transform(predicted_day1_price_scaled.reshape(-1, 1))[0, 0]

        # Update the features with the predicted price for the second prediction
        next_day_features = X[-1].copy()
        next_day_features[-2] = predicted_day1_price
        next_day_features_scaled = scaler_X.transform(next_day_features.reshape(1, -1))

        predicted_day2_price_scaled = model.predict(next_day_features_scaled)
        predicted_day2_price = scaler_y.inverse_transform(predicted_day2_price_scaled.reshape(-1, 1))[0, 0]

        # Get the dates for the next two trading days
        last_trading_date = pd.to_datetime(data.index[-1])
        next_trading_date_1 = last_trading_date + pd.offsets.BDay(1)
        next_trading_date_2 = last_trading_date + pd.offsets.BDay(2)

        # Display the predictions and R-Squared
        output_text = f"""
        R-Squared for {ticker}: {r_squared:.4f}\n
        Predicted Closing Price for {ticker} on {next_trading_date_1.strftime('%Y-%m-%d')}: ${predicted_day1_price:.2f}\n
        Predicted Closing Price for {ticker} on {next_trading_date_2.strftime('%Y-%m-%d')}: ${predicted_day2_price:.2f}
        """

        # Create the plot with predictions
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
    
    return "", go.Figure()

# Run the app
if __name__ == '__main__':
    app.run_server()
