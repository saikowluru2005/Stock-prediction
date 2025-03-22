from flask import Flask, render_template, request, jsonify
from nsetools import Nse
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)
nse = Nse()

def get_stock_prediction(symbol):
    try:
        symbol = symbol.strip().upper()
        yahoo_symbol = f"{symbol}.NS"
        
        stock = yf.Ticker(yahoo_symbol)
        info = stock.info
        
        if not info:
            return {'error': f'Could not find stock with symbol {symbol}'}

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            return {'error': f'No trading data available for {symbol}'}
            
        if len(df) < 30:
            return {
                'error': f'Limited data available for {symbol}. Found {len(df)} days, need at least 30 days.',
                'company_name': info.get('longName', symbol)
            }

        # Prepare historical data for the chart
        historical_data = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'prices': df['Close'].round(2).tolist()
        }

        df['Prediction'] = df['Close'].shift(-1)
        df['Days'] = range(len(df))
        
        X = np.array(df['Days'][:-1]).reshape(-1, 1)
        y = np.array(df['Close'][:-1])
        
        test_size = min(0.2, 1/len(X))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predict next 3 days
        next_days = np.array([len(df), len(df)+1, len(df)+2]).reshape(-1, 1)
        next_days_predictions = model.predict(next_days)
        
        # Get previous days' values
        prev_days_data = {
            'day_1': {
                'date': df.index[-2].strftime('%Y-%m-%d'),
                'close': round(df['Close'].iloc[-2], 2),
                'volume': int(df['Volume'].iloc[-2]),
                'high': round(df['High'].iloc[-2], 2),
                'low': round(df['Low'].iloc[-2], 2)
            },
            'day_2': {
                'date': df.index[-3].strftime('%Y-%m-%d'),
                'close': round(df['Close'].iloc[-3], 2),
                'volume': int(df['Volume'].iloc[-3]),
                'high': round(df['High'].iloc[-3], 2),
                'low': round(df['Low'].iloc[-3], 2)
            }
        }

        # Calculate daily changes
        current_price = df['Close'].iloc[-1]
        day_1_change = ((current_price - prev_days_data['day_1']['close']) / prev_days_data['day_1']['close']) * 100
        day_2_change = ((current_price - prev_days_data['day_2']['close']) / prev_days_data['day_2']['close']) * 100

        prev_days_data['day_1']['change'] = round(day_1_change, 2)
        prev_days_data['day_2']['change'] = round(day_2_change, 2)

        # Calculate technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # RSI calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Get latest indicators
        current_sma20 = df['SMA_20'].iloc[-1]
        current_sma50 = df['SMA_50'].iloc[-1]
        current_rsi = df['RSI'].iloc[-1]
        
        # Generate trading signals
        signals = []
        if current_price > current_sma20 and current_price > current_sma50:
            signals.append("Price above both moving averages (Bullish)")
        elif current_price < current_sma20 and current_price < current_sma50:
            signals.append("Price below both moving averages (Bearish)")
            
        if current_rsi > 70:
            signals.append("RSI indicates overbought conditions (Consider Selling)")
        elif current_rsi < 30:
            signals.append("RSI indicates oversold conditions (Consider Buying)")
            
        # Calculate average prediction trend
        pred_trend = (next_days_predictions[-1] - current_price) / current_price * 100
        
        # Generate recommendation
        if pred_trend > 5 and current_rsi < 60:
            recommendation = "Strong Buy"
            reason = "Strong upward trend predicted with reasonable RSI"
        elif pred_trend > 2 and current_rsi < 70:
            recommendation = "Buy"
            reason = "Moderate upward trend predicted"
        elif pred_trend < -5 or current_rsi > 70:
            recommendation = "Sell"
            reason = "Downward trend predicted or overbought conditions"
        elif pred_trend < -2:
            recommendation = "Consider Selling"
            reason = "Slight downward trend predicted"
        else:
            recommendation = "Hold"
            reason = "No strong trend detected"

        return {
            'current_price': round(current_price, 2),
            'predictions': [round(p, 2) for p in next_days_predictions],
            'company_name': info.get('longName', symbol),
            'symbol': symbol,
            'data_points': len(df),
            'day_high': df['High'].iloc[-1],
            'day_low': df['Low'].iloc[-1],
            'volume': int(df['Volume'].iloc[-1]),
            'prev_close': df['Close'].iloc[-2],
            'historical_data': historical_data,
            'technical_indicators': {
                'sma20': round(current_sma20, 2),
                'sma50': round(current_sma50, 2),
                'rsi': round(current_rsi, 2)
            },
            'signals': signals,
            'recommendation': recommendation,
            'reason': reason,
            'prev_days_data': prev_days_data
        }
        
    except Exception as e:
        return {'error': f'Error processing stock data: {str(e)}'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['GET'])
def search():
    term = request.args.get('term', '').upper()
    all_stocks = nse.get_stock_codes()
    matches = {k: v for k, v in all_stocks.items() if term in k.upper() or term in v.upper()}
    return jsonify(list(matches.items())[:10])

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['ticker']
    result = get_stock_prediction(symbol)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
