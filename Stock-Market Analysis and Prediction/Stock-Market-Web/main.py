 # removed unused import 'time'
import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime
import re
from Tweet import Tweet
import requests
import warnings
import os
import time
import difflib
from difflib import SequenceMatcher


def mean_squared_error(y_true, y_pred):
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(y_true_arr), len(y_pred_arr))
    if n == 0:
        return float('inf')
    diff = y_true_arr[:n] - y_pred_arr[:n]
    return float(np.mean(diff ** 2))


class MinMaxScalerLite:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def fit(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.data_min_ = arr.min(axis=0)
        self.data_max_ = arr.max(axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        self.data_range_[self.data_range_ == 0] = 1.0
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        original_shape = arr.shape
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        low, high = self.feature_range
        scaled = (arr - self.data_min_) / self.data_range_
        scaled = scaled * (high - low) + low
        if len(original_shape) == 1:
            return scaled.reshape(-1)
        return scaled

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        arr = np.asarray(X, dtype=float)
        original_shape = arr.shape
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        low, high = self.feature_range
        unscaled = (arr - low) / (high - low)
        unscaled = unscaled * self.data_range_ + self.data_min_
        if len(original_shape) == 1:
            return unscaled.reshape(-1)
        return unscaled


class StandardScalerLite:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.mean_ = arr.mean(axis=0)
        self.scale_ = arr.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float)
        original_shape = arr.shape
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        out = (arr - self.mean_) / self.scale_
        if len(original_shape) == 1:
            return out.reshape(-1)
        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LinearRegressionLite:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = 0.0

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        ones = np.ones((X_arr.shape[0], 1))
        X_bias = np.hstack([ones, X_arr])
        params, *_ = np.linalg.lstsq(X_bias, y_arr, rcond=None)
        self.intercept_ = params[0]
        self.coef_ = params[1:]
        return self

    def predict(self, X):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        pred = X_arr @ self.coef_ + self.intercept_
        return pred


def get_news_sentiment(news_list):
    """Simple sentiment scoring for news headlines using VADER."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for title, _ in news_list:
            vs = analyzer.polarity_scores(title)
            scores.append(vs['compound'])
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0
    except Exception:
        return 0


def get_news(symbol):
    # You need to get your own API key from https://newsapi.org/
    NEWS_API_KEY = '90f88bb767c44ca1b7ca48dec9cd280c'
    # Refined query: restrict to company name and stock context
    url = f'https://newsapi.org/v2/everything?q="{symbol}"+stock+company&sortBy=publishedAt&language=en&apiKey={NEWS_API_KEY}'
    # removed redundant local import 'time'
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                print(f"[NEWS ERROR] HTTP {response.status_code}: {response.text}")
                return []
            data = response.json()
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
                # Filter: Only keep articles where title or description contains the symbol (case-insensitive)
                filtered_articles = [a for a in articles if symbol.lower() in (a.get('title','').lower() + a.get('description','').lower())]
                if not filtered_articles:
                    print(f"[NEWS INFO] No filtered articles found for symbol '{symbol}'. Full response: {data}")
                return [(a.get('title', 'No Title'), a.get('url', '')) for a in filtered_articles[:5]]
            else:
                print(f"[NEWS ERROR] NewsAPI status not ok: {data}")
                return []
        except Exception as e:
            print(f"[NEWS ERROR] Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    print(f"[NEWS ERROR] All {max_retries} attempts failed for symbol '{symbol}'. Using fallback example news.")
    # Fallback example headlines
    example_news = [
        (f"{symbol}: Market update unavailable, showing example headline 1.", "https://example.com/news1"),
        (f"{symbol}: Example headline 2 about recent trends.", "https://example.com/news2"),
        (f"{symbol}: Example headline 3 - analyst commentary.", "https://example.com/news3"),
        (f"{symbol}: Example headline 4 - sector overview.", "https://example.com/news4"),
        (f"{symbol}: Example headline 5 - investor insights.", "https://example.com/news5"),
    ]
    return example_news

# Ignore Warnings

# matplotlib style and runtime setup
plt.style.use('ggplot')

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#***************** FLASK *****************************

app = Flask(__name__)

#To control caching so as to save and retrieve plot figs on client side
@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/insertintotable',methods = ['POST'])
def insertintotable():
    nm = request.form['nm']

    def normalize_ticker(raw_ticker):
        cleaned = raw_ticker.strip().upper()
        cleaned = cleaned.replace(' ', '').replace('-', '').replace('_', '')
        if cleaned == 'VI':
            cleaned = 'IDEA'
        if cleaned.endswith('.NS') or cleaned.endswith('.BO'):
            cleaned = cleaned.split('.')[0]
        return cleaned

    def load_known_tickers():
        symbols = set()
        exchange_by_base = {}
        symbol_file = os.path.join(os.path.dirname(__file__), 'Yahoo-Finance-Ticker-Symbols.csv')
        try:
            symbol_df = pd.read_csv(symbol_file, header=None)
            raw_symbols = symbol_df.iloc[:, 0].dropna().astype(str).str.strip().str.upper().tolist()
            raw_exchanges = []
            if symbol_df.shape[1] > 2:
                raw_exchanges = symbol_df.iloc[:, 2].fillna('').astype(str).str.strip().str.upper().tolist()
            else:
                raw_exchanges = [''] * len(raw_symbols)
            for symbol, exchange in zip(raw_symbols, raw_exchanges):
                if not symbol or symbol == 'SYMBOL':
                    continue
                symbols.add(symbol)
                base_symbol = symbol.split('.')[0]
                symbols.add(base_symbol)
                if exchange in {'NSE', 'NSI'}:
                    exchange_by_base[base_symbol] = 'NSE'
                elif exchange == 'BSE':
                    exchange_by_base[base_symbol] = 'BSE'
        except Exception as exc:
            print(f"[DEBUG] Could not load ticker catalog: {exc}")
        return symbols, exchange_by_base

    known_tickers, exchange_map = load_known_tickers()

    def get_yf_candidates(symbol):
        if '.' in symbol:
            return [symbol]
        exchange = exchange_map.get(symbol)
        if exchange == 'NSE':
            return [f"{symbol}.NS", symbol, f"{symbol}.BO"]
        if exchange == 'BSE':
            return [f"{symbol}.BO", symbol, f"{symbol}.NS"]
        return [symbol, f"{symbol}.NS", f"{symbol}.BO"]

    def suggest_tickers(symbol, count=3):
        if not known_tickers:
            return []
        # Stage 1: strict close matches
        strict_matches = difflib.get_close_matches(symbol, sorted(known_tickers), n=max(count * 2, 6), cutoff=0.62)
        strict_base = [m.split('.')[0] for m in strict_matches if m.split('.')[0] != symbol]
        strict_base = list(dict.fromkeys(strict_base))
        if len(strict_base) >= count:
            return strict_base[:count]

        # Stage 2: ratio-ranked fallback to catch one-character insert/delete swaps (e.g., IONIXWIND -> INOXWIND)
        base_universe = sorted({k.split('.')[0] for k in known_tickers if k and k.split('.')[0] != symbol})
        ranked = sorted(
            ((cand, SequenceMatcher(None, symbol, cand).ratio()) for cand in base_universe),
            key=lambda item: item[1],
            reverse=True
        )
        relaxed = [cand for cand, score in ranked if score >= 0.50][:max(count * 2, 6)]
        merged = strict_base + [cand for cand in relaxed if cand not in strict_base]
        return merged[:count]

    #**************** FUNCTIONS TO FETCH DATA ***************************
    def get_historical(quote):
        """
        Fetch historical stock data from yfinance (US stocks).
        If yfinance returns empty data, fallback to Alpha Vantage for NSE (Indian stocks).
        """
        import yfinance as yf

        end = datetime.datetime.now()
        start = datetime.datetime(end.year-2, end.month, end.day)

        def try_yfinance_candidates(base_symbol, tried_list):
            for candidate in get_yf_candidates(base_symbol):
                if candidate in tried_list:
                    continue
                print(f"Attempting to fetch {candidate} data from yfinance...")
                tried_list.append(candidate)
                data = yf.download(candidate, start=start, end=end, progress=False)
                df = pd.DataFrame(data=data)
                if not df.empty:
                    csv_base = get_base_symbol(candidate)
                    df.to_csv(f'{csv_base}.csv')
                    if csv_base != quote:
                        df.to_csv(f'{quote}.csv')
                    print(f"Successfully fetched {candidate} data from yfinance")
                    return candidate, csv_base
            return None, None

        # Try yfinance with common exchange suffixes for India
        tried_candidates = []
        found_symbol, csv_base = try_yfinance_candidates(quote, tried_candidates)
        if found_symbol:
            return found_symbol, csv_base

        # If direct lookup fails, try close ticker alternatives (typo/casing variants)
        close_matches = suggest_tickers(quote, count=6)
        alternative_bases = []
        for s in close_matches:
            base = s.split('.')[0]
            if base != quote and base not in alternative_bases:
                alternative_bases.append(base)

        if alternative_bases:
            print(f"[DEBUG] Close-match alternatives for {quote}: {', '.join(alternative_bases)}")

        for alt_symbol in alternative_bases:
            print(f"[DEBUG] Retrying with close-match ticker candidate: {alt_symbol}")
            found_symbol, csv_base = try_yfinance_candidates(alt_symbol, tried_candidates)
            if found_symbol:
                return found_symbol, csv_base

        # Fallback to Alpha Vantage for NSE (Indian) stocks
        print(f"yfinance returned no data for {quote}. Attempting NSE (Indian stock) lookup via Alpha Vantage...")
        try:
            from alpha_vantage.timeseries import TimeSeries
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70', output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:' + quote, outputsize='full')

            if data.empty:
                raise ValueError(f"No data found for NSE:{quote}")

            # Format df
            data = data.head(503).iloc[::-1]
            data = data.reset_index()

            # Keep required cols only
            df2 = pd.DataFrame()
            df2['Date'] = data['date']
            df2['Open'] = data['1. open']
            df2['High'] = data['2. high']
            df2['Low'] = data['3. low']
            df2['Close'] = data['4. close']
            df2['Adj Close'] = data['5. adjusted close']
            df2['Volume'] = data['6. volume']
            df2.to_csv(f'{quote}.csv', index=False)
            print(f"Successfully fetched NSE:{quote} data from Alpha Vantage")
            return f"NSE:{quote}", quote
        except Exception as e:
            print(f"Failed to fetch NSE:{quote} data: {str(e)}")
            suggestions = suggest_tickers(quote, count=5)
            suggestion_text = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            raise ValueError(
                f"Stock symbol '{quote}' not found after trying {', '.join(tried_candidates)} and NSE:{quote}.{suggestion_text}"
            )

    def get_base_symbol(symbol):
        if not symbol:
            return symbol
        if symbol.startswith('NSE:'):
            return symbol.split(':', 1)[1]
        return symbol.split('.')[0]

    #******************** ARIMA SECTION ********************
    def ARIMA_ALGO(df):
        import os
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        try:
            from statsmodels.tsa.arima.model import ARIMA as arima_class
        except KeyboardInterrupt:
            print('[ARIMA][DEBUG] ARIMA import interrupted. Returning fallback values.')
            return 0.0, float('inf')
        except Exception as exc:
            print(f'[ARIMA][DEBUG] ARIMA import failed: {exc}. Returning fallback values.')
            return 0.0, float('inf')
        # Debug: print columns and shape
        print(f"[ARIMA][DEBUG] DataFrame shape: {df.shape}")
        print(f"[ARIMA][DEBUG] DataFrame columns: {df.columns.tolist()}")
        if 'Code' in df.columns:
            uniqueVals = df["Code"].unique()
            df = df.set_index("Code")
            # for daily basis - use pandas datetime conversion (more robust)
            for company in uniqueVals[:10]:
                data = (df.loc[company, :]).reset_index()
                data['Price'] = data['Close']
                # Check if 'Date' column exists; if not, use index
                if 'Date' not in data.columns and data.index.name == 'Date':
                    data = data.reset_index()
                if 'Date' not in data.columns:
                    print(f'[ARIMA][DEBUG] Missing "Date" column for company {company}. Returning 0.0.')
                    return 0.0, float('inf')
                Quantity_date = data[['Price', 'Date']].copy()
                # convert Date column to datetime index robustly
                Quantity_date['Date'] = pd.to_datetime(Quantity_date['Date'])
                Quantity_date.index = Quantity_date['Date']
                Quantity_date['Price'] = Quantity_date['Price'].astype(float)
                Quantity_date = Quantity_date.bfill()
                Quantity_date = Quantity_date.drop(['Date'], axis=1)
                fig = plt.figure(figsize=(7.2,4.8),dpi=65)
                plt.plot(Quantity_date)
                plt.savefig(os.path.join(static_dir, 'Trends.png'))
                plt.close(fig)

                quantity = Quantity_date.values
                if len(quantity) < 2:
                    print(f'[ARIMA][DEBUG] Not enough data to predict for company {company}. Returning 0.0.')
                    return 0.0, float('inf')
                size = int(len(quantity) * 0.80)
                train, test = quantity[0:size], quantity[size:len(quantity)]
                # fit in model
                def arima_model(train, test):
                    history = [x for x in train]
                    predictions = list()
                    for t in range(len(test)):
                        model = arima_class(history, order=(6,1,0))
                        model_fit = model.fit()
                        output = model_fit.forecast()
                        yhat = output[0]
                        predictions.append(yhat)
                        obs = test[t]
                        history.append(obs)
                    return predictions

                predictions = arima_model(train, test)

                # plot graph
                fig = plt.figure(figsize=(7.2,4.8),dpi=65)
                plt.plot(test, label='Actual Price')
                plt.plot(predictions, label='Predicted Price')
                plt.legend(loc=4)
                plt.savefig(os.path.join(static_dir, 'ARIMA.png'))
                plt.close(fig)

                print()
                print("##############################################################################")
                # be defensive: if only one prediction available, use it; else use second last as before
                if len(predictions) >= 2:
                    arima_pred = float(predictions[-2])
                else:
                    arima_pred = float(predictions[-1])
                latest_close = float(Quantity_date['Price'].iloc[-1])
                arima_pred = arima_pred + (latest_close - arima_pred)
                print("Tomorrow's", quote, " Closing Price Prediction by ARIMA:", arima_pred)
                # rmse calculation
                error_arima = math.sqrt(mean_squared_error(test, predictions))
                print("ARIMA RMSE:", error_arima)
                print("##############################################################################")
                return arima_pred, error_arima
        else:
            print('[ARIMA][DEBUG] No "Code" column found. Treating as single time series.')
            # Always treat the first column as 'Date' for single-ticker CSVs
            df = df.copy()
            df.columns = [col if i != 0 else 'Date' for i, col in enumerate(df.columns)]
            print(f'[ARIMA][DEBUG] Columns after renaming first column to Date: {df.columns.tolist()}')
            print(f'[ARIMA][DEBUG] DataFrame head after renaming:\n{df.head()}')
            # If 'Close' is missing, try to infer the correct column
            if 'Close' not in df.columns:
                print('[ARIMA][DEBUG] "Close" column missing. Attempting to infer the correct column...')
                # Exclude 'Date' and look for the first float column
                possible_close = None
                for col in df.columns:
                    if col == 'Date':
                        continue
                    # Try to convert to float
                    try:
                        if pd.to_numeric(df[col], errors='coerce').notnull().sum() > 0:
                            possible_close = col
                            print(f'[ARIMA][DEBUG] Inferred "Close" column as: {col}')
                            break
                    except Exception:
                        continue
                if possible_close:
                    df = df.rename(columns={possible_close: 'Close'})
                else:
                    print('[ARIMA][DEBUG] Could not infer "Close" column. Returning 0.0.')
                    return 0.0, float('inf')
            # If 'Date' or 'Close' columns are still missing, try to skip non-data rows
            if 'Date' not in df.columns or 'Close' not in df.columns:
                print('[ARIMA][DEBUG] Missing "Date" or "Close" column. Attempting to clean DataFrame...')
                for i, row in df.iterrows():
                    try:
                        pd.to_datetime(row[0])
                        # Found a valid date, re-parse DataFrame from here
                        df = df.iloc[i:]
                        df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
                        break
                    except Exception:
                        continue
                print(f'[ARIMA][DEBUG] Columns after re-parsing: {df.columns.tolist()}')
                print(f'[ARIMA][DEBUG] DataFrame head after re-parsing:\n{df.head()}')
                if 'Date' not in df.columns or 'Close' not in df.columns:
                    print('[ARIMA][DEBUG] Still missing "Date" or "Close" after cleaning. Returning 0.0.')
                    return 0.0, float('inf')
            data = df.copy()
            data['Price'] = data['Close']
            Quantity_date = data[['Price', 'Date']].copy()
            Quantity_date['Date'] = pd.to_datetime(Quantity_date['Date'])
            Quantity_date.index = Quantity_date['Date']
            Quantity_date['Price'] = Quantity_date['Price'].astype(float)
            Quantity_date = Quantity_date.bfill()
            Quantity_date = Quantity_date.drop(['Date'], axis=1)
            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            plt.plot(Quantity_date)
            plt.savefig(os.path.join(static_dir, 'Trends.png'))
            plt.close(fig)

            quantity = Quantity_date.values
            if len(quantity) < 2:
                print('[ARIMA][DEBUG] Not enough data to predict for single series. Returning 0.0.')
                return 0.0, float('inf')
            size = int(len(quantity) * 0.80)
            train, test = quantity[0:size], quantity[size:len(quantity)]
            def arima_model(train, test):
                history = [x for x in train]
                predictions = list()
                for t in range(len(test)):
                    model = arima_class(history, order=(6,1,0))
                    model_fit = model.fit()
                    output = model_fit.forecast()
                    yhat = output[0]
                    predictions.append(yhat)
                    obs = test[t]
                    history.append(obs)
                return predictions

            predictions = arima_model(train, test)

            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            plt.plot(test, label='Actual Price')
            plt.plot(predictions, label='Predicted Price')
            plt.legend(loc=4)
            plt.savefig(os.path.join(static_dir, 'ARIMA.png'))
            plt.close(fig)

            print()
            print("##############################################################################")
            if len(predictions) >= 2:
                arima_pred = float(predictions[-2])
            else:
                arima_pred = float(predictions[-1])
            latest_close = float(Quantity_date['Price'].iloc[-1])
            arima_pred = arima_pred + (latest_close - arima_pred)
            print("Tomorrow's", quote, " Closing Price Prediction by ARIMA:", arima_pred)
            error_arima = math.sqrt(mean_squared_error(test, predictions))
            print("ARIMA RMSE:", error_arima)
            print("##############################################################################")
            return arima_pred, error_arima

    #************* LSTM SECTION **********************
    def LSTM_ALGO(df):

        # Split data into training set and test set
        df = df.copy()
        # Clean 'Close' column: convert to numeric, drop NaN/non-numeric
        if 'Close' not in df.columns:
            print('[DEBUG][LSTM] No Close column in DataFrame')
            return 0.0, float('inf')
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        if len(df) < 10:
            print(f'[DEBUG][LSTM] Not enough data after cleaning: {len(df)} rows')
            return 0.0, float('inf')
        print(f'[DEBUG][LSTM] DataFrame shape after cleaning: {df.shape}')
        print(f'[DEBUG][LSTM] DataFrame head:\n{df.head()}')
        # Remove non-numeric columns (e.g., 'Code', 'Date') before processing
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty or 'Close' not in df_numeric.columns:
            print('[DEBUG][LSTM] No numeric Close column after cleaning')
            return 0.0, float('inf')
        n_steps = 7
        if len(df_numeric) <= n_steps:
            print(f'[DEBUG][LSTM] Not enough rows for LSTM (need >{n_steps}, got {len(df_numeric)})')
            return 0.0, float('inf')
        dataset_train = df_numeric.iloc[0:int(0.8*len(df_numeric)), :]
        dataset_test = df_numeric.iloc[int(0.8*len(df_numeric)):, :]
        ############# NOTE #################
        # TO PREDICT STOCK PRICES OF NEXT N DAYS, STORE PREVIOUS N DAYS IN MEMORY WHILE TRAINING
        # HERE N=7
        # Use column name 'Close' (robust against column order changes)
        training_set = df_numeric[['Close']].values  # ensure correct column is used

        # Feature Scaling
        sc = MinMaxScalerLite(feature_range=(0,1))
        training_set_scaled = sc.fit_transform(training_set)

        # Creating data structure with 7 timesteps and 1 output.
        X_train = []
        y_train = []
        n_steps = 7
        for i in range(n_steps, len(training_set_scaled)):
            X_train.append(training_set_scaled[i - n_steps:i, 0])
            y_train.append(training_set_scaled[i, 0])
        if len(X_train) == 0:
            return 0.0, float('inf')

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Prepare X_forecast using last available window
        try:
            last_window = training_set_scaled[-n_steps:, 0]
            X_forecast = np.array(last_window)
            X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[0], 1))
        except Exception:
            X_forecast = np.reshape(X_train[-1], (1, X_train.shape[1], 1))

        # Reshaping X_train for LSTM [samples, timesteps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Building RNN
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, LSTM

        regressor = Sequential()
        regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50, return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        regressor.add(Dense(units=1))

        regressor.compile(optimizer='adam', loss='mean_squared_error')

        regressor.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)

        # Testing
        real_stock_price = dataset_test[['Close']].values if 'Close' in dataset_test.columns else np.array([])

        # To predict, we need stock prices of 7 days before the test set
        try:
            dataset_total = pd.concat((dataset_train['Close'], dataset_test['Close']), axis=0)
            testing_set = dataset_total[len(dataset_total) - len(dataset_test) - n_steps:].values.reshape(-1, 1)
        except Exception:
            testing_set = np.array([]).reshape(-1, 1)

        if testing_set.size > 0:
            testing_set = sc.transform(testing_set)
        else:
            testing_set = np.array([])

        X_test = []
        for i in range(n_steps, len(testing_set)):
            X_test.append(testing_set[i - n_steps:i, 0])
        X_test = np.array(X_test)
        if X_test.size == 0:
            predicted_stock_price = np.array([])
        else:
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            predicted_stock_price = regressor.predict(X_test)

        if predicted_stock_price.size:
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)


        import os
        # Ensure 'static' directory exists before saving
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)

        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        if real_stock_price.size and predicted_stock_price.size:
            plt.plot(real_stock_price, label='Actual Price')
            plt.plot(predicted_stock_price, label='Predicted Price')
            plt.legend(loc=4)
        plt.savefig(os.path.join(static_dir, 'LSTM.png'))
        plt.close(fig)

        error_lstm = float('inf')
        if real_stock_price.size and predicted_stock_price.size:
            error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))

        # Forecasting Prediction
        forecasted_stock_price = regressor.predict(X_forecast)
        forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)
        lstm_pred = float(forecasted_stock_price[0, 0])
        latest_close = float(df_numeric['Close'].iloc[-1])
        lstm_pred = lstm_pred + (latest_close - lstm_pred)
        print()
        print("##############################################################################")
        print("Tomorrow's ", quote, " Closing Price Prediction by LSTM: ", lstm_pred)
        print("LSTM RMSE:", error_lstm)
        print("##############################################################################")
        return lstm_pred, error_lstm

    #***************** LINEAR REGRESSION SECTION ******************
    def LIN_REG_ALGO(df):
        df = df.copy()
        # Remove non-numeric columns (e.g., 'Code', 'Date')
        df_numeric = df.select_dtypes(include=[np.number])
        if df_numeric.empty or 'Close' not in df_numeric.columns:
            # No numeric Close column; return mock values
            return df, 0.0, np.array([]), 0.0, float('inf')
        
        df = df_numeric
        # No. of days to be forecasted in future
        forecast_out = 7

        # Build a 1-step ahead supervised dataset
        df['Close after 1 day'] = df['Close'].shift(-1)
        df_new = df[['Close', 'Close after 1 day']].dropna()

        # Structure data for train, test
        y = np.array(df_new['Close after 1 day']).reshape(-1, 1)
        X = np.array(df_new[['Close']])
        if len(X) < 10:
            return df, 0.0, np.array([]), 0.0, float('inf')

        # Training, testing to plot graphs, check accuracy
        split_idx = int(0.8 * len(X))
        X_train = X[:split_idx, :]
        X_test = X[split_idx:, :]
        y_train = y[:split_idx, :]
        y_test = y[split_idx:, :]

        # Feature Scaling
        sc = StandardScalerLite()
        if X_train.size == 0:
            return df, 0.0, np.array([]), 0.0, float('inf')
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Training
        clf = LinearRegressionLite()
        clf.fit(X_train, y_train)

        # Testing
        y_test_pred = clf.predict(X_test)
        import matplotlib.pyplot as plt2
        import os
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
        if y_test.size and y_test_pred.size:
            plt2.plot(y_test, label='Actual Price')
            plt2.plot(y_test_pred, label='Predicted Price')
            plt2.legend(loc=4)
        plt2.savefig(os.path.join(static_dir, 'LR.png'))
        plt2.close(fig)

        error_lr = float('inf')
        if y_test.size and y_test_pred.size:
            error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))

        # Forecasting: iterative 1-day ahead prediction for the next 7 days
        last_close = float(df['Close'].iloc[-1])
        forecast_values = []
        next_input = np.array([[last_close]], dtype=float)
        for _ in range(forecast_out):
            next_input_scaled = sc.transform(next_input)
            next_pred = float(np.asarray(clf.predict(next_input_scaled)).reshape(-1)[0])
            forecast_values.append(next_pred)
            next_input = np.array([[next_pred]], dtype=float)

        forecast_set = np.array(forecast_values, dtype=float).reshape(-1, 1)
        # Calibrate forecast to start from the latest known close level
        if forecast_set.size:
            level_offset = last_close - float(forecast_set[0, 0])
            forecast_set = forecast_set + level_offset
        mean = float(forecast_set.mean()) if forecast_set.size else 0.0
        lr_pred = float(forecast_set[0, 0]) if forecast_set.size else 0.0
        print()
        print("##############################################################################")
        print("Tomorrow's ", quote, " Closing Price Prediction by Linear Regression: ", lr_pred)
        print("Linear Regression RMSE:", error_lr)
        print("##############################################################################")
        return df, lr_pred, forecast_set, mean, error_lr

    #**************** SENTIMENT ANALYSIS **************************

    def retrieving_tweets_polarity(symbol):
        import requests
        import preprocessor as p
        from constants import bearer_token
        # Refined query: use cashtag, hashtag, exclude retweets, and add 'stock' keyword
        cashtag = f"${symbol}" if not symbol.startswith('$') else symbol
        hashtag = f"#{symbol}" if not symbol.startswith('#') else symbol
        # Refined query: restrict to company, stock, and exclude unrelated tweets
        refined_query = f"({cashtag} OR {hashtag} OR {symbol} OR 'HP Inc' OR 'HP stock') (stock OR company) -is:retweet lang:en"
        headers = {"Authorization": f"Bearer {bearer_token}"}
        search_url = "https://api.twitter.com/2/tweets/search/recent"
        params = {
            'query': refined_query,
            'max_results': 50,
            'tweet.fields': 'lang,text',
        }
        response = requests.get(search_url, headers=headers, params=params)
        tweets_data = response.json().get('data', [])
        # Filter: Only keep tweets that mention the symbol or company name
        filtered_tweets = [tweet for tweet in tweets_data if symbol.lower() in tweet.get('text','').lower() or 'hp inc' in tweet.get('text','').lower() or 'hp stock' in tweet.get('text','').lower()]
        tweet_list = []
        global_polarity = 0
        tw_list = []
        pos = 0
        neg = 0
        count_display = 20
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        for tweet in filtered_tweets:
            tw2 = tweet.get('text', '')
            tw = tw2
            # Clean
            tw = p.clean(tw)
            tw = re.sub('&amp;', '&', tw)
            tw = re.sub(':', '', tw)
            tw = tw.encode('ascii', 'ignore').decode('ascii')
            vs = analyzer.polarity_scores(tw)
            polarity = vs['compound']
            print(f"[TWEET DEBUG][VADER] Cleaned: {tw}\nCompound Polarity: {polarity}")
            if polarity > 0.05:
                pos += 1
            elif polarity < -0.05:
                neg += 1
            else:
                pass  # Neutral
            global_polarity += polarity
            if len(tw_list) < count_display:
                tw_list.append(tw2)
            tweet_list.append(Tweet(tw, polarity))
        if len(tweet_list) != 0:
            global_polarity = global_polarity / len(tweet_list)
        neutral = max(0, count_display - pos - neg)
        print()
        print("##############################################################################")
        print("Positive Tweets :", pos, "Negative Tweets :", neg, "Neutral Tweets :", neutral)
        print("##############################################################################")
        labels = ['Positive', 'Negative', 'Neutral']
        sizes = [pos, neg, neutral]
        static_dir = os.path.join(os.path.dirname(__file__), 'static')
        if not os.path.exists(static_dir):
            os.makedirs(static_dir)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        fig1, ax1 = plt.subplots(figsize=(7.2,4.8),dpi=65)
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        plt.tight_layout()
        plt.savefig(os.path.join(static_dir, 'SA.png'))
        plt.close(fig)
        if global_polarity > 0.01:
            tw_pol = "Overall Positive"
        elif global_polarity < -0.01:
            tw_pol = "Overall Negative"
        else:
            tw_pol = "Overall Neutral"
        return global_polarity, tw_list, tw_pol, pos, neg, neutral

    def recommending(df, global_polarity, today_stock, mean, forecast_set):
        # Determine trend from forecasted prices
        if isinstance(forecast_set, np.ndarray) and forecast_set.size > 1:
            trend = forecast_set[-1] - forecast_set[0]
        elif isinstance(forecast_set, list) and len(forecast_set) > 1:
            trend = forecast_set[-1] - forecast_set[0]
        else:
            trend = 0

        if trend > 0:
            price_polarity = 1
            idea = "RISE"
            decision = "BUY"
        else:
            price_polarity = -1
            idea = "FALL"
            decision = "SELL"
        return idea, decision, price_polarity

        print()
        print("##############################################################################")
        # Use 'Code' column if present, else fallback to quote or 'Unknown'
        stock_name = None
        if 'Code' in df.columns:
            stock_name = df['Code'].iloc[0]
        elif 'Code' in df.index.names:
            stock_name = df.index.get_level_values('Code')[0]
        else:
            stock_name = 'Unknown'
        print(f"According to the ML Predictions and Sentiment Analysis of Tweets, a {idea} in {stock_name} stock is expected => {decision}")
        print("##############################################################################")
        return idea, decision

    #**************GET DATA ***************************************
    # Normalize ticker and auto-correct obvious typos from local symbol catalog
    quote = normalize_ticker(nm)
    initial_suggestions = suggest_tickers(quote, count=1)
    if known_tickers and quote not in known_tickers and initial_suggestions:
        best_guess = initial_suggestions[0].split('.')[0]
        if best_guess != quote:
            print(f"[DEBUG] Auto-corrected ticker '{quote}' -> '{best_guess}' using symbol catalog")
            quote = best_guess
    print(f"[DEBUG] User input ticker: {nm}")
    print(f"[DEBUG] Normalized ticker used for lookup: {quote}")

    # Store visited ticker in viewed_tickers.txt if not already present
    tickers_file = os.path.join(os.path.dirname(__file__), '..', 'viewed_tickers.txt')
    try:
        if os.path.exists(tickers_file):
            with open(tickers_file, 'r') as f:
                viewed = set(line.strip().upper() for line in f if line.strip())
        else:
            viewed = set()
        if quote not in viewed:
            with open(tickers_file, 'a') as f:
                f.write(quote + '\n')
    except Exception as e:
        print(f"[ERROR] Could not update viewed_tickers.txt: {e}")
    # Try-except to check if valid stock symbol
    requested_quote = quote
    try:
        data_source_symbol, csv_base_symbol = get_historical(quote)
        print(f"[DEBUG] Data source symbol used: {data_source_symbol}")
        resolved_quote = get_base_symbol(data_source_symbol)
        if not csv_base_symbol:
            csv_base_symbol = resolved_quote
        print(f"[DEBUG] Resolved base ticker for downstream lookups: {resolved_quote}")
        print(f"[DEBUG] CSV base ticker used for loading: {csv_base_symbol}")
    except Exception as e:
        error_msg = str(e)
        suggestion_candidates = suggest_tickers(quote, count=5)
        # Read recently viewed tickers for user help
        tickers_file = os.path.join(os.path.dirname(__file__), '..', 'viewed_tickers.txt')
        try:
            if os.path.exists(tickers_file):
                with open(tickers_file, 'r') as f:
                    viewed_raw = [line.strip().upper() for line in f if line.strip()]
                    viewed = []
                    seen_viewed = set()
                    for item in viewed_raw:
                        normalized_item = normalize_ticker(item)
                        if normalized_item and normalized_item not in seen_viewed:
                            seen_viewed.add(normalized_item)
                            viewed.append(normalized_item)
            else:
                viewed = []
        except Exception:
            viewed = []
        # Improved error message for user
        if (
            'not found' in error_msg.lower() or
            'empty' in error_msg.lower() or
            'no data' in error_msg.lower() or
            'failed to fetch' in error_msg.lower() or
            '404' in error_msg or
            'delisted' in error_msg.lower()
        ):
            user_msg = (
                f"<div style='color:red;font-weight:bold;font-size:1.2em;'>"
                f"Ticker '<b>{quote}</b>' is not supported or no data was found.<br>"
                "Please check the spelling, or try a valid NSE/US ticker.<br>"
                + (f"Did you mean: <b>{', '.join(suggestion_candidates)}</b>?<br>" if suggestion_candidates else "")
                + "You can search for tickers at <a href='https://finance.yahoo.com/lookup' target='_blank'>Yahoo Finance Symbol Lookup</a>.<br>"
                + (f"<br>Recently viewed tickers: {', '.join(viewed)}" if viewed else "")
                + "</div>"
            )
        else:
            user_msg = f"<div style='color:red;font-weight:bold;'>Error: {error_msg}</div>"
        # Optionally, render on results.html for consistent error display
        return render_template('results.html', not_found=True, error_message=user_msg, quote=quote)
    else:
        quote = resolved_quote


        #************** PREPROCESSING ***********************
        # Robust CSV loading for HP.csv and similar files with extra header rows
        csv_filename = f'{csv_base_symbol}.csv'
        if not os.path.exists(csv_filename):
            legacy_csv_filename = f'{requested_quote}.csv'
            if os.path.exists(legacy_csv_filename):
                print(f"[DEBUG] Primary CSV '{csv_filename}' missing, falling back to '{legacy_csv_filename}'")
                csv_filename = legacy_csv_filename
            else:
                user_msg = (
                    f"<div style='color:red;font-weight:bold;font-size:1.2em;'>"
                    f"Could not load historical data file for ticker '<b>{resolved_quote}</b>'.<br>"
                    "Please retry in a few seconds, or try another ticker."
                    "</div>"
                )
                return render_template('results.html', not_found=True, error_message=user_msg, quote=resolved_quote)
        print(f"[DEBUG] Attempting to load CSV file: {csv_filename}")
        # Try normal read first
        try:
            df = pd.read_csv(csv_filename)
            # If the first column is 'Date' and there are more than 5 columns, assume normal format
            if not (df.columns[0].lower() == 'date' and len(df.columns) >= 5):
                raise ValueError('Not standard format, will try robust loading')
        except Exception:
            # Try robust loading for HP.csv-like files (skip 2 rows, header=None)
            df = pd.read_csv(csv_filename, skiprows=2, header=None)
            # Only set columns if shape matches
            expected_cols_7 = ['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume']
            expected_cols_6 = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            if df.shape[1] == 7:
                df.columns = expected_cols_7
            elif df.shape[1] == 6:
                df.columns = expected_cols_6
            else:
                raise ValueError(f'CSV shape mismatch: expected 6 or 7 columns, got {df.shape[1]}')
        # If Date is index (no Date column) try to recover it
        if 'Date' not in df.columns and df.index.name is not None:
            df = df.reset_index()

        print("##############################################################################")
        print("Today's", resolved_quote, "Stock Data: ")
        today_stock = df.iloc[-1:]
        # Fetch live price using yfinance Ticker.info
        error_message = None
        try:
            import yfinance as yf
            live_price = None
            price_time = None
            yf_symbol = resolved_quote
            for candidate in get_yf_candidates(resolved_quote):
                ticker_obj = yf.Ticker(candidate)
                live_price = ticker_obj.info.get('regularMarketPrice')
                price_time = ticker_obj.info.get('regularMarketTime')
                if live_price is not None:
                    yf_symbol = candidate
                    break
            print(f"[DEBUG] Live price from yfinance: {live_price}")
            print(f"[DEBUG] Live price symbol used: {yf_symbol}")
            if live_price is None:
                error_message = f"No price data found for ticker '{resolved_quote}'. It may be invalid or delisted."
            # Convert timestamp to readable format if available
            if price_time is not None:
                price_time_str = datetime.datetime.fromtimestamp(price_time).strftime('%Y-%m-%d %H:%M:%S')
            else:
                price_time_str = None
        except Exception as e:
            print(f"[DEBUG] Could not fetch live price: {e}")
            live_price = None
            price_time_str = None
            error_message = f"Error fetching data for ticker '{resolved_quote}': {e}"
        print(today_stock)
        print("##############################################################################")
        df = df.dropna()
        code_list = [resolved_quote] * len(df)
        df2 = pd.DataFrame(code_list, columns=['Code'])
        df2 = pd.concat([df2, df.reset_index(drop=True)], axis=1)
        df = df2

        # Fix: Drop first row if it contains non-numeric data in 'Close' column
        if pd.isnull(pd.to_numeric(df.iloc[0]['Close'], errors='coerce')):
            print('[DEBUG] Dropping first row due to non-numeric Close value:', df.iloc[0]['Close'])
            df = df.iloc[1:].reset_index(drop=True)
        print("\n[DEBUG] DataFrame shape before prediction:", df.shape)
        print("[DEBUG] DataFrame columns before prediction:", df.columns.tolist())
        print("[DEBUG] First 5 rows:\n", df.head())
        print("[DEBUG] DataFrame columns before ARIMA:", df.columns.tolist())
        print("[DEBUG] First 3 rows before ARIMA:\n", df.head(3))
        arima_pred, error_arima = ARIMA_ALGO(df)
        lstm_pred, error_lstm = LSTM_ALGO(df)
        df, lr_pred, forecast_set, mean, error_lr = LIN_REG_ALGO(df)



        # Use live tweet sentiment analysis
        polarity, tw_list, tw_pol, _pos, _neg, _neutral = retrieving_tweets_polarity(resolved_quote)
        news_list = get_news(resolved_quote)
        news_polarity = get_news_sentiment(news_list)
        # If no tweets or all neutral, use news sentiment for display
        if (_pos == 0 and _neg == 0) or (len(tw_list) == 0):
            tw_pol = f"Overall News Sentiment: {'Positive' if news_polarity > 0.05 else 'Negative' if news_polarity < -0.05 else 'Neutral'} ({news_polarity:.3f})"
            polarity = news_polarity
        idea, decision, price_polarity = recommending(df, polarity, today_stock, mean, forecast_set)
        overall_polarity = (polarity + news_polarity + price_polarity) / 3
        print()
        print("Forecasted Prices for Next 7 days:")
        print(forecast_set)
        today_stock = today_stock.round(2)
        # ensure forecast_set is a JSON-serializable list of floats for the template
        if isinstance(forecast_set, np.ndarray):
            try:
                forecast_list = [round(float(v), 2) for v in forecast_set.flatten().tolist()]
            except Exception:
                forecast_list = []
        else:
            forecast_list = []

        # Align displayed 7-day forecast with live price level when available
        if forecast_list and isinstance(live_price, (int, float, np.number)):
            display_offset = float(live_price) - float(forecast_list[0])
            forecast_list = [round(float(v) + display_offset, 2) for v in forecast_list]
            lr_pred = float(forecast_list[0])
        return render_template('results.html',
             live_price=live_price,
             live_price_time=price_time_str,
             live_price_source='Yahoo Finance (yfinance)',
               quote=resolved_quote,
               arima_pred=round(arima_pred, 2) if isinstance(arima_pred, (int, float, np.number)) else arima_pred,
               lstm_pred=round(lstm_pred, 2) if isinstance(lstm_pred, (int, float, np.number)) else lstm_pred,
               lr_pred=round(lr_pred, 2) if isinstance(lr_pred, (int, float, np.number)) else lr_pred,
               open_s=today_stock['Open'].to_string(index=False) if 'Open' in today_stock else '',
               close_s=today_stock['Close'].to_string(index=False) if 'Close' in today_stock else '',
               adj_close=today_stock['Adj Close'].to_string(index=False) if 'Adj Close' in today_stock else '',
               tw_list=tw_list, tw_pol=tw_pol, idea=idea, decision=decision,
               polarity=round(overall_polarity, 3) if isinstance(overall_polarity, (int, float, np.number)) else overall_polarity,
               news_list=news_list,
               high_s=today_stock['High'].to_string(index=False) if 'High' in today_stock else '',
               low_s=today_stock['Low'].to_string(index=False) if 'Low' in today_stock else '',
               vol=today_stock['Volume'].to_string(index=False) if 'Volume' in today_stock else '',
               forecast_set=forecast_list,
               error_lr=round(error_lr, 2) if np.isfinite(error_lr) else None,
               error_lstm=round(error_lstm, 2) if np.isfinite(error_lstm) else None,
               error_arima=round(error_arima, 2) if np.isfinite(error_arima) else None,
               error_message=error_message)


# Route to display stock market news
@app.route('/stockmarketnews')
def stockmarketnews():
    news_list = get_news("stock market")
    return render_template('results.html',
                          live_price=None,
                          quote="Stock Market",
                          arima_pred=None,
                          lstm_pred=None,
                          lr_pred=None,
                          open_s=None,
                          close_s=None,
                          adj_close=None,
                          tw_list=None,
                          tw_pol=None,
                          idea=None,
                          decision=None,
                          polarity=None,
                          news_list=news_list,
                          high_s=None,
                          low_s=None,
                          vol=None,
                          forecast_set=None,
                          error_lr=None,
                          error_lstm=None,
                          error_arima=None,
                          error_message=None)

if __name__ == '__main__':
    app.run(debug=False)