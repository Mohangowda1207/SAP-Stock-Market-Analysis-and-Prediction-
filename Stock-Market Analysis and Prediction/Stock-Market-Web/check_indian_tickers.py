import yfinance as yf
from alpha_vantage.timeseries import TimeSeries

# Your Alpha Vantage API key
API_KEY = 'N6A6QT6IBFJOPJ70'  # Replace with your own if needed

tickers = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'HINDUNILVR', 'KOTAKBANK', 'SBIN', 'LT', 'ITC',
    'BHARTIARTL', 'MARUTI', 'SUNPHARMA', 'HDFC', 'BAJFINANCE', 'AXISBANK', 'TATAMOTORS', 'EICHERMOT', 'MRF', 'ONGC',
    'TITAN', 'POWERGRID', 'NTPC', 'WIPRO', 'BHARATFORG', 'GRASIM', 'ULTRACEMCO', 'DIVISLAB', 'JSWSTEEL', 'HCLTECH'
]

ts = TimeSeries(key=API_KEY, output_format='pandas')

for ticker in tickers:
    # Try yfinance with .NS first
    try:
        data = yf.download(f"{ticker}.NS", period="1mo", progress=False)
        if not data.empty:
            print(f'SUCCESS: {ticker} found on yfinance (.NS).')
            continue
        else:
            print(f'FAIL: {ticker} not found on yfinance (.NS), trying Alpha Vantage...')
    except Exception as e:
        print(f'ERROR: {ticker} yfinance - {str(e)}, trying Alpha Vantage...')
    # Fallback to Alpha Vantage
    try:
        data, meta_data = ts.get_daily_adjusted(symbol=f'NSE:{ticker}', outputsize='compact')
        if not data.empty:
            print(f'SUCCESS: {ticker} found on Alpha Vantage.')
        else:
            print(f'FAIL: {ticker} not found (empty data) on Alpha Vantage.')
    except Exception as e:
        print(f'ERROR: {ticker} Alpha Vantage - {str(e)}')
