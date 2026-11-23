from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/insertintotable', methods=['POST'])
def insertintotable():
    nm = request.form.get('nm', 'MOCK')
    # Quick mocked predictions and values
    quote = nm.upper()
    arima_pred = 123.45
    lstm_pred = 124.56
    lr_pred = 122.33
    open_s = '120.00'
    close_s = '121.50'
    adj_close = '121.50'
    high_s = '125.00'
    low_s = '119.50'
    vol = '1,234,567'
    tw_list = ['Tweet 1 for ' + quote, 'Tweet 2 for ' + quote]
    tw_pol = 'Overall Positive'
    idea = 'RISE'
    decision = 'BUY'
    forecast_set = [125.0, 126.5, 127.0, 128.0, 129.5, 130.0, 131.5]
    error_lr = 1.23
    error_lstm = 2.34
    error_arima = 1.11
    return render_template('results.html',
                           quote=quote,
                           arima_pred=arima_pred,
                           lstm_pred=lstm_pred,
                           lr_pred=lr_pred,
                           open_s=open_s,
                           close_s=close_s,
                           adj_close=adj_close,
                           tw_list=tw_list,
                           tw_pol=tw_pol,
                           idea=idea,
                           decision=decision,
                           high_s=high_s,
                           low_s=low_s,
                           vol=vol,
                           forecast_set=forecast_set,
                           error_lr=error_lr,
                           error_lstm=error_lstm,
                           error_arima=error_arima)

if __name__ == '__main__':
    app.run(port=5001)
