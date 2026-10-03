# Crypto price forecast (ARIMA + TCN), instant version

The website never trains. You train once with `train.py`; the app only loads the
saved models and predicts, so results show in a few seconds.

## Setup (once)
```
python -m venv venv
venv\Scripts\activate            (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

## 1. Train the models (once, ~2-3 min per coin)
```
python train.py
```
Trains Bitcoin, Ethereum, Solana, BNB, XRP, Cardano and Dogecoin and saves them
in the `models/` folder. Add more coins any time:
```
python train.py --coins litecoin chainlink
```
Re-run every week or two so the models learn recent prices.

## 2. Start the website
```
streamlit run app.py
```
Type a coin name (bitcoin, ETH, solana…) and press "Get forecast".
Coins without a saved model still work, using a quick ARIMA-only forecast.

## Deploy online (Streamlit Community Cloud)
Push the whole folder **including `models/`** to GitHub, then create the app at
https://share.streamlit.io with `app.py` as the main file (Python 3.11 or 3.12).

## Website Link 
https://cryptoapp-lrfapa8dv3n34pbs8x7t8w.streamlit.app/
