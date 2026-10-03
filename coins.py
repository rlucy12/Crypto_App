"""Turn what the user types ("bitcoin", "eth", "Solana") into a Yahoo Finance ticker."""

COINS = {
    "BTC-USD": ("Bitcoin", ["bitcoin", "btc"]),
    "ETH-USD": ("Ethereum", ["ethereum", "eth", "ether"]),
    "SOL-USD": ("Solana", ["solana", "sol"]),
    "BNB-USD": ("BNB", ["bnb", "binance", "binance coin"]),
    "XRP-USD": ("XRP", ["xrp", "ripple"]),
    "ADA-USD": ("Cardano", ["cardano", "ada"]),
    "DOGE-USD": ("Dogecoin", ["dogecoin", "doge"]),
    "TRX-USD": ("TRON", ["tron", "trx"]),
    "LTC-USD": ("Litecoin", ["litecoin", "ltc"]),
    "DOT-USD": ("Polkadot", ["polkadot", "dot"]),
    "LINK-USD": ("Chainlink", ["chainlink", "link"]),
    "AVAX-USD": ("Avalanche", ["avalanche", "avax"]),
    "SHIB-USD": ("Shiba Inu", ["shiba inu", "shiba", "shib"]),
    "BCH-USD": ("Bitcoin Cash", ["bitcoin cash", "bch"]),
    "XLM-USD": ("Stellar", ["stellar", "xlm"]),
}

# Coins that train.py prepares by default
DEFAULT_TRAIN = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD"]


def resolve(text: str):
    """Return a ticker like 'BTC-USD', or None for empty input."""
    t = (text or "").strip().lower()
    if not t:
        return None
    for ticker, (_, names) in COINS.items():
        if t in names or t == ticker.lower():
            return ticker
    sym = t.upper().replace(" ", "")
    return sym if sym.endswith("-USD") else f"{sym}-USD"   # unknown coin: try its symbol


def display_name(ticker: str) -> str:
    return COINS.get(ticker, (ticker.replace("-USD", ""), []))[0]
