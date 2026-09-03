import re
import difflib
from typing import Optional, List, Tuple

# Pre-defined mapping for popular cryptocurrencies and friendly aliases
COMMON_NAME_MAP = {
    # Bitcoin
    "bitcoin": "BTCUSDT",
    "btc": "BTCUSDT",
    "xbt": "BTCUSDT",
    # Ethereum
    "ethereum": "ETHUSDT",
    "ether": "ETHUSDT",
    "eth": "ETHUSDT",
    # Binance Coin
    "binance": "BNBUSDT",
    "bnb": "BNBUSDT",
    # Solana
    "solana": "SOLUSDT",
    "sol": "SOLUSDT",
    # Ripple
    "ripple": "XRPUSDT",
    "xrp": "XRPUSDT",
    # Cardano
    "cardano": "ADAUSDT",
    "ada": "ADAUSDT",
    # Dogecoin
    "dogecoin": "DOGEUSDT",
    "doge": "DOGEUSDT",
    # Avalanche
    "avalanche": "AVAXUSDT",
    "avax": "AVAXUSDT",
    # Polkadot
    "polkadot": "DOTUSDT",
    "dot": "DOTUSDT",
    # Polygon / MATIC
    "polygon": "MATICUSDT",
    "matic": "MATICUSDT",
    "pol": "POLUSDT",
    # Chainlink
    "chainlink": "LINKUSDT",
    "link": "LINKUSDT",
    # Near Protocol
    "near": "NEARUSDT",
    # Sui
    "sui": "SUIUSDT",
    # Aptos
    "aptos": "APTUSDT",
    "apt": "APTUSDT",
    # Toncoin
    "ton": "TONUSDT",
    "toncoin": "TONUSDT",
    # Shiba Inu
    "shiba": "SHIBUSDT",
    "shib": "SHIBUSDT",
    # Pepe
    "pepe": "PEPEUSDT",
    # Litecoin
    "litecoin": "LTCUSDT",
    "ltc": "LTCUSDT",
    # Uniswap
    "uniswap": "UNIUSDT",
    "uni": "UNIUSDT",
    # Cosmos
    "cosmos": "ATOMUSDT",
    "atom": "ATOMUSDT",
    # Stellar
    "stellar": "XLMUSDT",
    "xlm": "XLMUSDT",
    # Fetch.ai / ASI
    "fet": "FETUSDT",
    "render": "RENDERUSDT",
    "rndr": "RENDERUSDT",
}

def clean_input(raw_input: str) -> str:
    """Strip whitespace, slash, dash, and lower/uppercase normalization."""
    if not raw_input:
        return ""
    cleaned = raw_input.strip()
    cleaned = re.sub(r"[\s\-_/]+", "", cleaned)
    return cleaned

def resolve_symbol(user_input: str) -> str:
    """
    Resolves user input into a Binance valid symbol pair (e.g. 'bitcoin' -> 'BTCUSDT', 'eth' -> 'ETHUSDT').
    If only a token symbol is provided (e.g. 'ADA'), it defaults to '<TOKEN>USDT'.
    """
    raw = clean_input(user_input)
    if not raw:
        return "BTCUSDT"

    lower_raw = raw.lower()

    # 1. Check friendly common name dictionary
    if lower_raw in COMMON_NAME_MAP:
        return COMMON_NAME_MAP[lower_raw]

    upper_raw = raw.upper()

    # 2. Check if already ends with a standard quote currency
    quotes = ["USDT", "FDUSD", "USDC", "BTC", "ETH", "BUSD", "TUSD"]
    for q in quotes:
        if upper_raw.endswith(q) and len(upper_raw) > len(q):
            return upper_raw

    # 3. Default to USDT pair
    return f"{upper_raw}USDT"

def suggest_symbols(user_input: str, limit: int = 3) -> List[Tuple[str, str]]:
    """
    Returns suggestions as a list of (display_name, symbol) for typo correction.
    """
    query = user_input.strip().lower()
    if not query:
        return []

    names = list(COMMON_NAME_MAP.keys())
    matches = difflib.get_close_matches(query, names, n=limit, cutoff=0.5)
    return [(name, COMMON_NAME_MAP[name]) for name in matches]
