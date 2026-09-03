import json
import time
import threading
import requests
import websocket
import pandas as pd
from typing import Callable, Optional, Dict, Any

from config import (
    BINANCE_REST_BASE,
    BINANCE_REST_FALLBACKS,
    BINANCE_WS_BASE,
    BINANCE_API_KEY,
    logger,
)

class BinanceClient:
    """
    Handles Binance REST API calls for historical data and real-time WebSocket streaming
    for persistent live candlestick tracking.
    """

    def __init__(self, symbol: str, interval: str):
        self.symbol = symbol.upper()
        self.interval = interval.lower()
        self.ws: Optional[websocket.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.is_running = False
        self.reconnect_delay = 2
        self.max_reconnect_delay = 30
        self.on_update_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_error_callback: Optional[Callable[[str], None]] = None

    def fetch_historical_klines(self, limit: int = 150) -> pd.DataFrame:
        """
        Fetch historical Kline/Candlestick data via REST API.
        Attempts primary endpoint and fallback endpoints.
        """
        endpoints = [BINANCE_REST_BASE] + BINANCE_REST_FALLBACKS
        params = {
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": limit
        }
        headers = {}
        if BINANCE_API_KEY:
            headers["X-MBX-APIKEY"] = BINANCE_API_KEY

        last_error = None
        for base_url in endpoints:
            url = f"{base_url}/api/v3/klines"
            try:
                response = requests.get(url, params=params, headers=headers, timeout=10)
                if response.status_code == 200:
                    raw_data = response.json()
                    df = self._format_kline_df(raw_data)
                    logger.info(f"Loaded {len(df)} historical candles for {self.symbol} ({self.interval}) from {base_url}")
                    return df
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    logger.warning(f"Failed to fetch from {url}: {error_msg}")
                    last_error = Exception(error_msg)
            except Exception as e:
                logger.warning(f"Error connecting to {url}: {e}")
                last_error = e

        raise ConnectionError(f"Failed to fetch historical klines for {self.symbol} after trying all endpoints. Last error: {last_error}")

    def _format_kline_df(self, raw_data: list) -> pd.DataFrame:
        """
        Converts raw Binance kline array into a typed pandas DataFrame.
        """
        columns = [
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(raw_data, columns=columns)
        
        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["is_closed"] = True
        return df[["timestamp", "open", "high", "low", "close", "volume", "is_closed"]]

    def start_stream(
        self,
        on_update: Callable[[Dict[str, Any]], None],
        on_error: Optional[Callable[[str], None]] = None
    ):
        """
        Starts WebSocket streaming in a dedicated daemon thread.
        """
        self.on_update_callback = on_update
        self.on_error_callback = on_error
        self.is_running = True
        self.ws_thread = threading.Thread(target=self._run_ws_loop, daemon=True)
        self.ws_thread.start()

    def _run_ws_loop(self):
        stream_name = f"{self.symbol.lower()}@kline_{self.interval}"
        ws_url = f"{BINANCE_WS_BASE}/{stream_name}"

        while self.is_running:
            try:
                logger.info(f"Connecting WebSocket to {ws_url}...")
                self.ws = websocket.WebSocketApp(
                    ws_url,
                    on_open=self._on_ws_open,
                    on_message=self._on_ws_message,
                    on_error=self._on_ws_error,
                    on_close=self._on_ws_close
                )
                self.ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                logger.error(f"WebSocket execution error: {e}")
                if self.on_error_callback:
                    self.on_error_callback(str(e))

            if self.is_running:
                logger.info(f"Reconnecting WebSocket in {self.reconnect_delay} seconds...")
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    def _on_ws_open(self, ws):
        logger.info("Binance WebSocket stream connection established.")
        self.reconnect_delay = 2

    def _on_ws_message(self, ws, message):
        try:
            data = json.loads(message)
            if "k" in data:
                k = data["k"]
                candle_dict = {
                    "timestamp": pd.to_datetime(k["t"], unit="ms"),
                    "open": float(k["o"]),
                    "high": float(k["h"]),
                    "low": float(k["l"]),
                    "close": float(k["c"]),
                    "volume": float(k["v"]),
                    "is_closed": bool(k["x"]),
                    "symbol": k["s"],
                    "interval": k["i"]
                }
                if self.on_update_callback:
                    self.on_update_callback(candle_dict)
        except Exception as e:
            logger.error(f"Error processing WS message: {e}")

    def _on_ws_error(self, ws, error):
        logger.error(f"WebSocket Error: {error}")
        if self.on_error_callback:
            self.on_error_callback(str(error))

    def _on_ws_close(self, ws, close_status_code, close_msg):
        logger.info(f"WebSocket stream closed (code: {close_status_code}, msg: {close_msg})")

    def stop(self):
        """
        Gracefully terminates the WebSocket stream and background worker.
        """
        self.is_running = False
        if self.ws:
            self.ws.close()
        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(timeout=2)
        logger.info("BinanceClient stopped.")
