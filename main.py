import sys
import time
import signal
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from config import (
    DEFAULT_TIMEFRAME,
    DEFAULT_CANDLE_LIMIT,
    SUPPORTED_TIMEFRAMES,
    logger
)
from symbol_mapper import resolve_symbol, suggest_symbols
from binance_client import BinanceClient
from price_action_engine import PriceActionEngine, AnalysisResult

console = Console() if HAS_RICH else None

def print_banner():
    banner = """
======================================================
           LIVE PRICE ACTION SCANNER
  Persistent Real-Time Candlestick & Pattern Analysis
======================================================
"""
    if HAS_RICH:
        console.print(Panel(
            "[bold cyan]LIVE PRICE ACTION SCANNER[/bold cyan]\n"
            "[italic white]Persistent Real-Time Candlestick & Pattern Analysis Engine[/italic white]\n"
            "[dim]Press Ctrl+C at any time to exit or switch pairs[/dim]",
            border_style="cyan"
        ))
    else:
        print(banner)

def get_user_inputs():
    print_banner()

    # 1. Market Pair Input
    while True:
        try:
            if HAS_RICH:
                market_input = console.input("[bold yellow]Enter market (e.g. bitcoin, eth, sol, BTCUSDT): [/bold yellow]").strip()
            else:
                market_input = input("Enter market (e.g. bitcoin, eth, sol, BTCUSDT): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)

        if not market_input:
            market_input = "bitcoin"

        resolved = resolve_symbol(market_input)
        suggestions = suggest_symbols(market_input)

        if HAS_RICH:
            console.print(f" Resolved symbol: [bold green]{resolved}[/bold green]")
        else:
            print(f" Resolved symbol: {resolved}")
        break

    # 2. Timeframe Input
    timeframe_prompt = f"Enter timeframe [{' / '.join(SUPPORTED_TIMEFRAMES[:6])}] (default: {DEFAULT_TIMEFRAME}): "
    try:
        if HAS_RICH:
            tf_input = console.input(f"[bold yellow]{timeframe_prompt}[/bold yellow]").strip().lower()
        else:
            tf_input = input(timeframe_prompt).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        sys.exit(0)

    timeframe = tf_input if tf_input in SUPPORTED_TIMEFRAMES else DEFAULT_TIMEFRAME
    if HAS_RICH:
        console.print(f" Active timeframe: [bold green]{timeframe}[/bold green]\n")
    else:
        print(f" Active timeframe: {timeframe}\n")

    return resolved, timeframe

def render_analysis_update(result: AnalysisResult):
    d = result.to_cli_display()

    # Color formatting
    bias_color = "green" if result.bias == "BULLISH" else ("red" if result.bias == "BEARISH" else "yellow")
    status_color = "cyan" if result.is_candle_closed else "magenta"

    if HAS_RICH:
        # Construct header line
        header = Text()
        header.append(f"[{d['timestamp']}] ", style="dim")
        header.append(f"{d['status']} ", style=status_color)
        header.append("| Price: ")
        header.append(f"${d['price']} ", style="bold white")
        header.append("| Bias: ")
        header.append(f"{d['bias']} ({d['confidence']}) ", style=f"bold {bias_color}")
        if result.patterns:
            header.append(f"| Pattern: {d['patterns']}", style="bold")

        console.print(header)

        # Context lines
        line2 = f"           Trend: {d['trend']} | {d['volume']} | Momentum: {d['momentum']}"
        line3 = f"           Support: {d['support']} | Resistance: {d['resistance']} | Rough Target: {d['target']}"
        console.print(f"[dim]{line2}[/dim]")
        console.print(f"[bold dim]{line3}[/bold dim]\n")
    else:
        print(f"[{d['timestamp']}] {d['status']} | Price: ${d['price']} | Bias: {d['bias']} ({d['confidence']}) | Pattern: {d['patterns']}")
        print(f"           Trend: {d['trend']} | {d['volume']} | Momentum: {d['momentum']}")
        print(f"           Support: {d['support']} | Resistance: {d['resistance']} | Rough Target: {d['target']}\n")

def main():
    symbol, timeframe = get_user_inputs()

    client = BinanceClient(symbol=symbol, interval=timeframe)
    engine = PriceActionEngine(max_candles=DEFAULT_CANDLE_LIMIT)

    # State tracking for display throttling
    last_print_time = 0.0
    last_price = 0.0
    last_patterns = []
    last_is_closed = False

    # Fetch initial historical candles
    msg_connect = f"Connecting to Binance REST API for historical {symbol} ({timeframe}) candles..."
    if HAS_RICH:
        console.print(f"[bold yellow]{msg_connect}[/bold yellow]")
    else:
        print(msg_connect)

    try:
        hist_df = client.fetch_historical_klines(limit=DEFAULT_CANDLE_LIMIT)
        engine.set_history(hist_df)
        success_msg = f"Connected! Seeded with {len(hist_df)} historical candles."
        if HAS_RICH:
            console.print(f"[bold green]{success_msg}[/bold green]")
            console.print(f"[bold cyan]Starting real-time WebSocket stream for {symbol} ({timeframe})...[/bold cyan]\n")
        else:
            print(success_msg)
            print(f"Starting real-time WebSocket stream for {symbol} ({timeframe})...\n")
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        if HAS_RICH:
            console.print(f"[bold red]Error loading market data: {e}[/bold red]")
        else:
            print(f"Error loading market data: {e}")
        return

    # Run initial analysis on seeded data
    initial_analysis = engine.analyze(symbol, timeframe)
    if initial_analysis:
        render_analysis_update(initial_analysis)
        last_price = initial_analysis.current_price
        last_patterns = initial_analysis.patterns
        last_is_closed = initial_analysis.is_candle_closed
        last_print_time = time.time()

    def on_candle_update(candle_dict):
        nonlocal last_print_time, last_price, last_patterns, last_is_closed
        try:
            engine.update_candle(candle_dict)
            now = time.time()
            current_price = candle_dict["close"]
            is_closed = candle_dict.get("is_closed", False)

            # Check if significant update occurred
            price_moved_pct = abs(current_price - last_price) / (last_price + 1e-9) * 100.0 if last_price > 0 else 0
            time_elapsed = now - last_print_time

            # Print criteria:
            # 1. Candle closed (high priority!)
            # 2. Or price moved >= 0.05%
            # 3. Or at least 3 seconds elapsed since last display
            should_print = (
                is_closed
                or (is_closed != last_is_closed)
                or (price_moved_pct >= 0.05 and time_elapsed >= 1.5)
                or (time_elapsed >= 3.0)
            )

            if should_print:
                analysis = engine.analyze(symbol, timeframe)
                if analysis:
                    render_analysis_update(analysis)
                    last_price = current_price
                    last_patterns = analysis.patterns
                    last_is_closed = is_closed
                    last_print_time = now

        except Exception as err:
            logger.error(f"Error during analysis loop: {err}")

    def on_stream_error(error_msg):
        logger.warning(f"Stream notification: {error_msg}")

    # Start live streaming
    client.start_stream(on_update=on_candle_update, on_error=on_stream_error)

    # Keep main thread alive waiting for user interrupt
    def shutdown_handler(sig, frame):
        print("\n\nShutting down stream gracefully...")
        client.stop()
        print("Live Price Action Scanner stopped. Goodbye!\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        while client.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_handler(None, None)

if __name__ == "__main__":
    main()
