"""
main.py - Interactive AI Price Action Assistant & Institutional Cockpit v2.0
Enables:
  - Interactive AI Chat & Command Interface (send messages, query any coin, change timeframes)
  - Real-Time Live Streaming Cockpit with glitch-free Windows console rendering
  - Smart Money Concepts (FVG, Order Blocks, Liquidity Sweeps, BOS/CHoCH)
  - Volume Profile (POC/VAH/VAL), VWAP Variance Bands, Wyckoff VSA, Scipy KDE S/R
  - Machine Learning Calibrated Probabilities & Directional Confidence
  - Institutional Trade Setup Generator (Strict Effective R:R >= 1.80)
"""

from __future__ import annotations

import os
import sys
import time
import signal
import argparse
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

# Windows Console & VT100 / ANSI Support Initialization
if os.name == "nt":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        h_stdout = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        if kernel32.GetConsoleMode(h_stdout, ctypes.byref(mode)):
            kernel32.SetConsoleMode(h_stdout, mode.value | 0x0004)
    except Exception:
        pass

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import colorama
    colorama.init(autoreset=True)
except ImportError:
    pass

try:
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

import pandas as pd
from config import (
    DEFAULT_TIMEFRAME,
    DEFAULT_CANDLE_LIMIT,
    SUPPORTED_TIMEFRAMES,
    logger,
)
from symbol_mapper import resolve_symbol, suggest_symbols
from binance_client import BinanceClient
from price_action_engine import PriceActionEngine, AnalysisResult
from core.models import BiasType, SetupType, TradeSetup


def clear_screen():
    """Cross-platform clean screen clear without raw ANSI leakage."""
    os.system("cls" if os.name == "nt" else "clear")


def get_bias_color(bias: BiasType | str) -> str:
    val = bias.value if isinstance(bias, BiasType) else str(bias).upper()
    if "STRONG_BULLISH" in val or val == "BULLISH":
        return "bold green"
    elif "STRONG_BEARISH" in val or val == "BEARISH":
        return "bold red"
    return "bold yellow"


def make_progress_bar(percentage: int, width: int = 12) -> str:
    pct = max(0, min(100, int(percentage)))
    filled = int(round(width * (pct / 100.0)))
    empty = width - filled
    return f"{'#' * filled}{'-' * empty}"


# ============================================================================
# Rich Dashboard Component Builders
# ============================================================================

def create_header_panel(res: AnalysisResult) -> Panel:
    color = get_bias_color(res.bias)
    status_tag = "[green]CLOSED CANDLE[/green]" if res.is_candle_closed else "[magenta]LIVE FORMING[/magenta]"

    grid = Table.grid(expand=True)
    grid.add_column(ratio=2)
    grid.add_column(ratio=1, justify="right")

    left = f"[bold cyan]{res.symbol}[/bold cyan] ({res.timeframe.upper()}) | Price: [bold white]${res.current_price:,.2f}[/bold white] | Status: {status_tag}"
    right = f"Bias: [{color}]{res.bias}[/{color}] ([bold]{res.confidence}%[/bold]) | UTC: [dim]{datetime.now(timezone.utc).strftime('%H:%M:%S')}[/dim]"
    grid.add_row(left, right)

    return Panel(grid, box=box.ROUNDED, style="cyan")


def create_mtf_panel(res: AnalysisResult) -> Panel:
    table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE, expand=True)
    table.add_column("Timeframe", width=10)
    table.add_column("Bias / State", justify="center")

    rep = res.confluence_report
    if rep and rep.timeframe_breakdown:
        for tf in ["1d", "4h", "1h", "15m"]:
            if tf in rep.timeframe_breakdown:
                tf_conf = rep.timeframe_breakdown[tf]
                c = get_bias_color(tf_conf.bias)
                table.add_row(f"[bold]{tf.upper()}[/bold]", f"[{c}]{tf_conf.bias.value}[/{c}]")
            else:
                table.add_row(f"[bold]{tf.upper()}[/bold]", "[dim]N/A[/dim]")
    else:
        table.add_row(f"[bold]{res.timeframe.upper()}[/bold]", f"[{get_bias_color(res.bias)}]{res.bias}[/{get_bias_color(res.bias)}]")
        table.add_row("[bold]Trend Structure[/bold]", f"{res.trend_detail}")

    gauge = make_progress_bar(res.confidence, width=12)
    footer = f"\n[bold]Confluence Score:[/bold] [{get_bias_color(res.bias)}]{res.confidence}%[/] [dim][{gauge}][/dim]"

    return Panel(Group(table, Text.from_markup(footer)), title="[bold]1. MULTI-TIMEFRAME CONFLUENCE[/bold]", border_style="cyan", box=box.ROUNDED)


def create_smc_panel(res: AnalysisResult) -> Panel:
    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Label", style="bold white", width=13)
    table.add_column("Value")

    smc = res.smc_report
    if smc:
        active_fvgs = getattr(smc, "active_bullish_fvgs", []) + getattr(smc, "active_bearish_fvgs", [])
        if active_fvgs:
            top_fvg = active_fvgs[0]
            fvg_c = "green" if getattr(top_fvg, "bias", BiasType.BULLISH) == BiasType.BULLISH else "red"
            table.add_row("Active FVG", f"[{fvg_c}]${top_fvg.bottom:,.0f} - ${top_fvg.top:,.0f} (CE: ${top_fvg.midpoint:,.0f})[/{fvg_c}]")
        else:
            table.add_row("Active FVG", "[dim]None active[/dim]")

        active_obs = getattr(smc, "active_bullish_obs", []) + getattr(smc, "active_bearish_obs", [])
        if active_obs:
            top_ob = active_obs[0]
            ob_c = "green" if getattr(top_ob, "bias", BiasType.BULLISH) == BiasType.BULLISH else "red"
            state_tag = "Breaker" if getattr(top_ob, "is_breaker", False) else "Fresh"
            table.add_row("Order Block", f"[{ob_c}]${top_ob.bottom:,.0f} - ${top_ob.top:,.0f} ({state_tag})[/{ob_c}]")
        else:
            table.add_row("Order Block", "[dim]None active[/dim]")

        sweeps = getattr(smc, "recent_sweeps", [])
        if sweeps:
            sw = sweeps[-1]
            table.add_row("Liquidity", f"[yellow]{sw.sweep_type.value} (${sw.sweep_level:,.0f})[/yellow]")
        else:
            table.add_row("Liquidity", "[dim]No sweep[/dim]")

        if smc.recent_structure_events:
            ev = smc.recent_structure_events[-1]
            c = "green" if "BULLISH" in ev.event_type.value else "red"
            table.add_row("Structure", f"[{c}]{ev.event_type.value} (${ev.broken_level:,.0f})[/{c}]")
        else:
            table.add_row("Structure", f"[dim]{res.trend}[/dim]")
    else:
        table.add_row("Patterns", ", ".join(res.patterns) if res.patterns else "[dim]None[/dim]")
        table.add_row("Momentum", f"{res.momentum_state}")
        table.add_row("Volume", f"{res.volume_state}")

    return Panel(table, title="[bold]2. SMART MONEY FOOTPRINT[/bold]", border_style="cyan", box=box.ROUNDED)


def create_srp_panel(res: AnalysisResult) -> Panel:
    table = Table(box=box.SIMPLE, show_header=False, expand=True)
    table.add_column("Indicator", style="bold white", width=12)
    table.add_column("Level / Metric")

    res_str = f"[red]${res.nearest_resistance:,.2f}[/red]" if res.nearest_resistance else "[dim]N/A[/dim]"
    supp_str = f"[green]${res.nearest_support:,.2f}[/green]" if res.nearest_support else "[dim]N/A[/dim]"
    table.add_row("Resistance", res_str)
    table.add_row("Support", supp_str)
    table.add_row("Volume State", f"{res.volume_state}")
    table.add_row("Momentum", f"{res.momentum_streak} bars ({res.momentum_state.split(' ')[-1]})")

    if res.ml_result:
        ml = res.ml_result
        table.add_row("ML Predictor", f"[green]P(Bull) {ml.prob_bullish:.0%}[/green] | [red]P(Bear) {ml.prob_bearish:.0%}[/red]")

    return Panel(table, title="[bold]3. KEY LEVELS & ML RADAR[/bold]", border_style="cyan", box=box.ROUNDED)


def create_trade_setup_card(res: AnalysisResult) -> Panel:
    setup = res.trade_setup
    if setup:
        is_long = setup.direction == "LONG"
        color = "green" if is_long else "red"

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        sl_dist = abs(setup.stop_loss - setup.entry_price) / max(1e-8, setup.entry_price) * 100
        tp1_dist = abs(setup.tp1_price - setup.entry_price) / max(1e-8, setup.entry_price) * 100
        tp2_dist = abs(setup.tp2_price - setup.entry_price) / max(1e-8, setup.entry_price) * 100

        col1 = (
            f"[bold white]Setup Type:[/bold white] {setup.setup_type.value} ([bold {color}]{setup.direction}[/bold {color}])\n"
            f"[bold white]Entry Target:[/bold white] [bold cyan]${setup.entry_price:,.2f}[/bold cyan]\n"
            f"[bold white]Stop Loss:[/bold white] [red]${setup.stop_loss:,.2f}[/red] (-{sl_dist:.2f}%)\n"
            f"[bold white]Invalidation:[/bold white] [dim]{setup.invalidation_reason}[/dim]"
        )

        col2 = (
            f"[bold white]Status:[/bold white] [bold green]QUALIFIED (R:R >= 1.80 ENFORCED)[/bold green]\n"
            f"[bold white]Take Profit 1:[/bold white] [green]${setup.tp1_price:,.2f}[/green] (+{tp1_dist:.2f}%) [dim](1:{setup.risk_reward_tp1:.1f})[/dim]\n"
            f"[bold white]Take Profit 2:[/bold white] [bold green]${setup.tp2_price:,.2f}[/bold green] (+{tp2_dist:.2f}%) [dim](1:{setup.risk_reward_tp2:.1f})[/dim]\n"
            f"[bold white]Effective Blended R:R:[/bold white] [bold yellow]1 : {setup.effective_rr:.2f}[/bold yellow]"
        )
        grid.add_row(col1, col2)

        track = f"[red][SL ${setup.stop_loss:,.0f}][/red] <---> [cyan][ENTRY ${setup.entry_price:,.0f}][/cyan] --------> [green][TP1 ${setup.tp1_price:,.0f}][/green] ------------> [bold green][TP2 ${setup.tp2_price:,.0f}][/bold green]"
        return Panel(Group(grid, Text(""), Text.from_markup(track, justify="center")), title=f"[{color}]* ACTIONABLE TRADE SETUP PLAN [{setup.direction}][/{color}]", border_style=color, box=box.ROUNDED)
    else:
        content = (
            "[bold yellow]No trade setup currently passes the strict institutional Effective R:R >= 1.80 filter gate.[/bold yellow]\n"
            f"[dim]Current Price: ${res.current_price:,.2f} | Market Structure: {res.trend_detail} | Confluence: {res.bias} ({res.confidence}%)\n"
            "Waiting for high-expectancy FVG pullback, Order Block retest, or clean liquidity sweep...[/dim]"
        )
        return Panel(content, title="[yellow]TRADE SETUP PLAN [CAPITAL PRESERVATION MODE][/yellow]", border_style="yellow", box=box.ROUNDED)


def print_single_report(res: AnalysisResult):
    """Prints a beautiful, clean single analysis report without screen clearing."""
    if HAS_RICH:
        console.print(create_header_panel(res))

        # Spacious 2-column layout for MTF and SMC
        cards_table = Table.grid(expand=True)
        cards_table.add_column(ratio=1)
        cards_table.add_column(ratio=1)
        cards_table.add_row(create_mtf_panel(res), create_smc_panel(res))
        console.print(cards_table)

        # Key Levels & ML Radar card
        console.print(create_srp_panel(res))

        # Actionable Trade Setup Plan card
        console.print(create_trade_setup_card(res))
    else:
        print("=" * 70)
        print(f"PAIR: {res.symbol} ({res.timeframe.upper()}) | PRICE: ${res.current_price:,.2f} | BIAS: {res.bias} ({res.confidence}%)")
        print(f"STRUCTURE: {res.trend_detail} | VOLUME: {res.volume_state}")
        print(f"LEVELS: Support ${res.nearest_support or 0:,.2f} | Resistance ${res.nearest_resistance or 0:,.2f}")
        if res.trade_setup:
            ts = res.trade_setup
            print(f"SETUP: {ts.direction} {ts.setup_type.value} | Entry: ${ts.entry_price:,.2f} | SL: ${ts.stop_loss:,.2f} | TP1: ${ts.tp1_price:,.2f} | R:R {ts.effective_rr:.2f}")
        print("=" * 70)


def create_logs_panel(logs: List[str]) -> Panel:
    """Backwards-compatible helper for activity logs panel."""
    content = Text()
    recent = logs[-4:] if len(logs) >= 4 else logs
    for line in recent:
        content.append(str(line) + "\n")
    if not recent:
        content.append("[dim]Waiting for incoming market events...[/dim]\n")
    return Panel(content, title="[dim]RECENT ACTIVITY LOGS[/dim]", border_style="dim", box=box.ROUNDED)


def render_cockpit(res: AnalysisResult, logs: Optional[List[str]] = None):
    """Backwards-compatible helper returning a composite widget layout."""
    cards = Table.grid(expand=True)
    cards.add_column(ratio=1)
    cards.add_column(ratio=1)
    cards.add_row(create_mtf_panel(res), create_smc_panel(res))
    return Panel(Group(create_header_panel(res), cards, create_srp_panel(res), create_trade_setup_card(res)), box=box.ROUNDED)


def render_text_cockpit(res: AnalysisResult, logs: Optional[List[str]] = None):
    """Backwards-compatible text renderer."""
    print_single_report(res)


# ============================================================================
# Live Streaming Watcher Mode
# ============================================================================

def run_live_stream(symbol: str, timeframe: str, engine: PriceActionEngine, client: BinanceClient):
    """Streams live real-time candle updates cleanly."""
    if HAS_RICH:
        console.print(Panel(
            f"[bold green]Starting Live Real-Time Stream for {symbol} ({timeframe})...[/bold green]\n"
            "[italic white]Listening to Binance Kline WebSocket stream tick-by-tick.[/italic white]\n"
            "[bold yellow]Press Ctrl+C at any time to exit back to the Interactive Assistant.[/bold yellow]",
            box=box.ROUNDED,
            border_style="green"
        ))
    else:
        print(f"Starting Live Real-Time Stream for {symbol} ({timeframe})... Press Ctrl+C to stop.")

    last_time = 0.0
    last_p = 0.0

    def on_update(candle):
        nonlocal last_time, last_p
        try:
            engine.update_candle(candle)
            now = time.time()
            curr_p = candle["close"]
            is_c = candle.get("is_closed", False)

            # Update on closed candle, or 0.05% price move, or every 3 seconds
            p_moved = abs(curr_p - last_p) / max(1e-8, last_p) * 100 if last_p > 0 else 0
            if is_c or p_moved >= 0.05 or (now - last_time >= 3.0):
                res = engine.analyze(symbol, timeframe)
                if res:
                    clear_screen()
                    print_single_report(res)
                    if HAS_RICH:
                        console.print("[dim white]-- Live WebSocket Active | Press Ctrl+C to return to Chat --[/dim white]")
                    last_p = curr_p
                    last_time = now
        except Exception as e:
            logger.error(f"Stream update error: {e}")

    client.start_stream(on_update=on_update)

    try:
        while client.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        client.stop()
        clear_screen()
        if HAS_RICH:
            console.print("[bold yellow]Exited live stream. Returned to Interactive AI Assistant.[/bold yellow]\n")
        else:
            print("\nExited live stream. Returned to Interactive AI Assistant.\n")


# ============================================================================
# Interactive AI Assistant / Chat Interface
# ============================================================================

def run_interactive_assistant(default_symbol: str = "BTCUSDT", default_timeframe: str = "1h"):
    """
    Main user interface providing an interactive conversational & command prompt.
    """
    clear_screen()
    active_symbol = default_symbol
    active_timeframe = default_timeframe

    if HAS_RICH:
        console.print(Panel(
            "[bold cyan]AI PRICE ACTION ASSISTANT v2.0[/bold cyan] - [bold white]INSTITUTIONAL EDITION[/bold white]\n"
            "[italic white]Smart Money Concepts (SMC), Volume Profile, KDE S/R, Machine Learning & Live Confluence[/italic white]\n\n"
            "[bold green]QUICK COMMANDS & ACTIONS:[/bold green]\n"
            "  * Type any coin name to analyze: [bold yellow]btc[/bold yellow], [bold yellow]eth[/bold yellow], [bold yellow]sol[/bold yellow], [bold yellow]bnb[/bold yellow], [bold yellow]doge[/bold yellow], [bold yellow]ada[/bold yellow]\n"
            "  * [bold cyan]stream[/bold cyan] or [bold cyan]live[/bold cyan]     : Launch persistent real-time streaming cockpit\n"
            "  * [bold cyan]setup[/bold cyan] or [bold cyan]signal[/bold cyan]    : View active institutional trade setup plan\n"
            "  * [bold cyan]smc[/bold cyan]                : Show Smart Money Concepts details (FVG, OB, Sweeps)\n"
            "  * [bold cyan]levels[/bold cyan]             : View Support, Resistance & Volume Profile (POC/VAH/VAL)\n"
            "  * [bold cyan]ml[/bold cyan]                 : View Machine Learning prediction & probability breakdown\n"
            "  * [bold cyan]tf <15m|1h|4h|1d>[/bold cyan]  : Switch active timeframe (e.g. 'tf 15m')\n"
            "  * [bold cyan]backtest[/bold cyan]           : Run rapid strategy backtest benchmark\n"
            "  * [bold cyan]help[/bold cyan]               : Show instructions\n"
            "  * [bold cyan]exit[/bold cyan] or [bold cyan]quit[/bold cyan]       : Close application",
            box=box.ROUNDED,
            border_style="cyan"
        ))
    else:
        print("=" * 70)
        print("AI PRICE ACTION ASSISTANT v2.0 - INSTITUTIONAL EDITION")
        print("Commands: <coin>, stream, setup, smc, levels, ml, tf <interval>, backtest, help, exit")
        print("=" * 70)

    # Prepare engines
    engine = PriceActionEngine(max_candles=DEFAULT_CANDLE_LIMIT)
    client = BinanceClient(symbol=active_symbol, interval=active_timeframe)

    def load_market_data(sym: str, tf: str) -> Optional[AnalysisResult]:
        nonlocal client, engine
        if HAS_RICH:
            console.print(f"[dim]Fetching live market data for [bold cyan]{sym}[/bold cyan] ({tf})...[/dim]")
        else:
            print(f"Fetching live market data for {sym} ({tf})...")

        client = BinanceClient(symbol=sym, interval=tf)
        try:
            hist_df = client.fetch_historical_klines(limit=150)
            engine.set_history(hist_df)
        except Exception as e:
            local_file = os.path.join("data", "historical", f"{sym}_{tf}.csv")
            if os.path.exists(local_file):
                hist_df = pd.read_csv(local_file)
                engine.set_history(hist_df)
            else:
                if HAS_RICH:
                    console.print(f"[bold red]Error loading {sym}: {e}[/bold red]")
                else:
                    print(f"Error loading {sym}: {e}")
                return None

        return engine.analyze(sym, tf)

    # Initial Analysis
    current_res = load_market_data(active_symbol, active_timeframe)
    if current_res:
        print_single_report(current_res)

    # Interactive Loop
    while True:
        try:
            prompt_text = f"\n[bold cyan]AI Assistant[/bold cyan] [{active_symbol} {active_timeframe}] > "
            if HAS_RICH:
                user_msg = console.input(prompt_text).strip()
            else:
                user_msg = input(f"AI Assistant [{active_symbol} {active_timeframe}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nShutting down gracefully. Goodbye!")
            break

        if not user_msg:
            continue

        cmd = user_msg.lower()

        # Handle Exit
        if cmd in ["exit", "quit", "q", "bye"]:
            if HAS_RICH:
                console.print("[bold yellow]Exiting AI Assistant. Have a profitable day![/bold yellow]")
            else:
                print("Exiting AI Assistant. Have a profitable day!")
            break

        # Handle Help
        elif cmd in ["help", "?", "commands"]:
            if HAS_RICH:
                console.print(Panel(
                    "• Type any cryptocurrency name: [bold yellow]btc, eth, sol, bnb, doge, xrp, ada[/bold yellow]\n"
                    "• [bold cyan]stream[/bold cyan] or [bold cyan]live[/bold cyan]: Live persistent real-time streaming view\n"
                    "• [bold cyan]setup[/bold cyan]: View institutional Entry, Stop Loss, and Take Profit targets\n"
                    "• [bold cyan]smc[/bold cyan]: View Fair Value Gaps, Order Blocks, Liquidity Sweeps, and BOS/CHoCH\n"
                    "• [bold cyan]levels[/bold cyan]: Support & Resistance zones + Volume Profile (POC/VAH/VAL)\n"
                    "• [bold cyan]ml[/bold cyan]: Calibrated machine learning probability breakdown\n"
                    "• [bold cyan]tf <15m|1h|4h|1d>[/bold cyan]: Change timeframe (e.g. 'tf 15m' or 'tf 4h')\n"
                    "• [bold cyan]backtest[/bold cyan]: Run strategy backtesting benchmarks\n"
                    "• [bold cyan]clear[/bold cyan]: Clear screen\n"
                    "• [bold cyan]exit[/bold cyan]: Quit application",
                    title="[bold]AVAILABLE COMMANDS[/bold]",
                    border_style="cyan",
                    box=box.ROUNDED
                ))
            else:
                print("Commands: btc, eth, sol, stream, setup, smc, levels, ml, tf <interval>, backtest, clear, exit")

        # Handle Clear Screen
        elif cmd in ["clear", "cls"]:
            clear_screen()
            if current_res:
                print_single_report(current_res)

        # Handle Live Stream
        elif cmd in ["stream", "live", "watch"]:
            run_live_stream(active_symbol, active_timeframe, engine, client)
            # When stream stops, reload analysis
            current_res = engine.analyze(active_symbol, active_timeframe)

        # Handle Setup Query
        elif cmd in ["setup", "signal", "trade", "plan"]:
            if current_res:
                if HAS_RICH:
                    console.print(create_trade_setup_card(current_res))
                else:
                    if current_res.trade_setup:
                        ts = current_res.trade_setup
                        print(f"SETUP: {ts.direction} | Entry: ${ts.entry_price:,.2f} | SL: ${ts.stop_loss:,.2f} | TP1: ${ts.tp1_price:,.2f} | R:R: {ts.effective_rr:.2f}")
                    else:
                        print("No trade setup passes the strict Effective R:R >= 1.80 filter gate.")

        # Handle SMC Query
        elif cmd in ["smc", "smart money", "fvg", "orderblock", "ob", "liquidity"]:
            if current_res:
                if HAS_RICH:
                    console.print(create_smc_panel(current_res))
                else:
                    print(f"SMC Footprint for {active_symbol}: Patterns={current_res.patterns}, Structure={current_res.trend_detail}")

        # Handle Key Levels Query
        elif cmd in ["levels", "sr", "support", "resistance", "vp", "volume profile"]:
            if current_res:
                if HAS_RICH:
                    console.print(create_srp_panel(current_res))
                else:
                    print(f"Levels: Support=${current_res.nearest_support or 0:,.2f}, Resistance=${current_res.nearest_resistance or 0:,.2f}")

        # Handle ML Query
        elif cmd in ["ml", "ai", "model", "probability"]:
            if current_res and current_res.ml_result:
                ml = current_res.ml_result
                if HAS_RICH:
                    console.print(Panel(
                        f"[bold white]ML Model Prediction:[/bold white] [{get_bias_color(ml.bias)}]{ml.bias.value}[/{get_bias_color(ml.bias)}]\n"
                        f"[bold white]Confidence Score:[/bold white] [bold cyan]{ml.confidence_score}%[/bold cyan]\n"
                        f"[green]Bullish Win Probability:[/green] {ml.prob_bullish:.1%}\n"
                        f"[red]Bearish Win Probability:[/red] {ml.prob_bearish:.1%}\n"
                        f"[yellow]Neutral/Consolidation:[/yellow] {ml.prob_neutral:.1%}\n"
                        f"[dim]Inference Latency: {ml.inference_latency_ms:.2f} ms | Model: HistGradientBoosting[/dim]",
                        title="[bold]MACHINE LEARNING PREDICTOR[/bold]",
                        border_style="magenta",
                        box=box.ROUNDED
                    ))
                else:
                    print(f"ML: Bias={ml.bias.value} | P(Bull)={ml.prob_bullish:.1%} | P(Bear)={ml.prob_bearish:.1%} | Conf={ml.confidence_score}%")

        # Handle Timeframe Switch
        elif cmd.startswith("tf ") or cmd.startswith("timeframe "):
            parts = cmd.split()
            if len(parts) >= 2:
                req_tf = parts[1].lower()
                if req_tf in SUPPORTED_TIMEFRAMES:
                    active_timeframe = req_tf
                    current_res = load_market_data(active_symbol, active_timeframe)
                    if current_res:
                        clear_screen()
                        print_single_report(current_res)
                else:
                    msg = f"Invalid timeframe '{req_tf}'. Choose from: {', '.join(SUPPORTED_TIMEFRAMES[:6])}"
                    if HAS_RICH:
                        console.print(f"[bold red]{msg}[/bold red]")
                    else:
                        print(msg)

        # Handle Backtest
        elif cmd in ["backtest", "test", "benchmark"]:
            if HAS_RICH:
                console.print(f"[bold yellow]Running strategy backtest benchmark for {active_symbol} ({active_timeframe})...[/bold yellow]")
            else:
                print(f"Running strategy backtest benchmark for {active_symbol} ({active_timeframe})...")
            try:
                from engines.backtest_engine import BacktestEngine
                engine_bt = BacktestEngine()
                df_bt = client.fetch_historical_klines(limit=500)
                bt_res = engine_bt.run(df_bt, symbol=active_symbol, timeframe=active_timeframe)
                m = bt_res.metrics
                if HAS_RICH:
                    t = Table(title=f"Backtest Benchmark: {active_symbol} {active_timeframe}", box=box.ROUNDED)
                    t.add_column("Metric", style="bold white")
                    t.add_column("Value", style="bold cyan")
                    t.add_row("Total Trades", str(m.total_trades))
                    t.add_row("Win Rate %", f"{m.win_rate:.1f}%")
                    t.add_row("Profit Factor", f"{m.profit_factor:.2f}")
                    t.add_row("Net Profit %", f"{m.net_profit_pct:+.2f}%")
                    t.add_row("Max Drawdown %", f"{m.max_drawdown_pct:.2f}%")
                    console.print(t)
                else:
                    print(f"Trades: {m.total_trades} | Win Rate: {m.win_rate:.1f}% | Profit Factor: {m.profit_factor:.2f} | Net Profit: {m.net_profit_pct:+.2f}%")
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[bold red]Backtest error: {e}[/bold red]")
                else:
                    print(f"Backtest error: {e}")

        # Natural Language / Symbol Parsing
        else:
            # Check if user mentioned timeframe
            words = cmd.split()
            found_tf = None
            for w in words:
                if w in SUPPORTED_TIMEFRAMES:
                    found_tf = w
                    break

            # Extract potential symbol
            clean_cmd = cmd.replace("analyze", "").replace("scan", "").replace("check", "").replace("how", "").replace("is", "").replace("looking", "").strip()
            if found_tf:
                clean_cmd = clean_cmd.replace(found_tf, "").strip()

            target_sym = resolve_symbol(clean_cmd)
            target_tf = found_tf if found_tf else active_timeframe

            new_res = load_market_data(target_sym, target_tf)
            if new_res:
                active_symbol = target_sym
                active_timeframe = target_tf
                current_res = new_res
                clear_screen()
                print_single_report(current_res)
            else:
                sugg = suggest_symbols(clean_cmd)
                sugg_str = ", ".join([f"{name} ({sym})" for name, sym in sugg]) if sugg else "None"
                if HAS_RICH:
                    console.print(f"[yellow]Could not resolve '{user_msg}'. Did you mean: {sugg_str}?[/yellow]")
                    console.print("[dim]Type 'help' to see all available commands.[/dim]")
                else:
                    print(f"Could not resolve '{user_msg}'. Suggestions: {sugg_str}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Live Institutional Price Action Scanner & Cockpit v2.0")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Default trading pair")
    parser.add_argument("--timeframe", type=str, default=DEFAULT_TIMEFRAME, help="Default candle timeframe")
    parser.add_argument("--stream", action="store_true", help="Launch directly into real-time streaming view")
    parser.add_argument("--demo", action="store_true", help="Run in single-pass demo mode")
    args = parser.parse_args()

    symbol = resolve_symbol(args.symbol)
    timeframe = args.timeframe.lower() if args.timeframe.lower() in SUPPORTED_TIMEFRAMES else DEFAULT_TIMEFRAME

    if args.demo:
        engine = PriceActionEngine(max_candles=100)
        client = BinanceClient(symbol=symbol, interval=timeframe)
        try:
            df = client.fetch_historical_klines(limit=100)
            engine.set_history(df)
            res = engine.analyze(symbol, timeframe)
            if res:
                print_single_report(res)
        except Exception as e:
            print(f"Demo error: {e}")
        return

    if args.stream:
        engine = PriceActionEngine(max_candles=DEFAULT_CANDLE_LIMIT)
        client = BinanceClient(symbol=symbol, interval=timeframe)
        try:
            df = client.fetch_historical_klines(limit=150)
            engine.set_history(df)
            run_live_stream(symbol, timeframe, engine, client)
        except Exception as e:
            print(f"Streaming error: {e}")
        return

    # Default: Launch the Interactive AI Assistant
    run_interactive_assistant(default_symbol=symbol, default_timeframe=timeframe)


if __name__ == "__main__":
    main()
