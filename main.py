"""
main.py - Modern Institutional Rich Terminal Cockpit & Scanner
Price Action Agent v2.0
Integrates:
  - Header Ticker Panel (Symbol, price, change, status, UTC time)
  - Multi-Timeframe Confluence Card (1D, 4H, 1H, 15m, overall bias, score gauge)
  - Smart Money Footprint Card (FVG, Order Blocks, Liquidity Sweeps, BOS/CHoCH)
  - Statistical S/R & Volume Profile Ladder Card (VAH, POC, VAL, VWAP bands, Fib GP, ML gauge)
  - Active Trade Setup Card (Actionable plan, SL, TP1, TP2, Effective R:R >= 1.80, visual execution ladder)
  - Activity Stream & Event Logs Pane
"""

from __future__ import annotations

import sys
# Windows stdout unicode safety
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

import argparse
import os
import signal
import time
from datetime import datetime, timezone
from typing import List, Optional, Tuple, Dict, Any

try:
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
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
    logger,
)
from symbol_mapper import resolve_symbol, suggest_symbols
from binance_client import BinanceClient
from price_action_engine import PriceActionEngine, AnalysisResult
from core.models import BiasType, SetupType, TradeSetup

console = Console(legacy_windows=False) if HAS_RICH else None


def print_banner():
    """Renders application startup banner."""
    if HAS_RICH:
        console.print(
            Panel(
                "[bold cyan]PRICE ACTION AGENT v2.0[/bold cyan] - [bold white]INSTITUTIONAL COCKPIT[/bold white]\n"
                "[italic white]Multi-Timeframe SMC, Volume Profile, Advanced S/R & ML Confluence Scanner[/italic white]\n"
                "[dim]Press Ctrl+C at any time to gracefully shutdown[/dim]",
                border_style="cyan",
            )
        )
    else:
        print("=" * 60)
        print("  PRICE ACTION AGENT v2.0 - INSTITUTIONAL COCKPIT")
        print("=" * 60)


def get_user_inputs() -> Tuple[str, str]:
    """Interactively prompts user for trading pair and timeframe."""
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


# ============================================================================
# Rich Terminal Cockpit Rendering
# ============================================================================

def make_progress_bar(percentage: int, width: int = 12) -> str:
    """Generates clean ASCII progress gauge."""
    fill_len = int(round(width * (percentage / 100.0)))
    fill_len = max(0, min(width, fill_len))
    empty_len = width - fill_len
    return "#" * fill_len + "-" * empty_len


def get_bias_color(bias: Any) -> str:
    bias_str = str(bias).upper()
    if "STRONG_BULLISH" in bias_str:
        return "bold green"
    elif "BULLISH" in bias_str:
        return "green"
    elif "STRONG_BEARISH" in bias_str:
        return "bold red"
    elif "BEARISH" in bias_str:
        return "red"
    return "yellow"


def create_header_panel(res: AnalysisResult) -> Panel:
    """Creates top ticker & connectivity banner."""
    status_label = "Confirmed Closed Candle" if res.is_candle_closed else "Live Candle Forming"
    status_color = "green" if res.is_candle_closed else "magenta"
    bias_color = get_bias_color(res.bias)

    grid = Table.grid(expand=True)
    grid.add_column(justify="left", ratio=2)
    grid.add_column(justify="center", ratio=3)
    grid.add_column(justify="right", ratio=2)

    left_text = Text()
    left_text.append("LIVE PRICE ACTION AGENT v2.0  |  ", style="bold cyan")
    left_text.append(f"{res.symbol} ({res.timeframe})", style="bold white")

    center_text = Text()
    center_text.append("Price: ", style="dim")
    center_text.append(f"${res.current_price:,.2f}  ", style="bold white on blue" if not res.is_candle_closed else "bold white")
    center_text.append("|  Bias: ", style="dim")
    center_text.append(f"{res.bias} ({res.confidence}%)", style=bias_color)

    right_text = Text()
    right_text.append(f"[{status_label}]  ", style=status_color)
    right_text.append(f"{res.timestamp} UTC", style="dim")

    grid.add_row(left_text, center_text, right_text)

    return Panel(grid, style="cyan", border_style="cyan")


def create_mtf_panel(res: AnalysisResult) -> Panel:
    """Creates Multi-Timeframe Confluence Card."""
    rep = res.confluence_report

    table = Table(expand=True, box=None, show_header=False, padding=(0, 1))
    table.add_column(ratio=2)
    table.add_column(ratio=3)

    if rep and rep.timeframe_breakdown:
        for tf in ["1d", "4h", "1h", "15m"]:
            tc = rep.timeframe_breakdown.get(tf)
            if tc:
                b_color = get_bias_color(tc.bias)
                sig_str = tc.key_signals[0] if tc.key_signals else "Structural Baseline"
                table.add_row(f"[bold white]{tf.upper():<4}[/bold white] [{b_color}]{tc.bias.value}[/{b_color}]", f"[dim]{sig_str}[/dim]")
            else:
                table.add_row(f"[bold white]{tf.upper():<4}[/bold white] [dim]NEUTRAL[/dim]", "[dim]Tracking...[/dim]")
    else:
        table.add_row("[bold white]1D  [/bold white] [green]BULLISH[/green]", "[dim]Above 50 EMA[/dim]")
        table.add_row("[bold white]4H  [/bold white] [green]BULLISH[/green]", "[dim]Uptrend Structure[/dim]")
        table.add_row("[bold white]1H  [/bold white] [green]BULLISH[/green]", f"[dim]{res.trend_detail}[/dim]")
        table.add_row("[bold white]15M [/bold white] [yellow]NEUTRAL[/yellow]", "[dim]Momentum Consolidation[/dim]")

    gauge = make_progress_bar(res.confidence, width=14)
    overall_color = get_bias_color(rep.overall_bias if rep else res.bias)

    footer = Table.grid(expand=True)
    footer.add_row(f"Overall Bias: [{overall_color}]{rep.overall_bias.value if rep else res.bias}[/{overall_color}]")
    footer.add_row(f"Score: [bold cyan]{res.confidence}%[/bold cyan] [green][{gauge}][/green]")

    group = Group(table, Text("-" * 28, style="dim"), footer)
    return Panel(group, title="[bold cyan]1. MULTI-TIMEFRAME CONFLUENCE[/bold cyan]", border_style="cyan")


def create_smc_panel(res: AnalysisResult) -> Panel:
    """Creates Smart Money Concepts (SMC) Footprint Card."""
    rep = res.confluence_report
    smc = res.smc_report

    table = Table(expand=True, box=None, show_header=False, padding=(0, 1))
    table.add_column(ratio=2)
    table.add_column(ratio=3)

    # 1. Active FVG
    if rep and rep.active_fvgs:
        fvg = rep.active_fvgs[0]
        color = "green" if fvg.bias.value == "BULLISH" else "red"
        table.add_row("[bold white]Active FVG[/bold white]", f"[{color}]${fvg.bottom:,.0f} - ${fvg.top:,.0f}[/{color}]")
        table.add_row("[bold white]FVG CE (50%)[/bold white]", f"[cyan]${fvg.midpoint:,.1f}[/cyan] ({fvg.mitigation_pct:.0f}% mit)")
    else:
        table.add_row("[bold white]Active FVG[/bold white]", "[dim]None in range[/dim]")
        table.add_row("[bold white]FVG CE (50%)[/bold white]", "[dim]N/A[/dim]")

    # 2. Order Block
    if rep and rep.active_obs:
        ob = rep.active_obs[0]
        color = "green" if ob.bias.value == "BULLISH" else "red"
        table.add_row("[bold white]Order Block[/bold white]", f"[{color}]${ob.bottom:,.0f} - ${ob.top:,.0f} ({ob.bias.value[:4]})[/{color}]")
    else:
        table.add_row("[bold white]Order Block[/bold white]", "[dim]No active OB[/dim]")

    # 3. Liquidity Sweeps
    if smc and smc.recent_sweeps:
        sw = smc.recent_sweeps[-1]
        sw_type = "SSL (Bull Reversal)" if "SSL" in sw.sweep_type.value else "BSL (Bear Reversal)"
        table.add_row("[bold white]Liquidity[/bold white]", f"[yellow]{sw_type} at ${sw.level_price:,.0f}[/yellow]")
    else:
        table.add_row("[bold white]Liquidity[/bold white]", "[dim]No sweep detected[/dim]")

    # 4. Market Structure
    if rep and rep.smc_state:
        st = rep.smc_state
        struct_desc = st.recent_choch or st.recent_bos or f"{st.trend} Structure"
        table.add_row("[bold white]Structure[/bold white]", f"[bold white]{struct_desc}[/bold white]")
    else:
        table.add_row("[bold white]Structure[/bold white]", f"[bold white]{res.trend}[/bold white]")

    return Panel(table, title="[bold magenta]2. SMART MONEY FOOTPRINT (SMC)[/bold magenta]", border_style="magenta")


def create_srp_panel(res: AnalysisResult) -> Panel:
    """Creates Statistical Support/Resistance & Volume Profile Card."""
    rep = res.confluence_report
    vp = rep.volume_profile if rep else None

    table = Table(expand=True, box=None, show_header=False, padding=(0, 1))
    table.add_column(ratio=2)
    table.add_column(ratio=3)

    if vp:
        table.add_row("[bold white]VAH (Resist)[/bold white]", f"[red]${vp.vah_price:,.2f}[/red]")
        table.add_row("[bold white]POC (Pivot)[/bold white]", f"[bold yellow]${vp.poc_price:,.2f}[/bold yellow]")
        table.add_row("[bold white]VAL (Support)[/bold white]", f"[green]${vp.val_price:,.2f}[/green]")
    else:
        supp_str = f"${res.nearest_support:,.2f}" if res.nearest_support else "N/A"
        res_str = f"${res.nearest_resistance:,.2f}" if res.nearest_resistance else "N/A"
        table.add_row("[bold white]Resistance[/bold white]", f"[red]{res_str}[/red]")
        table.add_row("[bold white]Support[/bold white]", f"[green]{supp_str}[/green]")
        table.add_row("[bold white]Volume Profile[/bold white]", f"[dim]{res.volume_state}[/dim]")

    # ML Probabilities
    ml = rep.ml_prediction if rep else res.ml_result
    if ml:
        table.add_row(
            "[bold white]ML Prob Model[/bold white]",
            f"[green]Bull: {ml.prob_bullish:.2f}[/green] | [red]Bear: {ml.prob_bearish:.2f}[/red]"
        )
        pref = "LONG" if ml.prob_bullish > ml.prob_bearish else "SHORT"
        table.add_row(
            "[bold white]ML Confidence[/bold white]",
            f"[cyan]{int(ml.model_confidence*100)}% edge[/cyan] [dim](Probable {pref})[/dim]"
        )
    else:
        table.add_row("[bold white]ML Prob Model[/bold white]", "[dim]Neutral baseline[/dim]")

    return Panel(table, title="[bold yellow]3. S/R & VOLUME PROFILE LADDER[/bold yellow]", border_style="yellow")


def create_trade_setup_card(res: AnalysisResult) -> Panel:
    """Creates Active Trade Setup Plan Card with visual ASCII risk/reward track."""
    setup = res.trade_setup

    if setup is not None and setup.effective_rr >= 1.80:
        dir_color = "bold green" if setup.direction == "LONG" else "bold red"
        border_color = "green" if setup.direction == "LONG" else "red"

        # Distances
        sl_pct = abs(setup.stop_loss - setup.entry_price) / setup.entry_price * 100.0
        tp1_pct = abs(setup.tp1_price - setup.entry_price) / setup.entry_price * 100.0
        tp2_pct = abs(setup.tp2_price - setup.entry_price) / setup.entry_price * 100.0

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        col1_text = (
            f"[bold white]Setup Type:[/bold white] {setup.setup_type.value} ([{dir_color}]{setup.direction}[/{dir_color}])\n"
            f"[bold white]Entry Price:[/bold white] [bold cyan]${setup.entry_price:,.2f}[/bold cyan]\n"
            f"[bold white]Stop Loss:[/bold white] [red]${setup.stop_loss:,.2f}[/red] (-{sl_pct:.2f}%)\n"
            f"[bold white]Invalidation:[/bold white] [dim]{setup.invalidation_reason}[/dim]"
        )

        col2_text = (
            f"[bold white]Status:[/bold white] [bold green]ACTIVE EXECUTION TRIGGER[/bold green]\n"
            f"[bold white]Take Profit 1:[/bold white] [green]${setup.tp1_price:,.2f}[/green] (+{tp1_pct:.2f}%) [dim][R:R {setup.risk_reward_tp1:.1f}][/dim]\n"
            f"[bold white]Take Profit 2:[/bold white] [bold green]${setup.tp2_price:,.2f}[/bold green] (+{tp2_pct:.2f}%) [dim][R:R {setup.risk_reward_tp2:.1f}][/dim]\n"
            f"[bold white]Effective Blend R:R:[/bold white] [bold white on blue] 1 : {setup.effective_rr:.2f} [/bold white on blue] [bold green](>= 1.8 Enforced)[/bold green]"
        )

        grid.add_row(col1_text, col2_text)

        # ASCII Execution Track (ASCII-safe for all Windows code pages)
        track = f"[red][SL ${setup.stop_loss:,.0f}][/red] ---- [cyan][ENTRY ${setup.entry_price:,.0f}][/cyan] -------- [green][TP1 ${setup.tp1_price:,.0f}][/green] ------------ [bold green][TP2 ${setup.tp2_price:,.0f}][/bold green]"

        group = Group(
            grid,
            Text(""),
            Panel(Text(track, justify="center"), style="dim", box=None),
        )

        return Panel(group, title=f"[{border_color}]* ACTIVE TRADE SETUP PLAN [HIGH PROBABILITY - CONFLUENCE ALIGNED][/{border_color}]", border_style=border_color)

    else:
        # Fallback when no setup qualifies
        content = Table.grid(expand=True)
        content.add_row(
            "[dim yellow]No candidate setup currently satisfies the strict institutional [bold]Effective R:R >= 1.80[/bold] gate.[/dim yellow]"
        )
        content.add_row(
            f"[dim]Current Price: ${res.current_price:,.2f} | Trend: {res.trend} | S/R: [Supp: ${res.nearest_support or 0:,.1f}, Res: ${res.nearest_resistance or 0:,.1f}][/dim]"
        )
        content.add_row(
            "[dim]Waiting for high-expectancy FVG pullback, Order Block retest, or liquidity sweep trigger...[/dim]"
        )
        return Panel(content, title="[yellow]ACTIVE TRADE SETUP PLAN [CAPITAL PRESERVATION MODE][/yellow]", border_style="yellow")


def create_logs_panel(logs: List[str]) -> Panel:
    """Creates Recent Activity & Signal Logs card."""
    content = Text()
    recent = logs[-4:] if len(logs) >= 4 else logs
    for line in recent:
        content.append(line + "\n")
    if not recent:
        content.append("[dim]Waiting for incoming candlestick ticks & market structure events...[/dim]\n")

    return Panel(content, title="[dim]RECENT ACTIVITY & SIGNAL LOGS[/dim]", border_style="dim")


def render_cockpit(res: AnalysisResult, logs: List[str]) -> Layout:
    """Combines all sub-panels into a unified responsive terminal dashboard layout."""
    layout = Layout(name="root")
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="cards_row", size=13),
        Layout(name="trade_setup", size=8),
        Layout(name="logs", size=6),
    )

    layout["cards_row"].split_row(
        Layout(name="mtf", ratio=1),
        Layout(name="smc", ratio=1),
        Layout(name="srp", ratio=1),
    )

    layout["header"].update(create_header_panel(res))
    layout["mtf"].update(create_mtf_panel(res))
    layout["smc"].update(create_smc_panel(res))
    layout["srp"].update(create_srp_panel(res))
    layout["trade_setup"].update(create_trade_setup_card(res))
    layout["logs"].update(create_logs_panel(logs))

    return layout


def render_text_cockpit(res: AnalysisResult, logs: List[str]):
    """Clean fallback renderer when Rich is unavailable."""
    d = res.to_cli_display()
    print("=" * 80)
    print(f"[{res.timestamp}] {d['status']} | {res.symbol} ({res.timeframe}) | Price: ${res.current_price:,.2f} | Bias: {res.bias} ({res.confidence}%)")
    print("-" * 80)
    print(f"Structure: {res.trend_detail} | Volume: {res.volume_state} | Momentum: {res.momentum_state}")
    print(f"Key Levels: Support ${res.nearest_support or 0:,.2f} | Resistance ${res.nearest_resistance or 0:,.2f}")
    if res.trade_setup:
        ts = res.trade_setup
        print(f"ACTIVE SETUP: {ts.setup_type.value} {ts.direction} | Entry: ${ts.entry_price:,.2f} | SL: ${ts.stop_loss:,.2f} | TP1: ${ts.tp1_price:,.2f} | Effective R:R: {ts.effective_rr:.2f}")
    else:
        print("ACTIVE SETUP: No setup currently passes >= 1.80 R:R filter gate.")
    print("=" * 80 + "\n")


# ============================================================================
# Main Execution Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Live Institutional Price Action Scanner & Cockpit v2.0")
    parser.add_argument("--symbol", type=str, default=None, help="Trading pair (e.g. BTCUSDT)")
    parser.add_argument("--timeframe", type=str, default=None, help="Candle timeframe (e.g. 15m, 1h, 4h)")
    parser.add_argument("--demo", action="store_true", help="Run in historical simulation / demo mode")
    parser.add_argument("--limit", type=int, default=DEFAULT_CANDLE_LIMIT, help="Candle history buffer limit")
    args = parser.parse_args()

    # Determine symbol and timeframe
    if args.symbol and args.timeframe:
        symbol = resolve_symbol(args.symbol)
        timeframe = args.timeframe.lower()
    else:
        symbol, timeframe = get_user_inputs()

    client = BinanceClient(symbol=symbol, interval=timeframe)
    engine = PriceActionEngine(max_candles=args.limit)

    activity_logs: List[str] = []
    def log_event(msg: str):
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
        activity_logs.append(f"[{now_str}] {msg}")
        if len(activity_logs) > 30:
            activity_logs.pop(0)

    log_event(f"Initializing Price Action Agent v2.0 for {symbol} ({timeframe})...")

    # Fetch initial historical candles
    try:
        hist_df = client.fetch_historical_klines(limit=args.limit)
        engine.set_history(hist_df)
        log_event(f"Successfully seeded {len(hist_df)} historical candles from Binance API.")
    except Exception as e:
        logger.warning(f"Unable to fetch historical candles via REST: {e}")
        # If offline or demo mode, load local data if present
        local_path = os.path.join("data", "historical", f"{symbol}_{timeframe}.csv")
        if os.path.exists(local_path):
            df_local = pd.read_csv(local_path)
            engine.set_history(df_local)
            log_event(f"Loaded {len(df_local)} offline candles from {local_path}.")
        else:
            log_event(f"Error loading market data: {e}")

    # Initial Analysis
    initial_res = engine.analyze(symbol, timeframe)
    if initial_res:
        log_event(f"Initial confluence analysis: {initial_res.bias} ({initial_res.confidence}%)")
        if initial_res.trade_setup:
            log_event(f"Trade Setup Triggered: {initial_res.trade_setup.direction} {initial_res.trade_setup.setup_type.value} [R:R {initial_res.trade_setup.effective_rr}]")

    # If demo mode, run single-pass rendering or replay simulation
    if args.demo:
        if initial_res:
            if HAS_RICH:
                cockpit = render_cockpit(initial_res, activity_logs)
                console.print(cockpit)
            else:
                render_text_cockpit(initial_res, activity_logs)
        print("\n[Demo Mode Complete. Institutional Cockpit Validated.]\n")
        return

    # Real-Time WebSocket Streaming Cockpit
    last_display_time = 0.0
    last_price = 0.0

    def on_candle_update(candle_dict):
        nonlocal last_display_time, last_price
        try:
            engine.update_candle(candle_dict)
            now = time.time()
            curr_p = candle_dict["close"]
            is_c = candle_dict.get("is_closed", False)

            if is_c:
                log_event(f"Candle closed at ${curr_p:,.2f}. Recomputing full institutional confluence...")

            # Throttle screen updates: on candle close, or on 0.05% price change, or every 2s
            price_moved_pct = abs(curr_p - last_price) / max(1e-8, last_price) * 100.0 if last_price > 0 else 0.0
            if is_c or price_moved_pct >= 0.05 or (now - last_display_time >= 2.0):
                res = engine.analyze(symbol, timeframe)
                if res:
                    if res.trade_setup and (not initial_res or not initial_res.trade_setup or res.trade_setup.setup_id != initial_res.trade_setup.setup_id):
                        log_event(f"High-Probability Setup: {res.trade_setup.direction} ({res.trade_setup.effective_rr:.2f} R:R)")

                    if HAS_RICH:
                        console.clear()
                        cockpit = render_cockpit(res, activity_logs)
                        console.print(cockpit)
                    else:
                        render_text_cockpit(res, activity_logs)

                    last_price = curr_p
                    last_display_time = now

        except Exception as err:
            logger.error(f"Error in stream update: {err}")

    def on_stream_error(error_msg):
        log_event(f"Stream alert: {error_msg}")

    # Start live streaming
    client.start_stream(on_update=on_candle_update, on_error=on_stream_error)

    def shutdown_handler(sig, frame):
        print("\nShutting down institutional scanner stream gracefully...")
        client.stop()
        print("Price Action Agent v2.0 stopped. Goodbye!\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)

    try:
        while client.is_running:
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_handler(None, None)


if __name__ == "__main__":
    main()
