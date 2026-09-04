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
import threading
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
    from rich.live import Live
    from rich import box
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None
    Live = None

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

        # 1. State Machine Execution Banner
        state = setup.execution_state
        if state == "CONFIRMED_ENTRY_TRIGGER":
            banner_style = "bold black on bright_green"
            banner_text = f"🚨 CONFIRMED ENTRY TRIGGER — EXECUTE {setup.direction} NOW! 🚨"
        elif state == "WAITING_FOR_PRICE":
            banner_style = "bold black on yellow"
            banner_text = "⏳ DO NOT CHASE — WAITING FOR PULLBACK TO ENTRY ZONE ⏳"
        elif state == "IN_ENTRY_ZONE":
            banner_style = "bold white on dark_orange"
            banner_text = "⚠️ IN ENTRY ZONE — MONITORING CANDLE WICK & VOLUME CONFIRMATION ⚠️"
        elif state == "INVALIDATED":
            banner_style = "bold white on red"
            banner_text = "❌ SETUP INVALIDATED — PRICE BREACHED STOP LOSS ❌"
        elif state == "TARGET_HIT":
            banner_style = "bold black on bright_green"
            banner_text = "🎯 TAKE PROFIT TARGET HIT — SECURE PROFITS & TRAIL STOP 🎯"
        else:
            banner_style = "bold white on blue"
            banner_text = f"ACTIVE SETUP: {state}"

        banner = Table.grid(expand=True)
        banner.add_column(justify="center")
        banner.add_row(f"[{banner_style}]  {banner_text}  [/{banner_style}]")
        banner.add_row(f"[bold white]{setup.entry_action}[/bold white]")

        # 2. Setup Parameters Grid
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)

        sl_dist = abs(setup.stop_loss - setup.entry_price) / max(1e-8, setup.entry_price) * 100
        tp1_dist = abs(setup.tp1_price - setup.entry_price) / max(1e-8, setup.entry_price) * 100
        tp2_dist = abs(setup.tp2_price - setup.entry_price) / max(1e-8, setup.entry_price) * 100

        col1 = (
            f"[bold white]Setup Archetype:[/bold white] {setup.setup_type.value} ([bold {color}]{setup.direction}[/bold {color}])\n"
            f"[bold white]Entry Target:[/bold white] [bold cyan]${setup.entry_price:,.2f}[/bold cyan] (Current: [white]${res.current_price:,.2f}[/white] | Dist: [yellow]{setup.entry_distance_pct:+.2f}%[/yellow])\n"
            f"[bold white]Stop Loss:[/bold white] [red]${setup.stop_loss:,.2f}[/red] (-{sl_dist:.2f}%)\n"
            f"[bold white]Invalidation:[/bold white] [dim]{setup.invalidation_reason}[/dim]"
        )

        col2 = (
            f"[bold white]Risk-Reward Ratio:[/bold white] [bold yellow]Effective 1 : {setup.effective_rr:.2f}[/bold yellow] (R:R >= 1.80)\n"
            f"[bold white]Take Profit 1:[/bold white] [green]${setup.tp1_price:,.2f}[/green] (+{tp1_dist:.2f}%) [dim](1:{setup.risk_reward_tp1:.1f})[/dim]\n"
            f"[bold white]Take Profit 2:[/bold white] [bold green]${setup.tp2_price:,.2f}[/bold green] (+{tp2_dist:.2f}%) [dim](1:{setup.risk_reward_tp2:.1f})[/dim]\n"
            f"[bold white]Confluence Bias:[/bold white] [{color}]{setup.confidence_score}% Institutional Conviction[/{color}]"
        )
        grid.add_row(col1, col2)

        # 3. Probabilities and Position Sizing Sub-Panel
        tp1_bar = make_progress_bar(setup.tp1_probability, width=10)
        tp2_bar = make_progress_bar(setup.tp2_probability, width=10)

        prob_grid = Table.grid(expand=True)
        prob_grid.add_column(ratio=1)
        prob_grid.add_column(ratio=1)
        prob_col1 = (
            f"[bold cyan]🎯 TP1 Hit Probability:[/bold cyan] [bold green]{setup.tp1_probability}%[/bold green] [dim][{tp1_bar}][/dim]\n"
            f"[bold cyan]🎯 TP2 Hit Probability:[/bold cyan] [bold green]{setup.tp2_probability}%[/bold green] [dim][{tp2_bar}][/dim]"
        )
        prob_col2 = (
            f"[bold yellow]🛡️ Recommended Risk:[/bold yellow] [bold white]{setup.recommended_risk_pct}%[/bold white] [dim]of total equity[/dim]\n"
            f"[bold yellow]💼 Suggested Size ($10k):[/bold yellow] [bold white]${setup.position_size_usd:,.2f} USD[/bold white]"
        )
        prob_grid.add_row(prob_col1, prob_col2)

        track = f"[red][SL ${setup.stop_loss:,.0f}][/red] <---> [cyan][ENTRY ${setup.entry_price:,.0f}][/cyan] --------> [green][TP1 ${setup.tp1_price:,.0f} ({setup.tp1_probability}%)] [/green] ------------> [bold green][TP2 ${setup.tp2_price:,.0f} ({setup.tp2_probability}%)] [/bold green]"

        return Panel(
            Group(
                banner,
                Text(""),
                grid,
                Text(""),
                Panel(prob_grid, title="[bold white]PROBABILITY & RISK SIZING (CAPITAL PROTECTION)[/bold white]", border_style="dim", box=box.ROUNDED),
                Text(""),
                Text.from_markup(track, justify="center")
            ),
            title=f"[{color}]* ACTIONABLE TRADE SETUP PLAN [{setup.direction}][/{color}]",
            border_style=color,
            box=box.ROUNDED
        )
    else:
        content = (
            "[bold yellow]No trade setup currently passes the strict institutional Effective R:R >= 1.80 filter gate.[/bold yellow]\n"
            f"[dim]Current Price: ${res.current_price:,.2f} | Market Structure: {res.trend_detail} | Confluence: {res.bias} ({res.confidence}%)\n"
            "Continuously scanning every candle (Open, High, Low, Close, Volume) waiting for high-expectancy FVG pullback, Order Block retest, or liquidity sweep...[/dim]"
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
            print(f"STATUS: {ts.execution_state} -> {ts.entry_action}")
            print(f"SETUP: {ts.direction} {ts.setup_type.value} | Entry: ${ts.entry_price:,.2f} | SL: ${ts.stop_loss:,.2f}")
            print(f"TP1: ${ts.tp1_price:,.2f} (Chance: {ts.tp1_probability}%) | TP2: ${ts.tp2_price:,.2f} (Chance: {ts.tp2_probability}%)")
            print(f"RISK SIZING: {ts.recommended_risk_pct}% equity | Position: ${ts.position_size_usd:,.2f} | R:R {ts.effective_rr:.2f}")
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


def build_cockpit_renderable(res: AnalysisResult) -> Group:
    """
    Builds an ultra-compact (~17 line), fixed-height composite cockpit renderable
    for 100% zero-flicker and zero-scroll live streaming on Windows consoles.
    """
    color = get_bias_color(res.bias)
    status_tag = "[green]CLOSED CANDLE[/green]" if res.is_candle_closed else "[magenta]LIVE FORMING[/magenta]"

    # 1. Header (3 lines)
    header_grid = Table.grid(expand=True)
    header_grid.add_column(ratio=2)
    header_grid.add_column(ratio=1, justify="right")
    left_h = f"[bold cyan]{res.symbol}[/bold cyan] ({res.timeframe.upper()}) | Price: [bold white]${res.current_price:,.2f}[/bold white] | Status: {status_tag}"
    utc_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
    right_h = f"Bias: [{color}]{res.bias}[/{color}] ([bold]{res.confidence}%[/bold]) | UTC: [dim]{utc_str}[/dim]"
    header_grid.add_row(left_h, right_h)
    header_panel = Panel(header_grid, box=box.ROUNDED, style="cyan")

    # 2. Left Column: Market Structure & Confluence
    left_table = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
    left_table.add_column("Key", style="bold white", width=14)
    left_table.add_column("Value")

    # Multi-Timeframe Breakdown
    rep = res.confluence_report
    if rep and rep.timeframe_breakdown:
        mtf_items = []
        for tf in ["1d", "4h", "1h", "15m"]:
            if tf in rep.timeframe_breakdown:
                b = rep.timeframe_breakdown[tf].bias.value
                c = get_bias_color(rep.timeframe_breakdown[tf].bias)
                mtf_items.append(f"{tf.upper()}: [{c}]{b}[/{c}]")
        left_table.add_row("MTF Confluence", " | ".join(mtf_items))
    else:
        left_table.add_row("Structure", f"[{color}]{res.bias}[/{color}] - {res.trend_detail}")

    # SMC Features (FVG, Order Block, Liquidity Sweeps)
    smc = res.smc_report
    if smc:
        active_fvgs = getattr(smc, "active_bullish_fvgs", []) + getattr(smc, "active_bearish_fvgs", [])
        if active_fvgs:
            f = active_fvgs[0]
            fc = "green" if getattr(f, "bias", BiasType.BULLISH) == BiasType.BULLISH else "red"
            left_table.add_row("Active FVG", f"[{fc}]${f.bottom:,.0f} - ${f.top:,.0f} (CE: ${f.midpoint:,.0f})[/{fc}]")
        else:
            left_table.add_row("Active FVG", "[dim]None active[/dim]")

        active_obs = getattr(smc, "active_bullish_obs", []) + getattr(smc, "active_bearish_obs", [])
        if active_obs:
            ob = active_obs[0]
            oc = "green" if getattr(ob, "bias", BiasType.BULLISH) == BiasType.BULLISH else "red"
            stag = "Breaker" if getattr(ob, "is_breaker", False) else "Fresh"
            left_table.add_row("Order Block", f"[{oc}]${ob.bottom:,.0f} - ${ob.top:,.0f} ({stag})[/{oc}]")
        else:
            left_table.add_row("Order Block", "[dim]None active[/dim]")

        sweeps = getattr(smc, "recent_sweeps", [])
        if sweeps:
            sw = sweeps[-1]
            sc = "green" if getattr(sw, "bias", BiasType.BULLISH) == BiasType.BULLISH else "red"
            left_table.add_row("Liquidity", f"[{sc}]{sw.sweep_type.value} (${sw.sweep_level:,.0f})[/{sc}]")

    # S/R Channels & Volume State
    res_str = f"${res.nearest_resistance:,.2f}" if res.nearest_resistance else "N/A"
    sup_str = f"${res.nearest_support:,.2f}" if res.nearest_support else "N/A"
    left_table.add_row("S/R Channels", f"[red]Res: {res_str}[/red] | [green]Sup: {sup_str}[/green]")
    left_table.add_row("Volume / Mom", f"{res.volume_state} | Mom: {res.momentum_streak}b")

    # ML Radar Probabilities
    if res.ml_result:
        ml = res.ml_result
        p_str = f"[green]Bull {ml.prob_bullish*100:.0f}%[/green] | [red]Bear {ml.prob_bearish*100:.0f}%[/red] | [yellow]Neu {ml.prob_neutral*100:.0f}%[/yellow]"
        left_table.add_row("ML Radar", p_str)

    left_panel = Panel(left_table, title="[bold cyan]1. MARKET STRUCTURE & CONFLUENCE[/bold cyan]", border_style="cyan", box=box.ROUNDED)

    # 3. Right Column: Actionable Trade Setup Plan
    setup = res.trade_setup
    if setup:
        s_color = "green" if setup.direction == "LONG" else "red"
        right_table = Table(box=box.SIMPLE, show_header=False, expand=True, padding=(0, 1))
        right_table.add_column("Key", style="bold white", width=14)
        right_table.add_column("Value")

        # Dynamic State Machine Execution Banner
        if setup.execution_state == "CONFIRMED_ENTRY_TRIGGER":
            b_style = "bold black on bright_green"
            b_text = f" 🚨 CONFIRMED ENTRY: EXECUTE {setup.direction} NOW! 🚨 "
        elif setup.execution_state == "WAITING_FOR_PRICE":
            b_style = "bold black on yellow"
            b_text = " ⏳ DO NOT CHASE — WAITING FOR RETRACEMENT ⏳ "
        elif setup.execution_state == "TARGET_HIT":
            b_style = "bold black on bright_green"
            b_text = " 🎯 TARGET HIT — SCALE OUT & SECURE PROFITS 🎯 "
        elif setup.execution_state == "INVALIDATED":
            b_style = "bold white on red"
            b_text = " ❌ SETUP INVALIDATED — PRICE HIT STOP LOSS ❌ "
        else:
            b_style = "bold white on dark_orange"
            b_text = " ⚠️ IN ENTRY ZONE — CONFIRMING REACTION ⚠️ "

        right_table.add_row("Execution", f"[{b_style}]{b_text}[/{b_style}]")
        right_table.add_row("Setup Plan", f"[bold {s_color}]{setup.direction}[/bold {s_color}] {setup.setup_type.value} | Effective R:R [yellow]1:{setup.effective_rr:.2f}[/yellow]")
        right_table.add_row("Entry Target", f"[bold cyan]${setup.entry_price:,.2f}[/bold cyan] (Dist: [yellow]{setup.entry_distance_pct:+.2f}%[/yellow])")
        right_table.add_row("Stop Loss", f"[red]${setup.stop_loss:,.2f}[/red]")
        right_table.add_row("Take Profits", f"[green]TP1: ${setup.tp1_price:,.2f}[/green] | [bold green]TP2: ${setup.tp2_price:,.2f}[/bold green]")
        right_table.add_row("Hit Chance", f"[bold cyan]TP1: {setup.tp1_probability}%[/bold cyan] | [bold cyan]TP2: {setup.tp2_probability}%[/bold cyan]")
        right_table.add_row("Risk & Size", f"[yellow]Risk: {setup.recommended_risk_pct}%[/yellow] | Size ($10k): [white]${setup.position_size_usd:,.2f}[/white]")
        right_panel = Panel(right_table, title=f"[{s_color}]2. ACTIONABLE TRADE SETUP [{setup.direction}][/{s_color}]", border_style=s_color, box=box.ROUNDED)
    else:
        right_content = (
            "\n[bold yellow]No trade setup currently passes strict R:R >= 1.80 filter.[/bold yellow]\n\n"
            f"[dim]Current Price: ${res.current_price:,.2f} | Confluence: {res.bias} ({res.confidence}%)\n"
            "Monitoring incoming candles waiting for high-expectancy FVG / OB retest...[/dim]\n"
        )
        right_panel = Panel(right_content, title="[yellow]2. TRADE SETUP [CAPITAL PRESERVATION][/yellow]", border_style="yellow", box=box.ROUNDED)

    body_grid = Table.grid(expand=True)
    body_grid.add_column(ratio=1)
    body_grid.add_column(ratio=1)
    body_grid.add_row(left_panel, right_panel)

    footer = Text.from_markup(
        "[dim white]-- Live WebSocket Active | In-Place Fixed Cockpit (No Blink, No Scroll) | Press Ctrl+C to return to Chat --[/dim white]",
        justify="center"
    )

    return Group(header_panel, body_grid, footer)


# ============================================================================
# Live Streaming Watcher Mode (100% Zero Flicker & Zero Scroll)
# ============================================================================

def run_live_stream(symbol: str, timeframe: str, engine: PriceActionEngine, client: BinanceClient):
    """
    Streams live real-time candle updates cleanly with 100% ZERO FLICKER and ZERO SCROLL.
    Uses Rich Live in alternate screen buffer mode (screen=True, vertical_overflow="crop")
    without invoking os.system('cls'), completely preventing repeating borders or scrolling drift.
    """
    initial_res = engine.analyze(symbol, timeframe)
    if not initial_res:
        try:
            df = client.fetch_historical_klines(limit=100)
            engine.set_history(df)
            initial_res = engine.analyze(symbol, timeframe)
        except Exception:
            pass

    last_time = time.time()
    last_p = initial_res.current_price if initial_res else 0.0
    latest_candle: Optional[Dict[str, Any]] = None
    has_update = False
    update_lock = threading.Lock()

    def on_update(candle):
        nonlocal latest_candle, has_update
        try:
            engine.update_candle(candle)
            with update_lock:
                latest_candle = candle
                has_update = True
        except Exception as e:
            logger.error(f"Stream update error: {e}")

    client.start_stream(on_update=on_update)

    if HAS_RICH and Live is not None and initial_res:
        current_renderable = build_cockpit_renderable(initial_res)
        try:
            with Live(current_renderable, console=console, screen=True, auto_refresh=False, vertical_overflow="crop") as live:
                while client.is_running:
                    now = time.time()
                    should_refresh = False
                    candle = None
                    with update_lock:
                        if has_update:
                            should_refresh = True
                            has_update = False
                            candle = latest_candle

                    if should_refresh and candle:
                        curr_p = candle["close"]
                        is_c = candle.get("is_closed", False)
                        p_moved = abs(curr_p - last_p) / max(1e-8, last_p) * 100 if last_p > 0 else 0
                        # Smooth update on closed candle, or 0.02% price move, or every 1.5 seconds
                        if is_c or p_moved >= 0.02 or (now - last_time >= 1.5):
                            res = engine.analyze(symbol, timeframe)
                            if res:
                                if res.trade_setup:
                                    engine.trade_setup_engine.update_execution_state(
                                        res.trade_setup, current_price=curr_p, candle=candle
                                    )
                                live.update(build_cockpit_renderable(res), refresh=True)
                                last_p = curr_p
                                last_time = now
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
    else:
        # Fallback non-rich terminal mode (zero cls flicker)
        if initial_res:
            print_single_report(initial_res)
        try:
            while client.is_running:
                now = time.time()
                if has_update and (now - last_time >= 3.0):
                    res = engine.analyze(symbol, timeframe)
                    if res:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {symbol} ${res.current_price:,.2f} | Bias: {res.bias} ({res.confidence}%)")
                        last_time = now
                    has_update = False
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass

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
# Main Entry Point & Setup Wizard
# ============================================================================

def prompt_user_startup() -> Tuple[str, str, int]:
    """
    Interactive terminal onboarding questionnaire when running via start.bat:
    1. Market input (e.g. btc, sol, eth, bnb, doge, etc.)
    2. Trading style:
       - 1: Scalp (5m LTF - Fast Execution)
       - 2: Intraday (1h LTF - Day Trading) [Default]
       - 3: Swing (4h LTF - Multi-Day Position)
    3. Execution mode:
       - 1: Continuous Live Auto-Scanner (Real-time candle & confirmation monitor) [Default]
       - 2: Interactive AI Assistant Cockpit (Chat prompt & commands)
    """
    clear_screen()
    if HAS_RICH:
        console.print(Panel(
            "[bold cyan]AI PRICE ACTION ASSISTANT v2.0[/bold cyan] — [bold white]INSTITUTIONAL SETUP WIZARD[/bold white]\n"
            "[italic white]Configure your scanning session in 2 simple steps.[/italic white]",
            box=box.ROUNDED,
            border_style="cyan"
        ))
    else:
        print("=" * 60)
        print("  AI PRICE ACTION ASSISTANT v2.0 - SETUP WIZARD")
        print("=" * 60)

    # 1. Market prompt
    if HAS_RICH:
        console.print("[bold yellow]Step 1: Market Selection[/bold yellow]")
        console.print("  Enter any crypto symbol or shorthand (e.g. [bold green]btc[/bold green], [bold green]sol[/bold green], [bold green]eth[/bold green], [bold green]bnb[/bold green], [bold green]doge[/bold green])")
    else:
        print("Step 1: Market Selection (e.g. btc, sol, eth, bnb)")

    try:
        raw_market = input("  👉 Enter market to scan [default: btc]: ").strip()
    except (EOFError, KeyboardInterrupt):
        raw_market = "btc"

    market = raw_market if raw_market else "btc"
    resolved_sym = resolve_symbol(market)

    # 2. Trading style prompt
    if HAS_RICH:
        console.print("\n[bold yellow]Step 2: Trading Style[/bold yellow]")
        console.print("  [bold cyan]1[/bold cyan] / [bold white]Scalp[/bold white]    : 5m Timeframe (Fast scalping & high frequency execution)")
        console.print("  [bold cyan]2[/bold cyan] / [bold white]Intraday[/bold white] : 1h Timeframe (Day trading, balanced institutional confluence) [Default]")
        console.print("  [bold cyan]3[/bold cyan] / [bold white]Swing[/bold white]    : 4h Timeframe (Multi-day positions, macro structural trends)")
    else:
        print("\nStep 2: Trading Style:")
        print("  1 / Scalp    : 5m Timeframe")
        print("  2 / Intraday : 1h Timeframe [Default]")
        print("  3 / Swing    : 4h Timeframe")

    try:
        raw_style = input("  👉 Enter choice [1/2/3, default: 2]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raw_style = "2"

    if raw_style in ["1", "scalp", "s"]:
        timeframe = "5m"
        style_name = "Scalp (5m)"
    elif raw_style in ["3", "swing", "sw"]:
        timeframe = "4h"
        style_name = "Swing (4h)"
    else:
        timeframe = "1h"
        style_name = "Intraday (1h)"

    # 3. Execution mode prompt
    if HAS_RICH:
        console.print("\n[bold yellow]Step 3: Scanner Execution Mode[/bold yellow]")
        console.print("  [bold cyan]1[/bold cyan] / [bold green]Continuous Live Auto-Scanner[/bold green] (Live tick-by-tick candle & entry trigger monitor) [Default]")
        console.print("  [bold cyan]2[/bold cyan] / [bold white]Interactive AI Assistant[/bold white]     (Interactive chat prompt & manual commands)")
    else:
        print("\nStep 3: Scanner Execution Mode:")
        print("  1 / Continuous Live Auto-Scanner [Default]")
        print("  2 / Interactive AI Assistant")

    try:
        raw_mode = input("  👉 Enter choice [1/2, default: 1]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raw_mode = "1"

    mode = 2 if raw_mode in ["2", "chat", "interactive", "cockpit"] else 1

    if HAS_RICH:
        console.print(f"\n[bold green]✓ Session Initialized:[/bold green] [bold cyan]{resolved_sym}[/bold cyan] | Style: [bold white]{style_name}[/bold white] | Mode: [bold yellow]{'Continuous Live Stream' if mode == 1 else 'Interactive AI Assistant'}[/bold yellow]\n")
    else:
        print(f"\nSession Initialized: {resolved_sym} | Style: {style_name}\n")

    time.sleep(0.8)
    return resolved_sym, timeframe, mode


def main():
    parser = argparse.ArgumentParser(description="Live Institutional Price Action Scanner & Cockpit v2.0")
    parser.add_argument("--symbol", type=str, default=None, help="Trading pair (e.g. BTCUSDT, SOLUSDT)")
    parser.add_argument("--timeframe", type=str, default=None, help="Candle timeframe (e.g. 5m, 15m, 1h, 4h)")
    parser.add_argument("--stream", action="store_true", help="Launch directly into real-time streaming view")
    parser.add_argument("--demo", action="store_true", help="Run in single-pass demo mode")
    parser.add_argument("--no-wizard", action="store_true", help="Skip the startup interactive questionnaire")
    args = parser.parse_args()

    # If demo mode requested:
    if args.demo:
        sym = resolve_symbol(args.symbol or "BTCUSDT")
        tf = args.timeframe.lower() if args.timeframe and args.timeframe.lower() in SUPPORTED_TIMEFRAMES else DEFAULT_TIMEFRAME
        engine = PriceActionEngine(max_candles=100)
        client = BinanceClient(symbol=sym, interval=tf)
        try:
            df = client.fetch_historical_klines(limit=100)
            engine.set_history(df)
            res = engine.analyze(sym, tf)
            if res:
                print_single_report(res)
        except Exception as e:
            print(f"Demo error: {e}")
        return

    # If streaming directly requested from CLI:
    if args.stream:
        sym = resolve_symbol(args.symbol or "BTCUSDT")
        tf = args.timeframe.lower() if args.timeframe and args.timeframe.lower() in SUPPORTED_TIMEFRAMES else DEFAULT_TIMEFRAME
        engine = PriceActionEngine(max_candles=DEFAULT_CANDLE_LIMIT)
        client = BinanceClient(symbol=sym, interval=tf)
        try:
            df = client.fetch_historical_klines(limit=150)
            engine.set_history(df)
            run_live_stream(sym, tf, engine, client)
        except Exception as e:
            print(f"Streaming error: {e}")
        return

    # If CLI explicitly specifies symbol or requests skipping wizard:
    if args.symbol or args.no_wizard:
        sym = resolve_symbol(args.symbol or "BTCUSDT")
        tf = args.timeframe.lower() if args.timeframe and args.timeframe.lower() in SUPPORTED_TIMEFRAMES else DEFAULT_TIMEFRAME
        run_interactive_assistant(default_symbol=sym, default_timeframe=tf)
        return

    # Default flow (e.g. when double clicking start.bat):
    # Ask interactive questionnaire
    symbol, timeframe, mode = prompt_user_startup()

    if mode == 1:
        engine = PriceActionEngine(max_candles=DEFAULT_CANDLE_LIMIT)
        client = BinanceClient(symbol=symbol, interval=timeframe)
        try:
            df = client.fetch_historical_klines(limit=150)
            engine.set_history(df)
            run_live_stream(symbol, timeframe, engine, client)
        except Exception as e:
            logger.error(f"Streaming startup error: {e}")
            run_interactive_assistant(default_symbol=symbol, default_timeframe=timeframe)
    else:
        run_interactive_assistant(default_symbol=symbol, default_timeframe=timeframe)


if __name__ == "__main__":
    main()
