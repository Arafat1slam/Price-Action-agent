import pytest
from symbol_mapper import resolve_symbol, suggest_symbols, clean_input

def test_clean_input():
    assert clean_input(" btc / usdt ") == "btcusdt"
    assert clean_input("sol-usdt") == "solusdt"
    assert clean_input("  eth _ usdt  ") == "ethusdt"

def test_resolve_common_names():
    assert resolve_symbol("bitcoin") == "BTCUSDT"
    assert resolve_symbol("btc") == "BTCUSDT"
    assert resolve_symbol("ethereum") == "ETHUSDT"
    assert resolve_symbol("eth") == "ETHUSDT"
    assert resolve_symbol("solana") == "SOLUSDT"
    assert resolve_symbol("sol") == "SOLUSDT"
    assert resolve_symbol("dogecoin") == "DOGEUSDT"
    assert resolve_symbol("doge") == "DOGEUSDT"

def test_resolve_raw_symbols():
    assert resolve_symbol("BTCUSDT") == "BTCUSDT"
    assert resolve_symbol("ethusdt") == "ETHUSDT"
    assert resolve_symbol("ADA") == "ADAUSDT"
    assert resolve_symbol("XRPFDUSD") == "XRPFDUSD"
    assert resolve_symbol("SOLUSDC") == "SOLUSDC"

def test_default_empty():
    assert resolve_symbol("") == "BTCUSDT"
    assert resolve_symbol("   ") == "BTCUSDT"

def test_suggest_symbols():
    suggestions = suggest_symbols("bitcoi")
    assert len(suggestions) > 0
    assert suggestions[0][1] == "BTCUSDT"
