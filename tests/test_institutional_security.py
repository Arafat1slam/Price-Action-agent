"""
tests/test_institutional_security.py - Comprehensive Institutional Security & Risk Firewall Tests
Validates:
  - Credential masking (API keys, secrets, tokens)
  - Daily drawdown circuit breaker tripping & reset
  - Single trade risk limit enforcement and clamping
  - Flash crash kill switch detection
  - Data anomaly candle validation
  - Input & symbol injection sanitization
"""

import pytest
from core.security import (
    CredentialGuardian,
    RiskFirewall,
    DataAnomalyGuard,
    InputSanitizer,
    SecurityBreachException,
    DEVELOPER_NAME,
    verify_author_integrity,
)


def test_credential_masking():
    raw_log = "Error connecting to Binance: api_key=ab12cd34ef56gh78ij90klmn and secret_key=secret1234567890abcdef123456"
    masked = CredentialGuardian.mask(raw_log)
    assert "ab12cd34ef56gh78ij90klmn" not in masked
    assert "secret1234567890abcdef123456" not in masked
    assert "[REDACTED_API_KEY]" in masked
    assert "[REDACTED_SECRET_KEY]" in masked


def test_bearer_token_masking():
    raw_token = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
    masked = CredentialGuardian.mask(raw_token)
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in masked
    assert "[REDACTED_TOKEN]" in masked


def test_risk_firewall_normal_trade_allowed():
    rf = RiskFirewall(max_daily_drawdown_pct=5.0, max_single_trade_risk_pct=3.0)
    allowed, clamped_risk, msg = rf.check_trade_allowed(requested_risk_pct=1.5)
    assert allowed is True
    assert clamped_risk == 1.5


def test_risk_firewall_clamps_oversized_risk():
    rf = RiskFirewall(max_daily_drawdown_pct=5.0, max_single_trade_risk_pct=3.0)
    allowed, clamped_risk, msg = rf.check_trade_allowed(requested_risk_pct=7.5)
    assert allowed is True
    assert clamped_risk == 3.0  # Clamped to hard 3.0%


def test_daily_drawdown_circuit_breaker_trips():
    rf = RiskFirewall(max_daily_drawdown_pct=5.0)
    rf.reset_daily(10000.0)
    # Loss of $600 (-6%)
    rf.update_equity(9400.0)
    assert rf.is_circuit_breaker_tripped is True
    allowed, _, msg = rf.check_trade_allowed(1.0)
    assert allowed is False
    assert "CIRCUIT BREAKER ACTIVE" in msg

    # Test reset
    rf.reset_daily(10000.0)
    assert rf.is_circuit_breaker_tripped is False
    allowed, _, _ = rf.check_trade_allowed(1.0)
    assert allowed is True


def test_flash_crash_kill_switch():
    rf = RiskFirewall(flash_crash_atr_mult=5.0)
    atr = 100.0
    # Normal candle: $120 move (1.2x ATR)
    rf.check_flash_crash(candle_open=50000.0, candle_close=50120.0, atr=atr)
    assert rf.kill_switch_active is False

    # Flash crash: $600 drop (6x ATR)
    rf.check_flash_crash(candle_open=50000.0, candle_close=49400.0, atr=atr)
    assert rf.kill_switch_active is True
    allowed, _, msg = rf.check_trade_allowed(1.0)
    assert allowed is False
    assert "KILL SWITCH ACTIVE" in msg


def test_data_anomaly_guard_validates_clean_candle():
    clean_candle = {
        "open": 64000.0,
        "high": 64500.0,
        "low": 63800.0,
        "close": 64200.0,
        "volume": 125.4,
    }
    is_valid, msg = DataAnomalyGuard.is_candle_valid(clean_candle)
    assert is_valid is True


def test_data_anomaly_guard_rejects_corrupted_candles():
    # Negative price
    bad1 = {"open": -100, "high": 200, "low": -150, "close": 50, "volume": 10}
    assert DataAnomalyGuard.is_candle_valid(bad1)[0] is False

    # High < Low
    bad2 = {"open": 100, "high": 80, "low": 90, "close": 85, "volume": 10}
    assert DataAnomalyGuard.is_candle_valid(bad2)[0] is False

    # Negative volume
    bad3 = {"open": 100, "high": 120, "low": 90, "close": 110, "volume": -5}
    assert DataAnomalyGuard.is_candle_valid(bad3)[0] is False


def test_input_sanitizer_valid_symbols():
    assert InputSanitizer.sanitize_symbol("btc/usdt") == "BTCUSDT"
    assert InputSanitizer.sanitize_symbol("ETH-USDT") == "ETHUSDT"
    assert InputSanitizer.sanitize_symbol("solusdt") == "SOLUSDT"


def test_input_sanitizer_rejects_malicious_inputs():
    with pytest.raises(ValueError):
        InputSanitizer.sanitize_symbol("BTC; rm -rf /")

    with pytest.raises(ValueError):
        InputSanitizer.sanitize_symbol("../../../etc/passwd")

    with pytest.raises(ValueError):
        InputSanitizer.sanitize_symbol("")


def test_input_sanitizer_timeframes():
    assert InputSanitizer.sanitize_timeframe("15M") == "15m"
    assert InputSanitizer.sanitize_timeframe("1H") == "1h"
    with pytest.raises(ValueError):
        InputSanitizer.sanitize_timeframe("999x")
