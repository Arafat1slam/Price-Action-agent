"""
tests/test_author_integrity.py - Tests for Developer Attribution & Anti-Tamper Protection
Guarantees that 'Developed by Arafat Islam' attribution is permanently present,
cryptographically validated, and cannot be removed without stopping execution.
"""

import pytest
from rich.console import Console

from core.security import (
    DEVELOPER_NAME,
    DEVELOPER_TAG,
    SHORT_DEV_TAG,
    _AUTHOR_HASH,
    verify_author_integrity,
    get_attribution_badge,
)
from core.models import (
    BiasType,
    SetupType,
    TradeSetup,
    ConfluenceReport,
)
from price_action_engine import PriceActionEngine, AnalysisResult
from main import build_cockpit_renderable, create_header_panel


def _make_mock_analysis():
    setup = TradeSetup(
        setup_id="TEST-001",
        symbol="BTCUSDT",
        timestamp="2026-09-04 01:00:00",
        setup_type=SetupType.SMC_PULLBACK_FVG,
        direction="LONG",
        entry_price=80000.0,
        stop_loss=79000.0,
        tp1_price=81500.0,
        tp2_price=83000.0,
        risk_reward_tp1=1.5,
        risk_reward_tp2=3.0,
        effective_rr=2.25,
        confidence_score=80,
    )
    return AnalysisResult(
        symbol="BTCUSDT",
        timeframe="5m",
        current_price=80500.0,
        is_candle_closed=False,
        bias="BULLISH",
        confidence=80,
        patterns=["Bullish Pin Bar"],
        trend="UPTREND",
        trend_detail="Higher Highs",
        nearest_support=79000.0,
        nearest_resistance=82000.0,
        rough_target=83000.0,
        volume_state="Normal",
        momentum_streak=3,
        momentum_state="Bullish",
        timestamp="01:00:00",
        trade_setup=setup,
    )


class TestAuthorAttributionAndIntegrity:
    """Tests guaranteeing Arafat Islam developer attribution integrity."""

    def test_developer_constant_values(self):
        assert DEVELOPER_NAME == "Arafat Islam"
        assert "Arafat Islam" in DEVELOPER_TAG
        assert "Arafat Islam" in SHORT_DEV_TAG

    def test_cryptographic_hash_valid(self):
        import hashlib
        computed = hashlib.sha256(b"Arafat Islam").hexdigest()
        assert computed == _AUTHOR_HASH

    def test_verify_author_integrity_passes_when_intact(self):
        assert verify_author_integrity() is True

    def test_verify_author_integrity_passes_with_credited_text(self):
        valid_text = "Price Action Cockpit v2.0 | Dev: Arafat Islam | BTCUSDT"
        assert verify_author_integrity(valid_text) is True

    def test_verify_author_integrity_fails_when_attribution_stripped(self):
        stripped_text = "Price Action Cockpit v2.0 | Anonymous | BTCUSDT"
        with pytest.raises(SystemExit) as exc_info:
            verify_author_integrity(stripped_text)
        assert "CRITICAL ERROR" in str(exc_info.value)

    def test_verify_author_integrity_fails_when_constant_tampered(self, monkeypatch):
        import core.security
        monkeypatch.setattr(core.security, "DEVELOPER_NAME", "Unknown Author")
        with pytest.raises(SystemExit) as exc_info:
            core.security.verify_author_integrity()
        assert "CRITICAL ERROR" in str(exc_info.value)

    def test_attribution_badge_format(self):
        badge = get_attribution_badge()
        assert "Arafat Islam" in badge
        assert "Dev: Arafat Islam" in badge

    def test_cockpit_renderable_displays_author_attribution(self):
        res = _make_mock_analysis()
        cockpit = build_cockpit_renderable(res)
        console = Console(record=True, width=120)
        console.print(cockpit)
        output = console.export_text()
        assert "Arafat Islam" in output, "Rendered cockpit must visibly display 'Arafat Islam'"

    def test_header_panel_displays_author_attribution(self):
        res = _make_mock_analysis()
        panel = create_header_panel(res)
        console = Console(record=True, width=120)
        console.print(panel)
        output = console.export_text()
        assert "Arafat Islam" in output, "Header panel must display 'Arafat Islam'"

    def test_price_action_engine_enforces_integrity_on_init(self):
        engine = PriceActionEngine(max_candles=50)
        assert engine is not None

    def test_engine_fails_to_initialize_if_author_tampered(self, monkeypatch):
        import core.security
        monkeypatch.setattr(core.security, "DEVELOPER_NAME", "Impostor")
        with pytest.raises(SystemExit):
            PriceActionEngine(max_candles=50)
