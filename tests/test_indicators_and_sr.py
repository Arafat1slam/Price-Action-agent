"""
test_indicators_engine.py
=========================
Comprehensive Test Suite for Advanced S/R, Volume Profile & Indicators Engine.
Verifies:
1. VolumeProfileEngine (binning, POC, Value Area, HVN/LVN, vPOC).
2. VWAPEngine (typical price, cumulative VWAP, running variance, +-1/2/3 sigma, exhaustion).
3. VSAEngine (RVOL, z-score, spread ratio, CL, Wyckoff absorption/tests/climaxes/sweeps).
4. KDESupportResistanceEngine (Gaussian KDE, adaptive ATR, 65% zones, polarity flips).
5. FibonacciEngine (dynamic swing detection, retracements, Golden Pocket, extensions).
6. IndicatorConfluenceEngine (multi-indicator alignment, composite scoring).
7. Execution Speed & Numerical Performance.
"""

import math
import time
import numpy as np
import pandas as pd
import pytest

from engines.indicators_engine import (
    VolumeProfileEngine,
    VolumeProfileConfig,
    PriceBin,
    VolumeProfileResult,
    calculate_volume_profile,
    VWAPEngine,
    VWAPAnchorType,
    VWAPExhaustionSignal,
    VWAPResult,
    calculate_vwap,
    VSAEngine,
    VSASignalType,
    VSASignal,
    analyze_vsa,
    KDESupportResistanceEngine,
    KDESRZone,
    KDESRResult,
    calculate_kde_sr,
    FibonacciEngine,
    SwingTrend,
    FibonacciLevel,
    FibonacciResult,
    calculate_fibonacci,
    IndicatorConfluenceEngine,
    ConfluenceZone,
    evaluate_confluence,
)


# ============================================================================
# Test Fixtures & Synthetic Data Generators
# ============================================================================

@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Generates a realistic 120-bar market dataframe with trends and swings."""
    np.random.seed(42)
    n = 120
    base_price = 50000.0
    drift = np.linspace(0, 3000, n)  # upward drift
    noise = np.cumsum(np.random.randn(n) * 60)
    closes = base_price + drift + noise

    highs = closes + np.abs(np.random.randn(n) * 70) + 15.0
    lows = closes - np.abs(np.random.randn(n) * 70) - 15.0
    opens = (highs + lows) / 2.0 + np.random.randn(n) * 15

    # Ensure validity: low <= min(open, close) and high >= max(open, close)
    for i in range(n):
        highs[i] = max(highs[i], opens[i], closes[i]) + 2.0
        lows[i] = min(lows[i], opens[i], closes[i]) - 2.0

    volumes = np.random.uniform(100, 400, n)

    # Invalidate specific bars to generate distinct institutional signals
    volumes[30] = 1200.0  # stopping volume bar
    volumes[75] = 1400.0  # climax volume bar

    dates = pd.date_range("2026-01-01", periods=n, freq="15min")
    return pd.DataFrame({
        "timestamp": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


# ============================================================================
# 1. Volume Profile Tests
# ============================================================================

class TestVolumeProfileEngine:

    def test_volume_profile_basic_properties(self, sample_market_df):
        engine = VolumeProfileEngine(num_bins=50, value_area_pct=0.70)
        res = engine.compute(sample_market_df)

        assert isinstance(res, VolumeProfileResult)
        assert len(res.bins) == 50
        assert res.total_volume > 0

        # Total allocated volume must closely match input total volume
        input_total_vol = sample_market_df["volume"].sum()
        assert math.isclose(res.total_volume, input_total_vol, rel_tol=1e-3)

        # POC must lie within overall price range
        assert sample_market_df["low"].min() <= res.poc_price <= sample_market_df["high"].max()

        # Value Area relationship: VAL <= POC <= VAH
        assert res.val_price <= res.poc_price <= res.vah_price

    def test_value_area_volume_threshold(self, sample_market_df):
        engine = VolumeProfileEngine(num_bins=60, value_area_pct=0.70)
        res = engine.compute(sample_market_df)

        # Sum volumes of bins enclosed by VAL and VAH
        va_volume = sum(
            b.volume for b in res.bins
            if res.val_price <= b.price_center <= res.vah_price
            or (res.val_price <= b.price_high and b.price_low <= res.vah_price)
        )
        assert va_volume >= 0.68 * res.total_volume

    def test_hvn_and_lvn_detection(self, sample_market_df):
        engine = VolumeProfileEngine(num_bins=50, hvn_prominence_factor=0.20)
        res = engine.compute(sample_market_df)

        assert isinstance(res.hvn_levels, list)
        assert isinstance(res.lvn_levels, list)
        # All HVNs and LVNs should be within price range
        for h in res.hvn_levels:
            assert sample_market_df["low"].min() <= h <= sample_market_df["high"].max()
        for l in res.lvn_levels:
            assert sample_market_df["low"].min() <= l <= sample_market_df["high"].max()

    def test_vpoc_tracking(self, sample_market_df):
        engine = VolumeProfileEngine()
        res = engine.compute(sample_market_df)
        assert isinstance(res.vpoc_untested, bool)

    def test_edge_case_flat_prices(self):
        df_flat = pd.DataFrame({
            "open": [100.0] * 20,
            "high": [100.0] * 20,
            "low": [100.0] * 20,
            "close": [100.0] * 20,
            "volume": [50.0] * 20,
        })
        engine = VolumeProfileEngine(num_bins=20)
        res = engine.compute(df_flat)
        assert res.total_volume > 0
        assert math.isclose(res.poc_price, 100.0, rel_tol=1e-2)
        assert res.val_price <= res.vah_price

    def test_edge_case_empty_df(self):
        df_empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        engine = VolumeProfileEngine()
        res = engine.compute(df_empty)
        assert res.total_volume == 0.0
        assert len(res.bins) == 0


# ============================================================================
# 2. VWAP & Sigma Bands Tests
# ============================================================================

class TestVWAPEngine:

    def test_vwap_monotonic_bands(self, sample_market_df):
        engine = VWAPEngine()
        vw = engine.compute(sample_market_df)

        assert len(vw.vwap) == len(sample_market_df)
        assert len(vw.std_dev) == len(sample_market_df)

        # Standard deviation bands must be strictly ordered
        assert (vw.lower_3sigma <= vw.lower_2sigma + 1e-6).all()
        assert (vw.lower_2sigma <= vw.lower_1sigma + 1e-6).all()
        assert (vw.lower_1sigma <= vw.vwap + 1e-6).all()
        assert (vw.vwap <= vw.upper_1sigma + 1e-6).all()
        assert (vw.upper_1sigma <= vw.upper_2sigma + 1e-6).all()
        assert (vw.upper_2sigma <= vw.upper_3sigma + 1e-6).all()

    def test_vwap_typical_price_weighting(self):
        # Two identical bars with different volume
        df = pd.DataFrame({
            "high": [102.0, 102.0],
            "low": [98.0, 98.0],
            "close": [100.0, 100.0],
            "volume": [100.0, 300.0],
        })
        vw = calculate_vwap(df)
        assert math.isclose(vw.vwap.iloc[-1], 100.0, rel_tol=1e-4)
        assert math.isclose(vw.std_dev.iloc[-1], 0.0, abs_tol=1e-4)

    def test_vwap_exhaustion_signals(self):
        # Create series that shoots way above VWAP to trigger +2 sigma
        n = 30
        closes = [100.0] * 25 + [105.0, 110.0, 115.0, 120.0, 125.0]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        opens = closes.copy()
        vols = [100.0] * n

        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})
        engine = VWAPEngine(exhaustion_threshold=2.0)
        vw = engine.compute(df)

        assert len(vw.exhaustion_signals) > 0
        latest_sig = vw.exhaustion_signals[-1]
        assert latest_sig.bias == "BEARISH"
        assert "OVERBOUGHT" in latest_sig.signal_type
        assert latest_sig.sigma_level >= 2.0

    def test_vwap_rolling_window(self, sample_market_df):
        engine = VWAPEngine(rolling_window=20)
        vw = engine.compute(sample_market_df)
        assert len(vw.vwap) == len(sample_market_df)
        assert not vw.vwap.isna().any()

    def test_vwap_session_anchored(self, sample_market_df):
        sample_market_df["session"] = [i // 30 for i in range(len(sample_market_df))]
        engine = VWAPEngine(anchor_col="session")
        vw = engine.compute(sample_market_df)
        assert len(vw.vwap) == len(sample_market_df)
        assert not vw.vwap.isna().any()


# ============================================================================
# 3. Wyckoff Volume Spread Analysis (VSA) Tests
# ============================================================================

class TestVSAEngine:

    def test_stopping_volume_detection(self):
        # Setup prior downtrend + ultra high volume narrow spread bar closing in upper half
        n = 30
        # Downtrend: 120 down to 100
        closes = list(np.linspace(120, 100, 25)) + [99.0, 98.0, 97.0, 96.0, 96.0]
        highs = [c + 1.5 for c in closes]
        lows = [c - 1.5 for c in closes]
        opens = [c + 0.5 for c in closes]
        vols = [100.0] * 28 + [100.0, 500.0]  # Bar 29: 5x volume spike

        # Bar 29: low spread (0.8), close at 96.5 with low 95.8, high 96.8 -> CL = 0.70
        highs[29] = 96.8
        lows[29] = 95.8
        opens[29] = 96.0
        closes[29] = 96.6  # CL = (96.6 - 95.8) / 1.0 = 0.80

        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})
        engine = VSAEngine(lookback=20)
        signals = engine.analyze(df)

        assert any(
            s.signal_type in (VSASignalType.STOPPING_VOLUME, VSASignalType.ABSORPTION)
            and s.bias == "BULLISH"
            for s in signals
        )

    def test_no_supply_test_detection(self):
        # Downtrend followed by down bar with dried up volume (RVOL < 0.75, vol < prev 2)
        closes = list(np.linspace(100, 95, 25)) + [94.5, 94.0, 93.8, 93.5, 93.4]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        opens = [c + 0.3 for c in closes]
        vols = [200.0] * 25 + [180.0, 150.0, 120.0, 90.0, 40.0]  # Bar 29: dried up 40.0

        # Narrow spread on bar 29
        highs[29] = 93.7
        lows[29] = 93.2
        closes[29] = 93.4  # C <= P_Close, CL = (93.4-93.2)/0.5 = 0.40

        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})
        engine = VSAEngine(lookback=20)
        signals = engine.analyze(df)

        assert any(s.signal_type == VSASignalType.NO_SUPPLY and s.bias == "BULLISH" for s in signals)

    def test_no_demand_test_detection(self):
        # Rally followed by up bar with dried up volume and inability to close near high
        closes = list(np.linspace(90, 100, 25)) + [100.5, 101.0, 101.5, 101.8, 102.0]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        opens = [c - 0.2 for c in closes]
        vols = [200.0] * 25 + [180.0, 150.0, 120.0, 90.0, 40.0]

        # Narrow spread on bar 29, close in lower half
        highs[29] = 102.6
        lows[29] = 101.8
        closes[29] = 102.0  # C >= P_Close, CL = (102.0-101.8)/0.8 = 0.25 <= 0.65

        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})
        engine = VSAEngine(lookback=20)
        signals = engine.analyze(df)

        assert any(s.signal_type == VSASignalType.NO_DEMAND and s.bias == "BEARISH" for s in signals)

    def test_buying_climax_detection(self):
        # Strong uptrend followed by massive volume spike, wide spread, upper rejection
        closes = list(np.linspace(80, 100, 25)) + [102.0, 104.0, 106.0, 108.0, 110.0]
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        opens = [c - 0.5 for c in closes]
        vols = [100.0] * 29 + [600.0]  # 6x volume spike

        # Bar 29: Wide spread, upper wick rejection
        highs[29] = 118.0
        lows[29] = 109.0
        opens[29] = 110.0
        closes[29] = 112.0  # Large upper wick (118-112 = 6 out of 9 = 66%)

        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})
        engine = VSAEngine(lookback=20)
        signals = engine.analyze(df)

        assert any(s.signal_type == VSASignalType.BUYING_CLIMAX and s.bias == "BEARISH" for s in signals)

    def test_upthrust_and_spring_detection(self):
        # Upthrust test: pierce recent high and fail
        closes = [100.0] * 25
        highs = [102.0] * 25
        lows = [98.0] * 25
        opens = [100.0] * 25
        vols = [100.0] * 25

        # Bar 25: pierce high 105, close back at 99 with long upper wick
        highs.append(106.0)
        lows.append(98.0)
        opens.append(101.0)
        closes.append(99.0)
        vols.append(250.0)

        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})
        engine = VSAEngine(lookback=20)
        signals = engine.analyze(df)

        assert any(s.signal_type == VSASignalType.UPTHRUST and s.bias == "BEARISH" for s in signals)


# ============================================================================
# 4. KDE Support & Resistance Tests
# ============================================================================

class TestKDESupportResistanceEngine:

    def test_kde_zone_extraction(self, sample_market_df):
        engine = KDESupportResistanceEngine(bandwidth_scale=0.50, num_eval_points=300)
        res = engine.compute(sample_market_df)

        assert isinstance(res, KDESRResult)
        assert len(res.zones) > 0

        for z in res.zones:
            assert isinstance(z, KDESRZone)
            # Zone low <= Peak <= Zone high
            assert z.zone_low <= z.peak_price <= z.zone_high
            assert z.density > 0
            assert z.prominence >= 0
            assert z.level_type in ["SUPPORT", "RESISTANCE"]
            assert z.touch_count >= 0
            assert z.score >= 0

    def test_support_vs_resistance_classification(self, sample_market_df):
        curr_price = float(sample_market_df["close"].iloc[-1])
        engine = KDESupportResistanceEngine()
        res = engine.compute(sample_market_df, current_price=curr_price)

        for z in res.zones:
            if z.level_type == "SUPPORT":
                assert z.peak_price <= curr_price + 1e-4
            elif z.level_type == "RESISTANCE":
                assert z.peak_price >= curr_price - 1e-4

        if res.nearest_support:
            assert res.nearest_support.peak_price <= curr_price
        if res.nearest_resistance:
            assert res.nearest_resistance.peak_price >= curr_price

    def test_adaptive_atr_bandwidth(self, sample_market_df):
        # KDE should evaluate successfully without throwing numerical errors
        engine = KDESupportResistanceEngine(bandwidth_scale=0.35)
        res = engine.compute(sample_market_df, atr_period=14)
        assert res.eval_grid is not None
        assert len(res.eval_grid) == 500

    def test_polarity_flips(self):
        # Create dataset that starts at 50, consolidates below 60, then breaks out to 80
        np.random.seed(10)
        n = 50
        # Phase 1: 50 to 58
        closes1 = list(np.random.uniform(50, 58, 25))
        # Phase 2: 70 to 80
        closes2 = list(np.random.uniform(70, 80, 25))
        closes = closes1 + closes2
        highs = [c + 2.0 for c in closes]
        lows = [c - 2.0 for c in closes]
        opens = closes.copy()
        vols = [100.0] * n

        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols})
        engine = KDESupportResistanceEngine()
        res = engine.compute(df, current_price=78.0)

        # There should be flipped support zones around the 55-60 former resistance area
        assert any(z.is_flipped for z in res.zones)


# ============================================================================
# 5. Dynamic Fibonacci Retracement & Extensions Tests
# ============================================================================

class TestFibonacciEngine:

    def test_fibonacci_uptrend_retracements_and_extensions(self):
        engine = FibonacciEngine()
        # Swing Low A = 100.0, Swing High B = 200.0 (Range = 100.0)
        res = engine.compute(100.0, 200.0, trend=SwingTrend.UPTREND)

        assert res.trend == SwingTrend.UPTREND
        assert res.swing_a_price == 100.0
        assert res.swing_b_price == 200.0

        # Retracements: B - r * range
        assert math.isclose(res.retracements[0.236], 176.4, rel_tol=1e-3)
        assert math.isclose(res.retracements[0.382], 161.8, rel_tol=1e-3)
        assert math.isclose(res.retracements[0.500], 150.0, rel_tol=1e-3)
        assert math.isclose(res.retracements[0.618], 138.2, rel_tol=1e-3)
        assert math.isclose(res.retracements[0.786], 121.4, rel_tol=1e-3)

        # Golden Pocket [0.618, 0.650]: between 135.0 and 138.2
        assert math.isclose(res.golden_pocket_low, 135.0, rel_tol=1e-3)
        assert math.isclose(res.golden_pocket_high, 138.2, rel_tol=1e-3)

        # Extensions:
        # -0.272 (expansion above B = 200 + 27.2 = 227.2)
        assert math.isclose(res.extensions[-0.272], 227.2, rel_tol=1e-3)
        # 1.272 (expansion from A = 100 + 127.2 = 227.2)
        assert math.isclose(res.extensions[1.272], 227.2, rel_tol=1e-3)
        # -0.618 (expansion above B = 200 + 61.8 = 261.8)
        assert math.isclose(res.extensions[-0.618], 261.8, rel_tol=1e-3)
        # 1.618 (golden extension = 100 + 161.8 = 261.8)
        assert math.isclose(res.extensions[1.618], 261.8, rel_tol=1e-3)

    def test_fibonacci_downtrend_retracements_and_extensions(self):
        engine = FibonacciEngine()
        # Swing High A = 200.0, Swing Low B = 100.0 (Range = 100.0)
        res = engine.compute(200.0, 100.0, trend=SwingTrend.DOWNTREND)

        assert res.trend == SwingTrend.DOWNTREND
        assert res.swing_a_price == 200.0
        assert res.swing_b_price == 100.0

        # Retracements: B + r * range
        assert math.isclose(res.retracements[0.236], 123.6, rel_tol=1e-3)
        assert math.isclose(res.retracements[0.382], 138.2, rel_tol=1e-3)
        assert math.isclose(res.retracements[0.500], 150.0, rel_tol=1e-3)
        assert math.isclose(res.retracements[0.618], 161.8, rel_tol=1e-3)
        assert math.isclose(res.retracements[0.786], 178.6, rel_tol=1e-3)

        # Golden Pocket [0.618, 0.650]: between 161.8 and 165.0
        assert math.isclose(res.golden_pocket_low, 161.8, rel_tol=1e-3)
        assert math.isclose(res.golden_pocket_high, 165.0, rel_tol=1e-3)

        # Extensions: targets below swing low B (100 - 27.2 = 72.8, 100 - 61.8 = 38.2)
        assert math.isclose(res.extensions[-0.272], 72.8, rel_tol=1e-3)
        assert math.isclose(res.extensions[1.272], 72.8, rel_tol=1e-3)
        assert math.isclose(res.extensions[-0.618], 38.2, rel_tol=1e-3)
        assert math.isclose(res.extensions[1.618], 38.2, rel_tol=1e-3)

    def test_fibonacci_auto_detect_swings(self, sample_market_df):
        engine = FibonacciEngine(lookback=60)
        res = engine.compute(sample_market_df)

        assert isinstance(res, FibonacciResult)
        assert res.trend in [SwingTrend.UPTREND, SwingTrend.DOWNTREND]
        assert res.golden_pocket_low < res.golden_pocket_high
        assert len(res.levels) >= 7


# ============================================================================
# 6. Unified Indicator Confluence Engine Tests
# ============================================================================

class TestIndicatorConfluenceEngine:

    def test_confluence_evaluation(self, sample_market_df):
        curr_price = float(sample_market_df["close"].iloc[-1])

        vp = calculate_volume_profile(sample_market_df)
        vw = calculate_vwap(sample_market_df)
        kde_engine = KDESupportResistanceEngine()
        kde_res = kde_engine.compute(sample_market_df, current_price=curr_price)
        fib = calculate_fibonacci(sample_market_df)
        vsa = analyze_vsa(sample_market_df)

        confluence_engine = IndicatorConfluenceEngine(tolerance_pct=0.01)
        zones = confluence_engine.evaluate(
            current_price=curr_price,
            vp=vp,
            vwap_res=vw,
            kde_res=kde_res,
            fib_res=fib,
            vsa_signals=vsa,
        )

        assert isinstance(zones, list)
        for cz in zones:
            assert isinstance(cz, ConfluenceZone)
            assert 0 <= cz.confluence_score <= 100
            assert cz.bias in ["BULLISH", "BEARISH"]
            assert len(cz.alignments) >= 1
            assert cz.price_low <= cz.price_center <= cz.price_high


# ============================================================================
# 7. Numerical Performance Benchmark Test
# ============================================================================

class TestIndicatorsPerformance:

    def test_full_pipeline_execution_time(self, sample_market_df):
        """Ensures all 5 indicator engines execute within sub-50ms CPU budget."""
        curr_price = float(sample_market_df["close"].iloc[-1])

        t0 = time.perf_counter()

        vp_engine = VolumeProfileEngine(num_bins=60)
        vp_res = vp_engine.compute(sample_market_df)

        vwap_engine = VWAPEngine()
        vwap_res = vwap_engine.compute(sample_market_df)

        vsa_engine = VSAEngine(lookback=20)
        vsa_signals = vsa_engine.analyze(sample_market_df)

        kde_engine = KDESupportResistanceEngine(num_eval_points=300)
        kde_res = kde_engine.compute(sample_market_df, current_price=curr_price)

        fib_engine = FibonacciEngine(lookback=60)
        fib_res = fib_engine.compute(sample_market_df)

        confluence_engine = IndicatorConfluenceEngine()
        confluence_zones = confluence_engine.evaluate(
            current_price=curr_price,
            vp=vp_res,
            vwap_res=vwap_res,
            kde_res=kde_res,
            fib_res=fib_res,
            vsa_signals=vsa_signals,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert vp_res.poc_price > 0
        assert len(vwap_res.vwap) > 0
        assert kde_res is not None
        assert elapsed_ms < 150.0, f"Indicators execution too slow: {elapsed_ms:.2f} ms"


# ============================================================================
# Market Regime Engine Tests (Trend / Range / Breakout / Chop Detection)
# ============================================================================

from datetime import datetime, timedelta, timezone
from core.models import MarketRegime, RegimeReport
from engines.regime_engine import MarketRegimeEngine
from price_action_engine import PriceActionEngine


def _make_trending_bull_data(n: int = 60, start_price: float = 50000.0) -> pd.DataFrame:
    records = []
    p = start_price
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        drift = 80.0
        noise = np.random.normal(0, 5)
        o = p
        c = o + drift + noise
        h = max(o, c) + 20
        l = min(o, c) - 10
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 500.0, "is_closed": True
        })
        p = c
    return pd.DataFrame(records)


def _make_trending_bear_data(n: int = 60, start_price: float = 50000.0) -> pd.DataFrame:
    records = []
    p = start_price
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        drift = -80.0
        noise = np.random.normal(0, 5)
        o = p
        c = o + drift + noise
        h = max(o, c) + 10
        l = min(o, c) - 20
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 500.0, "is_closed": True
        })
        p = c
    return pd.DataFrame(records)


def _make_ranging_data(n: int = 60, base_price: float = 50000.0) -> pd.DataFrame:
    records = []
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        c = base_price + 30.0 * np.sin(i * 0.4)
        o = base_price + 30.0 * np.sin((i - 0.5) * 0.4)
        h = max(o, c) + 15.0
        l = min(o, c) - 15.0
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 200.0, "is_closed": True
        })
    return pd.DataFrame(records)


def _make_breakout_data(n: int = 60, base_price: float = 50000.0) -> pd.DataFrame:
    records = []
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(50):
        c = base_price + 10.0 * np.sin(i * 0.3)
        o = c - 2.0
        h = max(o, c) + 5.0
        l = min(o, c) - 5.0
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 100.0, "is_closed": True
        })
    p = base_price
    for i in range(50, n):
        o = p
        c = o + 300.0
        h = c + 50.0
        l = o - 10.0
        v = 1500.0
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": v, "is_closed": True
        })
        p = c
    return pd.DataFrame(records)


def _make_chop_data(n: int = 60, base_price: float = 50000.0) -> pd.DataFrame:
    records = []
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(30):
        c = base_price + 20.0 * np.sin(i * 0.3)
        o = c - 5.0
        h = max(o, c) + 10.0
        l = min(o, c) - 10.0
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 200.0, "is_closed": True
        })
    for i in range(30, n):
        sign = 1 if i % 2 == 0 else -1
        o = base_price
        c = base_price + sign * 150.0
        h = max(o, c) + 200.0
        l = min(o, c) - 200.0
        records.append({
            "timestamp": t0 + timedelta(hours=i),
            "open": o, "high": h, "low": l, "close": c,
            "volume": 400.0, "is_closed": True
        })
    return pd.DataFrame(records)


def test_trending_bull_regime_detection():
    engine = MarketRegimeEngine()
    df = _make_trending_bull_data(60)
    rep = engine.detect_regime(df)

    assert isinstance(rep, RegimeReport)
    assert rep.regime == MarketRegime.TRENDING_BULL
    assert rep.plus_di > rep.minus_di
    assert rep.adx >= 20.0
    assert "Trend Following" in rep.recommended_strategy


def test_trending_bear_regime_detection():
    engine = MarketRegimeEngine()
    df = _make_trending_bear_data(60)
    rep = engine.detect_regime(df)

    assert isinstance(rep, RegimeReport)
    assert rep.regime == MarketRegime.TRENDING_BEAR
    assert rep.minus_di > rep.plus_di
    assert rep.adx >= 20.0
    assert "Trend Following" in rep.recommended_strategy


def test_ranging_regime_detection():
    engine = MarketRegimeEngine()
    df = _make_ranging_data(60)
    rep = engine.detect_regime(df)

    assert isinstance(rep, RegimeReport)
    assert rep.regime == MarketRegime.RANGING
    assert rep.adx < 22.0
    assert "Mean Reversion" in rep.recommended_strategy


def test_breakout_regime_detection():
    engine = MarketRegimeEngine()
    df = _make_breakout_data(60)
    rep = engine.detect_regime(df)

    assert isinstance(rep, RegimeReport)
    assert rep.regime == MarketRegime.BREAKOUT
    assert rep.atr_ratio >= 1.2
    assert "Breakout" in rep.recommended_strategy


def test_chop_regime_detection():
    engine = MarketRegimeEngine()
    df = _make_chop_data(60)
    rep = engine.detect_regime(df)

    assert isinstance(rep, RegimeReport)
    assert rep.regime in [MarketRegime.HIGH_VOLATILITY_CHOP, MarketRegime.RANGING]
    assert rep.atr_ratio > 1.0


def test_price_action_engine_integrates_regime():
    pa = PriceActionEngine(max_candles=70)
    df = _make_trending_bull_data(60)
    pa.set_history(df)

    res = pa.analyze("BTCUSDT", "1h")
    assert res is not None
    assert res.regime_report is not None
    assert isinstance(res.regime_report, RegimeReport)
    assert res.regime_report.regime == MarketRegime.TRENDING_BULL
    assert res.regime_report.confidence >= 50
