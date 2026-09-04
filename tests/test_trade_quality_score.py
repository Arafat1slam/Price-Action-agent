"""
tests/test_trade_quality_score.py - Unit tests for Phase 5: Trade Quality Scoring (0-100)
"""

import pytest
from core.models import (
    BiasType,
    SetupType,
    TradeSetup,
    TradeQualityScore,
    QualityGrade,
    ConfluenceReport,
    TimeframeConfluence,
    MarketStructureState,
    FairValueGap,
    OrderBlock,
    VolumeProfileZone,
    MLInferenceResult,
    RegimeReport,
    MarketRegime,
)
from engines.trade_setup_engine import TradeSetupEngine


@pytest.fixture
def engine():
    return TradeSetupEngine()


@pytest.fixture
def basic_long_setup():
    return TradeSetup(
        setup_id="TEST-001",
        symbol="BTCUSDT",
        timestamp="2024-01-01T00:00:00",
        setup_type=SetupType.SMC_PULLBACK_FVG,
        direction="LONG",
        entry_price=80000.0,
        stop_loss=79000.0,
        tp1_price=81500.0,
        tp2_price=83000.0,
        risk_reward_tp1=1.50,
        risk_reward_tp2=3.00,
        effective_rr=2.25,
        confidence_score=75,
    )


@pytest.fixture
def basic_short_setup():
    return TradeSetup(
        setup_id="TEST-002",
        symbol="BTCUSDT",
        timestamp="2024-01-01T00:00:00",
        setup_type=SetupType.SMC_ORDER_BLOCK,
        direction="SHORT",
        entry_price=80000.0,
        stop_loss=81000.0,
        tp1_price=78500.0,
        tp2_price=77000.0,
        risk_reward_tp1=1.50,
        risk_reward_tp2=3.00,
        effective_rr=2.25,
        confidence_score=75,
    )


def _make_confluence(
    bias=BiasType.BULLISH,
    confidence=75,
    with_fvgs=False,
    with_obs=False,
    with_vp=False,
    with_mtf=False,
    with_smc_state=False,
):
    fvgs = []
    if with_fvgs:
        fvgs = [FairValueGap(
            top=80500, bottom=79500, midpoint=80000,
            bias=BiasType.BULLISH, created_time="t1",
        )]
    obs = []
    if with_obs:
        obs = [OrderBlock(
            top=80200, bottom=79800, bias=BiasType.BULLISH,
            created_time="t1", volume=1000.0,
        )]
    vp = None
    if with_vp:
        vp = VolumeProfileZone(
            poc_price=80000, vah_price=81000, val_price=79000, total_volume=50000
        )
    tf_breakdown = {}
    if with_mtf:
        for tf in ["4h", "1h", "15m", "5m"]:
            tf_breakdown[tf] = TimeframeConfluence(
                timeframe=tf, bias=bias, score=0.5
            )
    smc_state = None
    if with_smc_state:
        smc_state = MarketStructureState(
            trend="UPTREND",
            recent_bos="BULLISH_BOS",
            recent_choch=None,
            last_swing_high=81000,
            last_swing_low=79000,
        )

    return ConfluenceReport(
        symbol="BTCUSDT",
        timestamp="2024-01-01",
        current_price=80000,
        overall_bias=bias,
        confidence_score=confidence,
        htf_bias=bias,
        ltf_trigger=bias,
        timeframe_breakdown=tf_breakdown,
        smc_state=smc_state,
        active_fvgs=fvgs,
        active_obs=obs,
        volume_profile=vp,
    )


class TestTradeQualityScoring:
    """Tests for the unified 0-100 trade quality scoring system."""

    def test_score_returns_quality_score_object(self, engine, basic_long_setup):
        qs = engine.score_trade_quality(basic_long_setup)
        assert isinstance(qs, TradeQualityScore)
        assert 0 <= qs.raw_score <= 100
        assert isinstance(qs.grade, QualityGrade)

    def test_score_attached_to_setup(self, engine, basic_long_setup):
        qs = engine.score_trade_quality(basic_long_setup)
        assert basic_long_setup.quality_score is qs

    def test_grade_mapping_a_plus(self, engine):
        assert engine._map_grade(90) == QualityGrade.A_PLUS
        assert engine._map_grade(85) == QualityGrade.A_PLUS

    def test_grade_mapping_a(self, engine):
        assert engine._map_grade(80) == QualityGrade.A
        assert engine._map_grade(75) == QualityGrade.A

    def test_grade_mapping_b(self, engine):
        assert engine._map_grade(70) == QualityGrade.B
        assert engine._map_grade(65) == QualityGrade.B

    def test_grade_mapping_c(self, engine):
        assert engine._map_grade(55) == QualityGrade.C
        assert engine._map_grade(45) == QualityGrade.C

    def test_grade_mapping_filtered(self, engine):
        assert engine._map_grade(44) == QualityGrade.FILTERED
        assert engine._map_grade(0) == QualityGrade.FILTERED

    def test_smc_fvg_setup_scores_higher_smc(self, engine, basic_long_setup):
        # SMC FVG setup type should score higher than VALUE_AREA
        va_setup = TradeSetup(
            setup_id="TEST-VA",
            symbol="BTCUSDT",
            timestamp="2024-01-01",
            setup_type=SetupType.VALUE_AREA_MEAN_REVERSION,
            direction="LONG",
            entry_price=80000.0,
            stop_loss=79000.0,
            tp1_price=81500.0,
            tp2_price=83000.0,
            risk_reward_tp1=1.50,
            risk_reward_tp2=3.00,
            effective_rr=2.25,
            confidence_score=75,
        )
        smc_score_fvg = engine._score_smc_component(basic_long_setup, None)
        smc_score_va = engine._score_smc_component(va_setup, None)
        assert smc_score_fvg > smc_score_va

    def test_high_confluence_scores_high(self, engine, basic_long_setup):
        rep = _make_confluence(
            bias=BiasType.BULLISH,
            confidence=85,
            with_fvgs=True,
            with_obs=True,
            with_vp=True,
            with_mtf=True,
            with_smc_state=True,
        )
        regime = RegimeReport(
            regime=MarketRegime.TRENDING_BULL,
            regime_label="Trending Bull",
            confidence=80,
            adx=35.0,
            plus_di=30.0,
            minus_di=15.0,
            atr_ratio=1.2,
            bb_bandwidth_pct=5.0,
            volatility_state="NORMAL",
            recommended_strategy="Trend Following",
        )
        ml = MLInferenceResult(
            prob_bullish=0.75,
            prob_bearish=0.15,
            prob_neutral=0.10,
            model_confidence=0.30,
        )
        qs = engine.score_trade_quality(basic_long_setup, rep, regime, ml)
        assert qs.raw_score >= 65, f"High confluence setup should score >= 65 but got {qs.raw_score}"
        assert qs.grade in (QualityGrade.A_PLUS, QualityGrade.A, QualityGrade.B)

    def test_no_confluence_gets_baseline_score(self, engine, basic_long_setup):
        qs = engine.score_trade_quality(basic_long_setup)
        assert qs.raw_score > 0, "Even without confluence data, should get baseline score"

    def test_component_breakdown_sums_correctly(self, engine, basic_long_setup):
        rep = _make_confluence(bias=BiasType.BULLISH, confidence=70)
        qs = engine.score_trade_quality(basic_long_setup, rep)
        breakdown_sum = sum(qs.component_breakdown.values())
        assert abs(breakdown_sum - qs.raw_score) < 1.0, (
            f"Component sum {breakdown_sum} should match raw score {qs.raw_score}"
        )

    def test_short_with_bear_regime_scores_well(self, engine, basic_short_setup):
        regime = RegimeReport(
            regime=MarketRegime.TRENDING_BEAR,
            regime_label="Trending Bear",
            confidence=80,
            adx=32.0,
            plus_di=12.0,
            minus_di=28.0,
            atr_ratio=1.1,
            bb_bandwidth_pct=4.5,
            volatility_state="NORMAL",
            recommended_strategy="Trend Following Short",
        )
        structure_score = engine._score_structure_component(basic_short_setup, regime)
        assert structure_score >= 70, f"Short in trending bear should score well: {structure_score}"

    def test_ml_component_high_prob(self, engine, basic_long_setup):
        ml = MLInferenceResult(
            prob_bullish=0.80, prob_bearish=0.10, prob_neutral=0.10, model_confidence=0.30
        )
        ml_score = engine._score_ml_component(basic_long_setup, ml)
        assert ml_score >= 80, f"High ML probability should score >= 80: {ml_score}"

    def test_ml_component_no_ml_returns_neutral(self, engine, basic_long_setup):
        ml_score = engine._score_ml_component(basic_long_setup, None)
        assert ml_score == 50.0

    def test_mtf_all_aligned_scores_high(self, engine):
        rep = _make_confluence(bias=BiasType.BULLISH, with_mtf=True)
        mtf_score = engine._score_mtf_component(rep)
        assert mtf_score >= 80, f"All timeframes aligned should score high: {mtf_score}"
