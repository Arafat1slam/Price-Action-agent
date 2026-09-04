"""
core/security.py - Project Integrity & Author Attribution Protection
Guarantees that 'Developed by Arafat Islam' attribution remains intact across the project.
If anyone removes or tampers with the attribution, execution is immediately halted.
"""

import hashlib
from typing import Optional

DEVELOPER_NAME = "Arafat Islam"
DEVELOPER_GITHUB = "https://github.com/Arafat1slam/Price-Action-agent.git"
DEVELOPER_TAG = "Developed by Arafat Islam"
SHORT_DEV_TAG = "Dev: Arafat Islam"

# Cryptographic SHA-256 hash of b"Arafat Islam"
_AUTHOR_HASH = "0ddfaa906c3281e15da467d1fdcb4cb8008998a5b184698277bbc9b59e852f07"


def verify_author_integrity(target_text: Optional[str] = None) -> bool:
    """
    Guarantees that the project developer attribution 'Arafat Islam' cannot be removed or tampered with.
    If tampered with, execution is immediately halted with SystemExit.
    """
    # 1. Cryptographic hash check on author constant
    computed = hashlib.sha256(DEVELOPER_NAME.encode("utf-8")).hexdigest()
    if computed != _AUTHOR_HASH:
        raise SystemExit(
            "\n[CRITICAL ERROR] Unauthorized modification detected!\n"
            "Developer attribution integrity check failed. Project execution halted.\n"
        )

    # 2. Display check: if rendered text is passed, author credit must be present
    if target_text is not None:
        if DEVELOPER_NAME not in target_text:
            raise SystemExit(
                "\n[CRITICAL ERROR] Developer attribution 'Arafat Islam' was removed from display!\n"
                "This software requires visible author attribution to operate. Execution halted.\n"
            )

    return True


def get_attribution_badge() -> str:
    """Returns the standardized developer attribution badge with integrity validation."""
    verify_author_integrity()
    return f"[dim cyan]{SHORT_DEV_TAG}[/dim cyan]"


# Verify integrity on module load
verify_author_integrity()


# ============================================================================
# Enterprise Security Architecture & Institutional Risk Firewall
# ============================================================================

import re
import math
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("PriceActionScanner.Security")


class SecurityBreachException(Exception):
    """Raised when an institutional risk firewall rule or security boundary is breached."""
    pass


class CredentialGuardian:
    """
    Prevents API keys, secret hashes, and credentials from leaking into stdout,
    logs, or crash reports through regex redaction.
    """
    _PATTERNS = [
        (re.compile(r'(api[_-]?key\s*[:=]\s*["\']?)([a-zA-Z0-9_\-]{16,64})(["\']?)', re.IGNORECASE), r'\1[REDACTED_API_KEY]\3'),
        (re.compile(r'(secret[_-]?key\s*[:=]\s*["\']?)([a-zA-Z0-9_\-]{16,64})(["\']?)', re.IGNORECASE), r'\1[REDACTED_SECRET_KEY]\3'),
        (re.compile(r'(bearer\s+)([a-zA-Z0-9_\-\.]{20,})', re.IGNORECASE), r'\1[REDACTED_TOKEN]'),
        (re.compile(r'(binance[_-]?api[_-]?secret\s*[:=]\s*["\']?)([a-zA-Z0-9_\-]{16,64})(["\']?)', re.IGNORECASE), r'\1[REDACTED_SECRET]\3'),
    ]

    @classmethod
    def mask(cls, text: str) -> str:
        """Sanitizes text by redacting all known API key and secret patterns."""
        if not isinstance(text, str):
            text = str(text)
        sanitized = text
        for pattern, replacement in cls._PATTERNS:
            sanitized = pattern.sub(replacement, sanitized)
        return sanitized


class RiskFirewall:
    """
    Institutional Capital Preservation & Circuit Breaker Gate.
    Enforces maximum daily drawdown, single trade risk limits, and flash crash kill-switch.
    """
    def __init__(
        self,
        max_daily_drawdown_pct: float = 5.0,
        max_single_trade_risk_pct: float = 3.0,
        flash_crash_atr_mult: float = 5.0,
    ):
        self.max_daily_drawdown_pct = max_daily_drawdown_pct
        self.max_single_trade_risk_pct = max_single_trade_risk_pct
        self.flash_crash_atr_mult = flash_crash_atr_mult
        
        self.starting_daily_equity: float = 10000.0
        self.current_equity: float = 10000.0
        self.daily_pnl_pct: float = 0.0
        self.is_circuit_breaker_tripped: bool = False
        self.trip_reason: Optional[str] = None
        self.kill_switch_active: bool = False

    def reset_daily(self, new_equity: float = 10000.0):
        """Resets daily circuit breaker for a new trading day."""
        self.starting_daily_equity = max(1.0, float(new_equity))
        self.current_equity = float(new_equity)
        self.daily_pnl_pct = 0.0
        self.is_circuit_breaker_tripped = False
        self.trip_reason = None
        self.kill_switch_active = False

    def update_equity(self, current_equity: float):
        """Updates equity and evaluates daily drawdown circuit breaker."""
        self.current_equity = float(current_equity)
        self.daily_pnl_pct = ((self.current_equity - self.starting_daily_equity) / self.starting_daily_equity) * 100.0
        if self.daily_pnl_pct <= -self.max_daily_drawdown_pct:
            self.is_circuit_breaker_tripped = True
            self.trip_reason = f"Daily drawdown breached limit: {self.daily_pnl_pct:.2f}% <= -{self.max_daily_drawdown_pct:.2f}%"
            logger.critical(f"[RISK FIREWALL] {self.trip_reason}")

    def check_trade_allowed(self, requested_risk_pct: float) -> Tuple[bool, float, str]:
        """
        Validates if a new trade setup is permitted under current risk parameters.
        Returns (is_allowed, clamped_risk_pct, message).
        """
        if self.is_circuit_breaker_tripped:
            return False, 0.0, f"CIRCUIT BREAKER ACTIVE: {self.trip_reason}"

        if self.kill_switch_active:
            return False, 0.0, "KILL SWITCH ACTIVE: Extreme market volatility / flash crash detected"

        # Hard clamp on requested risk per trade
        clamped_risk = min(self.max_single_trade_risk_pct, max(0.1, float(requested_risk_pct)))
        return True, clamped_risk, "Trade allowed within institutional risk bounds"

    def check_flash_crash(self, candle_open: float, candle_close: float, atr: float):
        """Detects anomalous flash crash moves (> 5x ATR) and activates kill-switch."""
        if atr <= 0:
            return
        move = abs(candle_close - candle_open)
        if move >= (self.flash_crash_atr_mult * atr):
            self.kill_switch_active = True
            self.trip_reason = f"Flash crash detected! Move ${move:.2f} >= {self.flash_crash_atr_mult}x ATR (${atr:.2f})"
            logger.critical(f"[KILL SWITCH ACTIVATED] {self.trip_reason}")


class DataAnomalyGuard:
    """
    Validates candle and tick streams against abnormal data,
    including corrupt values, impossible OHLC relationships, and zero volume.
    """
    @staticmethod
    def is_candle_valid(candle: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            o = float(candle.get("open", 0))
            h = float(candle.get("high", 0))
            l = float(candle.get("low", 0))
            c = float(candle.get("close", 0))
            v = float(candle.get("volume", 0))

            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                return False, "Price values must be strictly positive"
            if h < l:
                return False, f"High (${h}) cannot be less than Low (${l})"
            if h < o or h < c:
                return False, f"High (${h}) must be >= Open (${o}) and Close (${c})"
            if l > o or l > c:
                return False, f"Low (${l}) must be <= Open (${o}) and Close (${c})"
            if v < 0 or math.isnan(v):
                return False, "Volume cannot be negative or NaN"

            return True, "Valid"
        except (ValueError, TypeError) as e:
            return False, f"Data type parsing error: {e}"


class InputSanitizer:
    """
    Sanitizes symbols, timeframes, and CLI commands against prompt/input injection.
    """
    _SYMBOL_PATTERN = re.compile(r'^[A-Z0-9_\-]{2,20}$')
    _VALID_TIMEFRAMES = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "1w"}

    @classmethod
    def sanitize_symbol(cls, raw_symbol: str) -> str:
        """Cleans and validates trading symbol, rejecting invalid or injection attempts."""
        if not raw_symbol:
            raise ValueError("Trading symbol cannot be empty")
        clean = raw_symbol.strip().upper()
        # Remove common separators like '/'
        clean = clean.replace("/", "").replace("-", "")
        if not cls._SYMBOL_PATTERN.match(clean):
            raise ValueError(f"Invalid trading symbol format: '{raw_symbol}'")
        return clean

    @classmethod
    def sanitize_timeframe(cls, raw_tf: str) -> str:
        """Validates and canonicalizes timeframe string."""
        if not raw_tf:
            return "1h"
        clean = raw_tf.strip().lower()
        if clean not in cls._VALID_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe '{raw_tf}'. Allowed: {cls._VALID_TIMEFRAMES}")
        return clean


# Global Institutional Risk Firewall Instance
risk_firewall = RiskFirewall()

