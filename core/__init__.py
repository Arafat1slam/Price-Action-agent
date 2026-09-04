"""Core package for Price Action Agent."""

from core.security import (
    DEVELOPER_NAME,
    DEVELOPER_TAG,
    SHORT_DEV_TAG,
    verify_author_integrity,
    get_attribution_badge,
)

# Enforce developer integrity check upon package initialization
verify_author_integrity()
