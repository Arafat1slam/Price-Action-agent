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
