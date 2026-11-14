"""
Custom exception handling module.
Provides a detailed traceback formatter for debugging ML pipeline errors.
"""

import sys
import traceback


def _format_error_detail(error: Exception, exc_info) -> str:
    """Build a human-readable error string with file name, line number, and message."""
    _, _, tb = exc_info
    if tb is None:
        return str(error)
    frame = tb.tb_frame
    return (
        f"[{frame.f_code.co_filename} | line {tb.tb_lineno}] {type(error).__name__}: {error}"
    )


class PipelineException(Exception):
    """Wrapper that captures rich traceback context for pipeline failures."""

    def __init__(self, message, exc_info=None):
        super().__init__(message)
        if exc_info is None:
            exc_info = sys.exc_info()
        self.detail = _format_error_detail(message, exc_info)

    def __str__(self):
        return self.detail
