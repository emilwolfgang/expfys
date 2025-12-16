"""Shared Matplotlib styling utilities."""

from __future__ import annotations

import locale
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter as _PercentFormatter

# Width (inches) applied to every figure. Height may vary per plot.
FIG_WIDTH_IN = 10.0

# Candidates cover common POSIX and Windows locale names.
_SWEDISH_LOCALE_CANDIDATES = (
    "sv_SE.UTF-8",
    "sv_SE.utf8",
    "sv_SE",
    "sv_SE.ISO8859-1",
    "Swedish_Sweden.1252",
)


def _enable_swedish_decimal_separator() -> None:
    """Configure Matplotlib to use comma as decimal separator."""
    for loc in _SWEDISH_LOCALE_CANDIDATES:
        try:
            locale.setlocale(locale.LC_NUMERIC, loc)
        except locale.Error:
            continue
        else:
            plt.rcParams["axes.formatter.use_locale"] = True
            return

    print(
        "Varning: kunde inte aktivera svensk decimalavskiljare "
        "(ingen svensk locale hittades).",
        file=sys.stderr,
    )


def set_fixed_figwidth(fig: plt.Figure, width: float = FIG_WIDTH_IN) -> plt.Figure:
    """Force a consistent figure width while keeping the current height."""
    fig.set_figwidth(width)
    return fig


_enable_swedish_decimal_separator()


class LocalePercentFormatter(_PercentFormatter):
    """Percent formatter that respects the active locale's decimal separator."""

    def __call__(self, x, pos=None):
        formatted = super().__call__(x, pos)
        head, sep, tail = formatted.partition(".")
        if not sep:
            return formatted

        # Split numeric tail (digits) from any suffix (e.g. "%").
        digit_chars = []
        suffix_start = len(tail)
        for idx, ch in enumerate(tail):
            if ch.isdigit():
                digit_chars.append(ch)
                continue
            suffix_start = idx
            break
        suffix = tail[suffix_start:]
        digits = "".join(digit_chars)

        # Drop the decimal part entirely if it only contains zeros.
        if digits and set(digits) == {"0"}:
            return f"{head}{suffix}"

        decimal_point = locale.localeconv().get("decimal_point") or ","
        if decimal_point == ".":
            return formatted
        return f"{head}{decimal_point}{tail}"


def percent_formatter(
    xmax: float = 100,
    decimals: int | None = None,
    symbol: str = "%",
    is_latex: bool = False,
) -> LocalePercentFormatter:
    """Wrapper mirroring matplotlib's PercentFormatter constructor."""
    return LocalePercentFormatter(
        xmax=xmax,
        decimals=decimals,
        symbol=symbol,
        is_latex=is_latex,
    )


__all__ = [
    "FIG_WIDTH_IN",
    "set_fixed_figwidth",
    "percent_formatter",
    "LocalePercentFormatter",
]
