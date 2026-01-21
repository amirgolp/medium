"""
Helper Utilities
================

Common helper functions for trading operations.
"""

from typing import Optional


def normalize_symbol(symbol: str) -> str:
    """
    Normalize symbol name.

    Args:
        symbol: Trading symbol

    Returns:
        Normalized symbol string
    """
    return symbol.upper().replace("/", "").replace("_", "")


def get_pip_value(symbol: str) -> float:
    """
    Get pip value for a symbol.

    Args:
        symbol: Trading symbol

    Returns:
        Pip value (0.0001 for most pairs, 0.01 for JPY pairs)
    """
    if "JPY" in symbol.upper():
        return 0.01
    else:
        return 0.0001


def get_point_value(symbol: str) -> float:
    """
    Get point value for a symbol.

    Args:
        symbol: Trading symbol

    Returns:
        Point value (0.00001 for most pairs, 0.001 for JPY pairs)
    """
    if "JPY" in symbol.upper():
        return 0.001
    else:
        return 0.00001


def points_to_pips(points: float, symbol: str) -> float:
    """
    Convert points to pips.

    Args:
        points: Number of points
        symbol: Trading symbol

    Returns:
        Number of pips
    """
    pip_value = get_pip_value(symbol)
    point_value = get_point_value(symbol)
    return points * point_value / pip_value


def pips_to_points(pips: float, symbol: str) -> float:
    """
    Convert pips to points.

    Args:
        pips: Number of pips
        symbol: Trading symbol

    Returns:
        Number of points
    """
    pip_value = get_pip_value(symbol)
    point_value = get_point_value(symbol)
    return pips * pip_value / point_value


def format_price(price: float, symbol: str) -> str:
    """
    Format price for display.

    Args:
        price: Price value
        symbol: Trading symbol

    Returns:
        Formatted price string
    """
    if "JPY" in symbol.upper():
        return f"{price:.3f}"
    else:
        return f"{price:.5f}"


def calculate_price_distance(
    price1: float,
    price2: float,
    symbol: str,
    in_pips: bool = True
) -> float:
    """
    Calculate distance between two prices.

    Args:
        price1: First price
        price2: Second price
        symbol: Trading symbol
        in_pips: Return distance in pips if True, otherwise in price units

    Returns:
        Distance value
    """
    distance = abs(price1 - price2)

    if in_pips:
        pip_value = get_pip_value(symbol)
        return distance / pip_value

    return distance
