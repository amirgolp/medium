"""
Cross-Asset Correlation Configuration
======================================

Defines correlation relationships between trading instruments.
"""

from typing import Dict, List


# Correlation groups for multi-asset feature engineering
CORRELATION_MAP: Dict[str, List[str]] = {
    # Major Forex Pairs
    'EURUSD': ['GBPUSD', 'AUDUSD', 'USDCHF'],  # USD majors
    'GBPUSD': ['EURUSD', 'EURGBP', 'GBPJPY'],
    'USDJPY': ['EURJPY', 'GBPJPY', 'AUDJPY'],
    'AUDUSD': ['EURUSD', 'NZDUSD', 'USDCAD'],
    'USDCAD': ['AUDUSD', 'USDCHF'],  # Commodity currencies
    'NZDUSD': ['AUDUSD', 'EURUSD'],
    'USDCHF': ['EURUSD', 'USDCAD'],  # Inverse to EURUSD

    # Cross Pairs
    'EURGBP': ['EURUSD', 'GBPUSD'],
    'EURJPY': ['EURUSD', 'USDJPY'],
    'GBPJPY': ['GBPUSD', 'USDJPY'],
    'AUDJPY': ['AUDUSD', 'USDJPY'],

    # Precious Metals
    'XAUUSD': ['XAGUSD', 'EURUSD'],  # Gold: inverse to USD, follows silver
    'XAGUSD': ['XAUUSD', 'EURUSD'],  # Silver: follows gold
    'XPTUSD': ['XAUUSD', 'XPDUSD'],  # Platinum
    'XPDUSD': ['XPTUSD', 'XAUUSD'],  # Palladium

    # Energy
    'CL': ['USDCAD', 'XAUUSD'],      # Crude Oil (WTI): affects CAD
    'BRENT': ['CL', 'USDCAD'],        # Brent Oil: follows WTI
    'NATGAS': ['CL', 'USDCAD'],       # Natural Gas

    # Base Metals
    'COPPER': ['AUDUSD', 'XAGUSD'],   # Copper: industrial metal
    'ALUMINUM': ['COPPER'],
    'ZINC': ['COPPER'],

    # Agricultural
    'WHEAT': ['CORN', 'SOYBEAN'],
    'CORN': ['WHEAT', 'SOYBEAN'],
    'SOYBEAN': ['CORN', 'WHEAT'],
    'COFFEE': ['SUGAR'],
    'SUGAR': ['COFFEE'],
    'COTTON': ['SUGAR'],
}


# Asset class mapping
ASSET_CLASS: Dict[str, str] = {
    # Forex
    'EURUSD': 'forex', 'GBPUSD': 'forex', 'USDJPY': 'forex',
    'AUDUSD': 'forex', 'NZDUSD': 'forex', 'USDCAD': 'forex',
    'USDCHF': 'forex', 'EURGBP': 'forex', 'EURJPY': 'forex',
    'GBPJPY': 'forex', 'AUDJPY': 'forex',

    # Precious Metals
    'XAUUSD': 'metal', 'XAGUSD': 'metal', 'XPTUSD': 'metal', 'XPDUSD': 'metal',

    # Energy
    'CL': 'energy', 'BRENT': 'energy', 'NATGAS': 'energy',

    # Base Metals
    'COPPER': 'metal', 'ALUMINUM': 'metal', 'ZINC': 'metal',

    # Agricultural
    'WHEAT': 'agriculture', 'CORN': 'agriculture', 'SOYBEAN': 'agriculture',
    'COFFEE': 'agriculture', 'SUGAR': 'agriculture', 'COTTON': 'agriculture',
}


# Commodity specifications
COMMODITY_SPECS = {
    # Precious Metals
    'XAUUSD': {  # Gold
        'tick_size': 0.01,
        'tick_value': 1.0,
        'contract_size': 100,  # 100 oz
        'currency': 'USD',
        'name': 'Gold'
    },
    'XAGUSD': {  # Silver
        'tick_size': 0.001,
        'tick_value': 5.0,
        'contract_size': 5000,  # 5000 oz
        'currency': 'USD',
        'name': 'Silver'
    },
    'XPTUSD': {  # Platinum
        'tick_size': 0.1,
        'tick_value': 5.0,
        'contract_size': 50,  # 50 oz
        'currency': 'USD',
        'name': 'Platinum'
    },
    'XPDUSD': {  # Palladium
        'tick_size': 0.1,
        'tick_value': 10.0,
        'contract_size': 100,  # 100 oz
        'currency': 'USD',
        'name': 'Palladium'
    },

    # Energy
    'CL': {  # Crude Oil WTI
        'tick_size': 0.01,
        'tick_value': 10.0,
        'contract_size': 1000,  # 1000 barrels
        'currency': 'USD',
        'name': 'Crude Oil WTI'
    },
    'BRENT': {  # Brent Oil
        'tick_size': 0.01,
        'tick_value': 10.0,
        'contract_size': 1000,  # 1000 barrels
        'currency': 'USD',
        'name': 'Brent Oil'
    },
    'NATGAS': {  # Natural Gas
        'tick_size': 0.001,
        'tick_value': 10.0,
        'contract_size': 10000,  # 10,000 MMBtu
        'currency': 'USD',
        'name': 'Natural Gas'
    },

    # Base Metals
    'COPPER': {  # Copper
        'tick_size': 0.0001,
        'tick_value': 25.0,
        'contract_size': 25000,  # 25,000 lbs
        'currency': 'USD',
        'name': 'Copper'
    },

    # Agricultural
    'WHEAT': {  # Wheat
        'tick_size': 0.25,
        'tick_value': 12.5,
        'contract_size': 5000,  # 5000 bushels
        'currency': 'USD',
        'name': 'Wheat'
    },
    'CORN': {  # Corn
        'tick_size': 0.25,
        'tick_value': 12.5,
        'contract_size': 5000,  # 5000 bushels
        'currency': 'USD',
        'name': 'Corn'
    },
    'SOYBEAN': {  # Soybeans
        'tick_size': 0.25,
        'tick_value': 12.5,
        'contract_size': 5000,  # 5000 bushels
        'currency': 'USD',
        'name': 'Soybeans'
    },
}


def get_correlations(symbol: str) -> List[str]:
    """
    Get correlated instruments for a symbol.

    Args:
        symbol: Trading symbol

    Returns:
        List of correlated symbols
    """
    return CORRELATION_MAP.get(symbol, [])


def get_asset_class(symbol: str) -> str:
    """
    Get asset class for a symbol.

    Args:
        symbol: Trading symbol

    Returns:
        Asset class ('forex', 'metal', 'energy', 'agriculture')
    """
    return ASSET_CLASS.get(symbol, 'forex')


def get_commodity_spec(symbol: str) -> dict:
    """
    Get commodity specifications.

    Args:
        symbol: Trading symbol

    Returns:
        Dict with tick_size, tick_value, contract_size
    """
    return COMMODITY_SPECS.get(symbol, None)


def is_commodity(symbol: str) -> bool:
    """
    Check if symbol is a commodity.

    Args:
        symbol: Trading symbol

    Returns:
        True if commodity
    """
    return symbol in COMMODITY_SPECS
