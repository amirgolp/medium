"""
Pattern Analyzer Module
=======================

HLC (Higher-Lower-Close) pattern recognition inspired by Nobel Prize-winning
neural network research on pattern classification.
"""

from .hlc_patterns import HLCPatternAnalyzer, PatternType, EntryZone

__all__ = ["HLCPatternAnalyzer", "PatternType", "EntryZone"]
