"""
Football Player Performance Analytics System
============================================
A modular, OOP-based framework for analysing EPL & La Liga player data
using StatsBomb open data.

Package structure
─────────────────
  src.config        → Project-wide configuration (singleton)
  src.utils         → Logging, decorators, math helpers
  src.loaders       → StatsBombLoader (data access layer)
  src.models        → Player, Team, League domain classes (Phase 2)
  src.analyzer      → PerformanceAnalyzer (Phase 3 & 4)
  src.visualizations→ Charts, pitch maps, radar plots (Phase 5)
"""

__version__ = "0.1.0"
__author__ = "Football Analytics Project"
