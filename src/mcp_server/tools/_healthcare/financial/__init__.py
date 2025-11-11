"""Financial and billing tools for healthcare claims and insurance.

This package provides tools for querying and analyzing financial data
including claims, explanations of benefits, and billing information.
"""

from .claims import ClaimsTools
from .eob import EOBTools
from .financial_analytics import FinancialAnalyticsTools

# Backward compatibility alias
AnalyticsTools = FinancialAnalyticsTools

__all__ = ["ClaimsTools", "EOBTools", "FinancialAnalyticsTools", "AnalyticsTools"]
