"""Analytics and population-level MCP tools for healthcare data.

This module provides a backward-compatible wrapper for analytics tools,
delegating to domain-specific analytics implementations.

ARCHITECTURE: This is now a wrapper class that uses composition to delegate to:
- ConditionAnalyticsTools from _healthcare/clinical_data/condition_analytics.py
- FinancialAnalyticsTools from _healthcare/financial/financial_analytics.py

This maintains backward compatibility for existing imports while organizing
code by domain (clinical data vs financial).

Key Tools (delegated):
    - analyze_conditions: Population-level condition analysis and trends
    - get_financial_summary: Financial analysis across claims and billing

Design Philosophy:
    - Domain-driven organization (clinical vs financial)
    - Backward compatibility maintained
    - Single unified API for analytics operations
"""

import logging
from typing import Any

from ..base_tool import BaseTool
from ..models import (
    ConditionAnalysisRequest,
    ConditionAnalysisResponse,
    FinancialSummaryRequest,
    FinancialSummaryResponse,
)
from .clinical_data.condition_analytics import ConditionAnalyticsTools
from .financial.financial_analytics import FinancialAnalyticsTools

logger = logging.getLogger(__name__)


class AnalyticsTools(BaseTool):
    """Unified analytics tools wrapper for healthcare data analysis.

    This class provides a backward-compatible interface for analytics operations,
    delegating to domain-specific implementations for condition and financial analysis.

    Inherits optimized connection management from BaseTool, ensuring
    efficient connection pooling and minimal connection overhead.
    """

    def __init__(self):
        """Initialize analytics tools with domain-specific delegates."""
        super().__init__()
        self._condition_analytics = ConditionAnalyticsTools()
        self._financial_analytics = FinancialAnalyticsTools()

    async def analyze_conditions(
        self, request: ConditionAnalysisRequest, security_context: Any = None
    ) -> ConditionAnalysisResponse:
        """Analyze conditions across patient populations.

        Delegates to ConditionAnalyticsTools for condition analysis.

        Args:
            request: Analysis parameters including filters and grouping options
            security_context: Security context for access control and data minimization

        Returns:
            Condition analysis response with records or grouped statistics
        """
        return await self._condition_analytics.analyze_conditions(request, security_context)

    async def get_financial_summary(
        self, request: FinancialSummaryRequest, security_context: Any = None
    ) -> FinancialSummaryResponse:
        """Analyze financial data from claims and explanation of benefits.

        Delegates to FinancialAnalyticsTools for financial analysis.

        Args:
            request: Financial analysis parameters including filters and grouping
            security_context: Security context for access control and data minimization

        Returns:
            Financial summary with records or grouped statistics
        """
        return await self._financial_analytics.get_financial_summary(request, security_context)
