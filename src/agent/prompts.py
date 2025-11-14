"""Agent system prompts and prompt templates.

This module provides centralized management of system prompts and prompt
templates used by the MCP agent orchestrator. Separating prompts from code
allows for easy customization and A/B testing.
"""

import logging

logger = logging.getLogger(__name__)


class SystemPrompts:
    """Collection of system prompts for different agent scenarios."""

    HEALTHCARE_ANALYST = (
        "You are a healthcare data analysis assistant with expertise in patient care, "
        "medical conditions, and pharmaceutical information. "
        "\n\nYour responsibilities:\n"
        "1. Query patient information using available tools\n"
        "2. Analyze medical conditions and treatment patterns\n"
        "3. Review medication histories and drug interactions\n"
        "4. Provide evidence-based insights from healthcare data\n"
        "5. Maintain strict patient privacy and HIPAA compliance\n"
        "\nGuidelines:\n"
        "- Always explain your reasoning and data sources\n"
        "- Flag any concerning patterns or anomalies\n"
        "- Provide context for all findings\n"
        "- Be cautious with sensitive health information\n"
        "- Suggest relevant follow-up queries when appropriate"
    )

    GENERIC_ASSISTANT = (
        "You are a helpful assistant with access to healthcare data tools. "
        "Use the available tools to answer questions accurately and thoroughly. "
        "Always provide context for your findings and explain your reasoning."
    )

    RESEARCH_ANALYST = (
        "You are a research analyst specializing in healthcare data analysis. "
        "Your role is to:\n"
        "1. Identify patterns and trends in healthcare data\n"
        "2. Perform statistical analysis on medical information\n"
        "3. Generate insights for research purposes\n"
        "4. Validate data quality and completeness\n"
        "\nApproach each query systematically and document your methodology."
    )

    @classmethod
    def get_prompt(cls, prompt_type: str) -> str:
        """Get a system prompt by type.

        Args:
            prompt_type: Type of prompt (healthcare_analyst, generic, research_analyst)

        Returns:
            The system prompt string

        Raises:
            ValueError: If prompt_type is not recognized
        """
        prompts = {
            "healthcare_analyst": cls.HEALTHCARE_ANALYST,
            "generic": cls.GENERIC_ASSISTANT,
            "research_analyst": cls.RESEARCH_ANALYST,
        }

        if prompt_type not in prompts:
            raise ValueError(
                f"Unknown prompt type: {prompt_type}. " f"Available: {', '.join(prompts.keys())}"
            )

        return prompts[prompt_type]


class QueryPromptTemplates:
    """Templates for query-specific prompts."""

    PATIENT_SEARCH = (
        "Search for patients matching the criteria: {criteria}. "
        "Return relevant patient information and summarize key findings."
    )

    CONDITION_ANALYSIS = (
        "Analyze medical conditions with the following parameters: {criteria}. "
        "Identify patterns, trends, and any notable findings. "
        "Provide clinical context where relevant."
    )

    MEDICATION_REVIEW = (
        "Review medication history for: {criteria}. "
        "Identify any potential drug interactions or concerns. "
        "Summarize medication patterns and adherence if available."
    )

    DRUG_INFORMATION = (
        "Search for drug information: {criteria}. "
        "Provide therapeutic class, indications, and relevant clinical data."
    )

    CARE_PLAN_ANALYSIS = (
        "Analyze care plans with the following criteria: {criteria}. "
        "Identify patterns in care delivery and treatment approaches. "
        "Flag any gaps or areas for improvement."
    )

    DIAGNOSTIC_REVIEW = (
        "Review diagnostic reports for: {criteria}. "
        "Summarize findings and highlight any abnormal results. "
        "Provide clinical interpretation where appropriate."
    )

    @classmethod
    def get_template(cls, template_type: str) -> str:
        """Get a query prompt template by type.

        Args:
            template_type: Type of template (patient_search, condition_analysis, etc.)

        Returns:
            The prompt template string

        Raises:
            ValueError: If template_type is not recognized
        """
        templates = {
            "patient_search": cls.PATIENT_SEARCH,
            "condition_analysis": cls.CONDITION_ANALYSIS,
            "medication_review": cls.MEDICATION_REVIEW,
            "drug_information": cls.DRUG_INFORMATION,
            "care_plan_analysis": cls.CARE_PLAN_ANALYSIS,
            "diagnostic_review": cls.DIAGNOSTIC_REVIEW,
        }

        if template_type not in templates:
            raise ValueError(
                f"Unknown template type: {template_type}. "
                f"Available: {', '.join(templates.keys())}"
            )

        return templates[template_type]

    @classmethod
    def format_query(cls, template_type: str, **kwargs) -> str:
        """Format a query using a template.

        Args:
            template_type: Type of template to use
            **kwargs: Variables to format into the template

        Returns:
            The formatted query string

        Example:
            >>> QueryPromptTemplates.format_query(
            ...     "patient_search",
            ...     criteria="first_name='John', age > 30"
            ... )
        """
        template = cls.get_template(template_type)
        return template.format(**kwargs)


def get_system_prompt(prompt_type: str = "healthcare_analyst") -> str:
    """Get a system prompt by type.

    This is a convenience function for accessing system prompts.

    Args:
        prompt_type: Type of prompt (default: healthcare_analyst)

    Returns:
        The system prompt string

    Example:
        >>> prompt = get_system_prompt("research_analyst")
    """
    return SystemPrompts.get_prompt(prompt_type)


def get_query_template(template_type: str) -> str:
    """Get a query prompt template by type.

    This is a convenience function for accessing query templates.

    Args:
        template_type: Type of template

    Returns:
        The prompt template string

    Example:
        >>> template = get_query_template("patient_search")
    """
    return QueryPromptTemplates.get_template(template_type)


def format_query(template_type: str, **kwargs) -> str:
    """Format a query using a template.

    This is a convenience function for formatting queries.

    Args:
        template_type: Type of template to use
        **kwargs: Variables to format into the template

    Returns:
        The formatted query string

    Example:
        >>> query = format_query("patient_search", criteria="age > 30")
    """
    return QueryPromptTemplates.format_query(template_type, **kwargs)
