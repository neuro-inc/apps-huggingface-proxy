"""Filter module for HuggingFace model filtering."""

import logging
from typing import TYPE_CHECKING, Any

from apolo_app_types.dynamic_outputs import (
    BaseModelFilter,
    FilterCondition,
    FilterOperator,
    parse_filter_string,
)
from pydantic import BaseModel, Field

__all__ = [
    "FilterCondition",
    "FilterOperator",
    "HFApiFilters",
    "ModelFilter",
]

if TYPE_CHECKING:
    from src.models import HFModel

logger = logging.getLogger(__name__)


class HFApiFilters(BaseModel):
    """Filters that can be passed to HuggingFace API."""

    search: str | None = None
    author: str | None = None
    tags: list[str] = Field(default_factory=list)


class ModelFilter(BaseModelFilter):
    """Filter for HuggingFace models.

    Extends BaseModelFilter with HF-specific features:
    - cached_only: Special filter for cached models only
    - API filter propagation: Extract filters that can be sent to HF Hub API
    - Local filtering: Apply filters not supported by HF API
    - Shorthand syntax: Simple string without colons = name:like:value

    Filter syntax: field:operator:value,field2:operator2:value2

    Examples:
        - llama → name:like:llama (shorthand)
        - meta-llama → name:like:meta-llama (shorthand)
        - visibility:eq:public
        - name:like:llama
        - gated:eq:true,cached:eq:true
        - tags:in:text-generation
        - cached_only (special filter for cached models only)
    """

    def __init__(self, filter_string: str | None) -> None:
        """Initialize filter from filter string.

        Args:
            filter_string: Filter string in format field:op:value,field:op:value
        """
        self.conditions: list[FilterCondition] = []
        self.cached_only: bool = False
        self._raw_filter = filter_string

        if filter_string:
            self._parse(filter_string)

    def _parse(self, filter_string: str) -> None:
        """Parse filter string into conditions.

        Supports shorthand syntax:
        - Simple string without colons (e.g., "llama") → name:like:llama

        Args:
            filter_string: Filter string to parse
        """
        if "cached_only" in filter_string.lower():
            self.cached_only = True
            filter_string = filter_string.lower().replace("cached_only", "").strip(",")
            if not filter_string:
                return

        # Simple string without colons = name prefix search
        if ":" not in filter_string:
            filter_string = f"name:like:{filter_string}"

        self.conditions = parse_filter_string(filter_string)

    def _get_field_value(self, model: "HFModel", field: str) -> Any:
        """Get field value from model, checking both id and value attributes.

        Args:
            model: HFModel to get field from
            field: Field name to retrieve

        Returns:
            Field value or None if not found
        """
        if field == "id":
            return model.id

        if hasattr(model, "value") and model.value is not None:
            return getattr(model.value, field, None)

        return None

    def _matches_in_operator(self, value: Any, filter_value: str) -> bool:
        """Handle IN operator for list fields.

        Args:
            value: Field value (expected to be a list)
            filter_value: Value to search for in the list

        Returns:
            True if filter_value is found in value
        """
        if isinstance(value, list):
            return any(filter_value.lower() == v.lower() for v in value)
        return False

    def has_conditions(self) -> bool:
        """Check if filter has any conditions to apply.

        Returns:
            True if there are conditions or cached_only is set
        """
        return bool(self.conditions) or self.cached_only

    def get_api_filters(self) -> HFApiFilters:
        """Extract filters that can be propagated to HuggingFace API.

        Analyzes conditions and extracts those that map to HF API parameters:
        - id:like:* or name:like:* → search
        - tags:in:* → tags list
        - author:eq:* → author

        Returns:
            HFApiFilters with extracted values
        """
        api_filters = HFApiFilters()

        for condition in self.conditions:
            if condition.field in ("id", "name") and condition.operator == FilterOperator.LIKE:
                if api_filters.search is None:
                    api_filters.search = condition.value
            elif condition.field == "tags" and condition.operator == FilterOperator.IN:
                api_filters.tags.append(condition.value)
            elif condition.field == "author" and condition.operator == FilterOperator.EQ:
                api_filters.author = condition.value

        return api_filters

    def get_local_conditions(self) -> list[FilterCondition]:
        """Get conditions that must be applied locally (not supported by HF API).

        Returns conditions for:
        - visibility (not in HF API)
        - gated (not in HF API)
        - cached (local cache check)
        - Any EQ/NE operators on id/name (HF API only supports search/like)

        Returns:
            List of FilterCondition objects for local filtering
        """
        local_conditions = []

        for condition in self.conditions:
            if condition.field in ("visibility", "gated", "cached"):
                local_conditions.append(condition)
            elif condition.field in ("id", "name") and condition.operator in (
                FilterOperator.EQ,
                FilterOperator.NE,
            ):
                local_conditions.append(condition)
            elif condition.field == "tags" and condition.operator != FilterOperator.IN:
                local_conditions.append(condition)
            elif condition.field == "author" and condition.operator != FilterOperator.EQ:
                local_conditions.append(condition)

        return local_conditions

    def apply_local(
        self, models: list["HFModel"], conditions: list[FilterCondition] | None = None
    ) -> list["HFModel"]:
        """Apply only local filter conditions to a list of models.

        Args:
            models: List of HFModel objects to filter
            conditions: Optional list of conditions to apply. If None, uses get_local_conditions()

        Returns:
            Filtered list of models matching all conditions (AND logic)
        """
        if conditions is None:
            conditions = self.get_local_conditions()

        if not conditions:
            return models

        result = models
        for condition in conditions:
            result = [m for m in result if self._matches(m, condition)]

        logger.debug(
            f"Local filter applied: {len(models)} -> {len(result)} models",
            extra={"conditions": len(conditions)},
        )
        return result

    def __repr__(self) -> str:
        """String representation of filter."""
        return f"ModelFilter(conditions={len(self.conditions)}, cached_only={self.cached_only})"
