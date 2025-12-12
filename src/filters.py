"""Filter module for HuggingFace model filtering."""

import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.models import HFModel

logger = logging.getLogger(__name__)


class FilterOperator(Enum):
    """Supported filter operators."""

    EQ = "eq"  # Exact match (case-insensitive)
    NE = "ne"  # Not equal (case-insensitive)
    LIKE = "like"  # Contains (case-insensitive)
    IN = "in"  # Value in list field


class FilterCondition(BaseModel):
    """A single filter condition."""

    field: str
    operator: FilterOperator
    value: str


class HFApiFilters(BaseModel):
    """Filters that can be passed to HuggingFace API.

    These filters are propagated to the HF Hub API for server-side filtering,
    improving performance by reducing data transfer.
    """

    search: str | None = None  # Maps to HF API 'search' param (substring match on repo names)
    author: str | None = None  # Maps to HF API 'author' param (filter by org/author)
    tags: list[str] = Field(default_factory=list)  # Maps to HF API 'filter' param


class ModelFilter:
    """Filter for HuggingFace models.

    Supports filtering by any field with various operators:
    - eq: Exact match (case-insensitive)
    - ne: Not equal (case-insensitive)
    - like: Contains substring (case-insensitive)
    - in: Value exists in list field

    Filter syntax: field:operator:value,field2:operator2:value2

    Examples:
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

        Args:
            filter_string: Filter string to parse
        """
        # Handle special filters
        if "cached_only" in filter_string.lower():
            self.cached_only = True
            # Remove cached_only from string and continue parsing
            filter_string = filter_string.lower().replace("cached_only", "").strip(",")
            if not filter_string:
                return

        # Parse field:op:value format
        for part in filter_string.split(","):
            part = part.strip()
            if not part:
                continue

            parts = part.split(":")
            if len(parts) == 3:
                field, op, value = parts
                try:
                    operator = FilterOperator(op.lower())
                    self.conditions.append(
                        FilterCondition(field=field.lower(), operator=operator, value=value)
                    )
                except ValueError:
                    logger.warning(f"Unknown filter operator: {op}")
            else:
                logger.warning(f"Invalid filter format: {part}. Expected field:operator:value")

    def apply(self, models: list["HFModel"]) -> list["HFModel"]:
        """Apply all filter conditions to a list of models.

        Args:
            models: List of HFModel objects to filter

        Returns:
            Filtered list of models matching all conditions (AND logic)
        """
        if not self.conditions:
            return models

        result = models
        for condition in self.conditions:
            result = [m for m in result if self._matches(m, condition)]

        logger.debug(
            f"Filter applied: {len(models)} -> {len(result)} models",
            extra={"conditions": len(self.conditions)},
        )
        return result

    def _matches(self, model: "HFModel", condition: FilterCondition) -> bool:
        """Check if a model matches a single filter condition.

        Args:
            model: HFModel to check
            condition: Filter condition to apply

        Returns:
            True if model matches the condition
        """
        value = self._get_field_value(model, condition.field)

        if value is None:
            return condition.operator == FilterOperator.NE

        match condition.operator:
            case FilterOperator.EQ:
                return self._compare_equal(value, condition.value)
            case FilterOperator.NE:
                return not self._compare_equal(value, condition.value)
            case FilterOperator.LIKE:
                return condition.value.lower() in str(value).lower()
            case FilterOperator.IN:
                if isinstance(value, list):
                    return any(condition.value.lower() == v.lower() for v in value)
                return False

        return False

    def _compare_equal(self, value: Any, filter_value: str) -> bool:
        """Compare value for equality, handling different types.

        Args:
            value: Model field value
            filter_value: Filter value (string)

        Returns:
            True if values are equal
        """
        if isinstance(value, bool):
            return str(value).lower() == filter_value.lower()
        if isinstance(value, int | float):
            try:
                return value == type(value)(filter_value)
            except (ValueError, TypeError):
                return False
        return str(value).lower() == filter_value.lower()

    def _get_field_value(self, model: "HFModel", field: str) -> Any:
        """Get field value from model, checking both id and value attributes.

        Args:
            model: HFModel to get field from
            field: Field name to retrieve

        Returns:
            Field value or None if not found
        """
        # Check top-level model fields first
        if field == "id":
            return model.id

        # Check value (HFModelDetail) fields
        if hasattr(model, "value") and model.value is not None:
            return getattr(model.value, field, None)

        return None

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
            # id:like or name:like → search parameter
            if condition.field in ("id", "name") and condition.operator == FilterOperator.LIKE:
                # Only use first search term (HF API accepts single search string)
                if api_filters.search is None:
                    api_filters.search = condition.value

            # tags:in → filter parameter
            elif condition.field == "tags" and condition.operator == FilterOperator.IN:
                api_filters.tags.append(condition.value)

            # author:eq → author parameter
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
            # These fields are not supported by HF API
            if condition.field in ("visibility", "gated", "cached"):
                local_conditions.append(condition)

            # EQ/NE on id/name need local filtering (HF API only has search/like)
            elif condition.field in ("id", "name") and condition.operator in (
                FilterOperator.EQ,
                FilterOperator.NE,
            ):
                local_conditions.append(condition)

            # NE on tags needs local filtering
            elif condition.field == "tags" and condition.operator != FilterOperator.IN:
                local_conditions.append(condition)

            # Author with non-EQ operator needs local filtering
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
