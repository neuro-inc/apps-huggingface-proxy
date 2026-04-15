"""Tests for the ModelFilter class."""

from src.filters import FilterCondition, FilterOperator, ModelFilter
from src.models import HFModel, HFModelDetail


def create_test_model(
    id: str = "test/model",
    name: str = "model",
    visibility: str = "public",
    gated: bool = False,
    tags: list[str] | None = None,
    cached: bool = False,
    last_modified: str | None = None,
) -> HFModel:
    """Create a test HFModel for testing."""
    return HFModel(
        id=id,
        value=HFModelDetail(
            id=id,
            name=name,
            visibility=visibility,
            gated=gated,
            tags=tags or [],
            cached=cached,
            last_modified=last_modified,
        ),
    )


class TestFilterParsing:
    """Tests for filter string parsing."""

    def test_empty_filter(self):
        """Test empty filter string."""
        model_filter = ModelFilter(None)
        assert model_filter.conditions == []
        assert model_filter.cached_only is False

    def test_empty_string_filter(self):
        """Test empty string filter."""
        model_filter = ModelFilter("")
        assert model_filter.conditions == []
        assert model_filter.cached_only is False

    def test_cached_only_filter(self):
        """Test cached_only special filter."""
        model_filter = ModelFilter("cached_only")
        assert model_filter.cached_only is True
        assert model_filter.conditions == []

    def test_cached_only_case_insensitive(self):
        """Test cached_only is case insensitive."""
        model_filter = ModelFilter("CACHED_ONLY")
        assert model_filter.cached_only is True

    def test_single_condition(self):
        """Test parsing single condition."""
        model_filter = ModelFilter("visibility:eq:public")
        assert len(model_filter.conditions) == 1
        assert model_filter.conditions[0].field == "visibility"
        assert model_filter.conditions[0].operator == FilterOperator.EQ
        assert model_filter.conditions[0].value == "public"

    def test_multiple_conditions(self):
        """Test parsing multiple conditions."""
        model_filter = ModelFilter("visibility:eq:public,gated:eq:true")
        assert len(model_filter.conditions) == 2
        assert model_filter.conditions[0].field == "visibility"
        assert model_filter.conditions[1].field == "gated"

    def test_like_operator(self):
        """Test LIKE operator parsing."""
        model_filter = ModelFilter("name:like:llama")
        assert len(model_filter.conditions) == 1
        assert model_filter.conditions[0].operator == FilterOperator.LIKE

    def test_ne_operator(self):
        """Test NE (not equal) operator parsing."""
        model_filter = ModelFilter("gated:ne:true")
        assert len(model_filter.conditions) == 1
        assert model_filter.conditions[0].operator == FilterOperator.NE

    def test_in_operator(self):
        """Test IN operator parsing."""
        model_filter = ModelFilter("tags:in:text-generation")
        assert len(model_filter.conditions) == 1
        assert model_filter.conditions[0].operator == FilterOperator.IN

    def test_simple_string_converts_to_name_like(self):
        """Test simple string without colons converts to name:like filter."""
        model_filter = ModelFilter("llama")
        assert len(model_filter.conditions) == 1
        assert model_filter.conditions[0].field == "name"
        assert model_filter.conditions[0].operator == FilterOperator.LIKE
        assert model_filter.conditions[0].value == "llama"

    def test_invalid_operator_ignored(self):
        """Test invalid operator is ignored."""
        model_filter = ModelFilter("field:invalid:value")
        assert len(model_filter.conditions) == 0

    def test_mixed_valid_invalid(self):
        """Test mix of valid and invalid filters."""
        model_filter = ModelFilter("visibility:eq:public,invalid,gated:eq:true")
        assert len(model_filter.conditions) == 2


class TestFilterApplication:
    """Tests for applying filters to models."""

    def test_no_filter_returns_all(self):
        """Test that no filter returns all models."""
        models = [
            create_test_model(id="model1"),
            create_test_model(id="model2"),
        ]
        model_filter = ModelFilter(None)
        result = model_filter.apply(models)
        assert len(result) == 2

    def test_eq_filter_visibility(self):
        """Test EQ filter on visibility."""
        models = [
            create_test_model(id="model1", visibility="public"),
            create_test_model(id="model2", visibility="private"),
            create_test_model(id="model3", visibility="public"),
        ]
        model_filter = ModelFilter("visibility:eq:public")
        result = model_filter.apply(models)
        assert len(result) == 2
        assert all(m.value.visibility == "public" for m in result)

    def test_eq_filter_case_insensitive(self):
        """Test EQ filter is case insensitive."""
        models = [
            create_test_model(id="model1", visibility="Public"),
            create_test_model(id="model2", visibility="PUBLIC"),
        ]
        model_filter = ModelFilter("visibility:eq:public")
        result = model_filter.apply(models)
        assert len(result) == 2

    def test_eq_filter_boolean(self):
        """Test EQ filter on boolean field."""
        models = [
            create_test_model(id="model1", gated=True),
            create_test_model(id="model2", gated=False),
            create_test_model(id="model3", gated=True),
        ]
        model_filter = ModelFilter("gated:eq:true")
        result = model_filter.apply(models)
        assert len(result) == 2
        assert all(m.value.gated is True for m in result)

    def test_ne_filter(self):
        """Test NE (not equal) filter."""
        models = [
            create_test_model(id="model1", visibility="public"),
            create_test_model(id="model2", visibility="private"),
        ]
        model_filter = ModelFilter("visibility:ne:private")
        result = model_filter.apply(models)
        assert len(result) == 1
        assert result[0].value.visibility == "public"

    def test_like_filter(self):
        """Test LIKE (contains) filter."""
        models = [
            create_test_model(id="meta-llama/Llama-2-7b", name="Llama-2-7b"),
            create_test_model(id="mistral/Mistral-7B", name="Mistral-7B"),
            create_test_model(id="meta-llama/Llama-3", name="Llama-3"),
        ]
        model_filter = ModelFilter("name:like:llama")
        result = model_filter.apply(models)
        assert len(result) == 2

    def test_like_filter_on_id(self):
        """Test LIKE filter on id field."""
        models = [
            create_test_model(id="meta-llama/Llama-2-7b"),
            create_test_model(id="mistral/Mistral-7B"),
        ]
        model_filter = ModelFilter("id:like:meta-llama")
        result = model_filter.apply(models)
        assert len(result) == 1

    def test_in_filter_tags(self):
        """Test IN filter on tags list."""
        models = [
            create_test_model(id="model1", tags=["text-generation", "llm"]),
            create_test_model(id="model2", tags=["image-classification"]),
            create_test_model(id="model3", tags=["text-generation"]),
        ]
        model_filter = ModelFilter("tags:in:text-generation")
        result = model_filter.apply(models)
        assert len(result) == 2

    def test_in_filter_case_insensitive(self):
        """Test IN filter is case insensitive."""
        models = [
            create_test_model(id="model1", tags=["Text-Generation"]),
        ]
        model_filter = ModelFilter("tags:in:text-generation")
        result = model_filter.apply(models)
        assert len(result) == 1

    def test_multiple_conditions_and_logic(self):
        """Test multiple conditions use AND logic."""
        models = [
            create_test_model(id="model1", visibility="public", gated=True),
            create_test_model(id="model2", visibility="public", gated=False),
            create_test_model(id="model3", visibility="private", gated=True),
        ]
        model_filter = ModelFilter("visibility:eq:public,gated:eq:true")
        result = model_filter.apply(models)
        assert len(result) == 1
        assert result[0].id == "model1"

    def test_filter_cached(self):
        """Test filter on cached field."""
        models = [
            create_test_model(id="model1", cached=True),
            create_test_model(id="model2", cached=False),
        ]
        model_filter = ModelFilter("cached:eq:true")
        result = model_filter.apply(models)
        assert len(result) == 1
        assert result[0].value.cached is True

    def test_filter_nonexistent_field(self):
        """Test filter on non-existent field returns empty."""
        models = [
            create_test_model(id="model1"),
        ]
        model_filter = ModelFilter("nonexistent:eq:value")
        result = model_filter.apply(models)
        assert len(result) == 0


class TestFilterHelpers:
    """Tests for filter helper methods."""

    def test_has_conditions_empty(self):
        """Test has_conditions returns False for empty filter."""
        model_filter = ModelFilter(None)
        assert model_filter.has_conditions() is False

    def test_has_conditions_with_conditions(self):
        """Test has_conditions returns True with conditions."""
        model_filter = ModelFilter("visibility:eq:public")
        assert model_filter.has_conditions() is True

    def test_has_conditions_cached_only(self):
        """Test has_conditions returns True for cached_only."""
        model_filter = ModelFilter("cached_only")
        assert model_filter.has_conditions() is True

    def test_repr(self):
        """Test string representation."""
        model_filter = ModelFilter("visibility:eq:public,gated:eq:true")
        repr_str = repr(model_filter)
        assert "conditions=2" in repr_str
        assert "cached_only=False" in repr_str


class TestApiFilterExtraction:
    """Tests for API filter extraction methods."""

    def test_get_api_filters_empty(self):
        """Test get_api_filters returns empty filters for no conditions."""
        model_filter = ModelFilter(None)
        api_filters = model_filter.get_api_filters()
        assert api_filters.search is None
        assert api_filters.author is None
        assert api_filters.tags == []

    def test_get_api_filters_name_like(self):
        """Test name:like maps to search parameter."""
        model_filter = ModelFilter("name:like:llama")
        api_filters = model_filter.get_api_filters()
        assert api_filters.search == "llama"

    def test_get_api_filters_id_like(self):
        """Test id:like maps to search parameter."""
        model_filter = ModelFilter("id:like:meta-llama")
        api_filters = model_filter.get_api_filters()
        assert api_filters.search == "meta-llama"

    def test_get_api_filters_tags_in(self):
        """Test tags:in maps to tags list."""
        model_filter = ModelFilter("tags:in:text-generation")
        api_filters = model_filter.get_api_filters()
        assert api_filters.tags == ["text-generation"]

    def test_get_api_filters_multiple_tags(self):
        """Test multiple tags:in conditions."""
        model_filter = ModelFilter("tags:in:text-generation,tags:in:pytorch")
        api_filters = model_filter.get_api_filters()
        assert "text-generation" in api_filters.tags
        assert "pytorch" in api_filters.tags

    def test_get_api_filters_author_eq(self):
        """Test author:eq maps to author parameter."""
        model_filter = ModelFilter("author:eq:meta-llama")
        api_filters = model_filter.get_api_filters()
        assert api_filters.author == "meta-llama"

    def test_get_api_filters_combined(self):
        """Test combined API filters."""
        model_filter = ModelFilter("name:like:llama,author:eq:meta-llama,tags:in:text-generation")
        api_filters = model_filter.get_api_filters()
        assert api_filters.search == "llama"
        assert api_filters.author == "meta-llama"
        assert api_filters.tags == ["text-generation"]

    def test_get_api_filters_only_first_search(self):
        """Test only first search term is used."""
        model_filter = ModelFilter("name:like:llama,id:like:mistral")
        api_filters = model_filter.get_api_filters()
        assert api_filters.search == "llama"  # First one wins

    def test_get_local_conditions_visibility(self):
        """Test visibility conditions are local only."""
        model_filter = ModelFilter("visibility:eq:public")
        local_conditions = model_filter.get_local_conditions()
        assert len(local_conditions) == 1
        assert local_conditions[0].field == "visibility"

    def test_get_local_conditions_gated(self):
        """Test gated conditions are local only."""
        model_filter = ModelFilter("gated:eq:true")
        local_conditions = model_filter.get_local_conditions()
        assert len(local_conditions) == 1
        assert local_conditions[0].field == "gated"

    def test_get_local_conditions_cached(self):
        """Test cached conditions are local only."""
        model_filter = ModelFilter("cached:eq:true")
        local_conditions = model_filter.get_local_conditions()
        assert len(local_conditions) == 1
        assert local_conditions[0].field == "cached"

    def test_get_local_conditions_name_eq(self):
        """Test name:eq is local (HF API only supports like/search)."""
        model_filter = ModelFilter("name:eq:Llama-3")
        local_conditions = model_filter.get_local_conditions()
        assert len(local_conditions) == 1
        assert local_conditions[0].field == "name"
        assert local_conditions[0].operator == FilterOperator.EQ

    def test_get_local_conditions_mixed(self):
        """Test mixed API and local conditions."""
        model_filter = ModelFilter("name:like:llama,visibility:eq:public,tags:in:text-generation")
        local_conditions = model_filter.get_local_conditions()
        # Only visibility should be local
        assert len(local_conditions) == 1
        assert local_conditions[0].field == "visibility"

    def test_apply_local_no_conditions(self):
        """Test apply_local with no conditions returns all models."""
        models = [
            create_test_model(id="model1"),
            create_test_model(id="model2"),
        ]
        model_filter = ModelFilter("name:like:llama")  # API filter only
        result = model_filter.apply_local(models)
        assert len(result) == 2  # No local filtering

    def test_apply_local_with_conditions(self):
        """Test apply_local filters by local conditions."""
        models = [
            create_test_model(id="model1", visibility="public"),
            create_test_model(id="model2", visibility="private"),
        ]
        model_filter = ModelFilter("visibility:eq:public")
        result = model_filter.apply_local(models)
        assert len(result) == 1
        assert result[0].id == "model1"

    def test_apply_local_explicit_conditions(self):
        """Test apply_local with explicit conditions list."""
        models = [
            create_test_model(id="model1", gated=True),
            create_test_model(id="model2", gated=False),
        ]
        conditions = [FilterCondition(field="gated", operator=FilterOperator.EQ, value="true")]
        model_filter = ModelFilter(None)
        result = model_filter.apply_local(models, conditions)
        assert len(result) == 1
        assert result[0].id == "model1"
