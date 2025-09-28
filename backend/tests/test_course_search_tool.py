"""
Unit tests for CourseSearchTool functionality.

Tests cover:
- Source deduplication when multiple chunks from same course/lesson
- Search result formatting with proper headers
- Course name resolution and filtering
- Lesson number filtering
- Empty results handling
- Error scenarios
"""

import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool class."""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly formatted."""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]

    def test_execute_basic_search(self, mock_vector_store):
        """Test basic search execution without filters."""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("RAG systems")

        assert "RAG Fundamentals" in result
        assert "Introduction to Retrieval-Augmented Generation" in result
        assert len(tool.last_sources) > 0

    def test_execute_with_course_filter(self, mock_vector_store):
        """Test search execution with course name filter."""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("RAG systems", course_name="RAG Fundamentals")

        assert "RAG Fundamentals" in result
        # Should call store.search with course_name parameter
        mock_vector_store.search.assert_called_with(
            query="RAG systems",
            course_name="RAG Fundamentals",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test search execution with lesson number filter."""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("RAG systems", lesson_number=1)

        assert "RAG Fundamentals" in result
        # Should call store.search with lesson_number parameter
        mock_vector_store.search.assert_called_with(
            query="RAG systems",
            course_name=None,
            lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store):
        """Test search execution with both course and lesson filters."""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("RAG systems", course_name="RAG Fundamentals", lesson_number=1)

        # Should call store.search with both parameters
        mock_vector_store.search.assert_called_with(
            query="RAG systems",
            course_name="RAG Fundamentals",
            lesson_number=1
        )

    def test_source_deduplication(self, mock_vector_store):
        """Test that duplicate sources are properly deduplicated."""
        tool = CourseSearchTool(mock_vector_store)

        # Execute search that returns duplicate sources
        result = tool.execute("RAG systems")

        # Check that sources are deduplicated
        unique_sources = tool.last_sources
        source_keys = [(s["text"], s.get("url")) for s in unique_sources]

        # Should have only one unique source despite multiple chunks from same lesson
        assert len(set(source_keys)) == len(unique_sources), "Sources should be deduplicated"

    def test_source_deduplication_comprehensive(self, mock_vector_store):
        """Test comprehensive source deduplication with custom search results."""
        tool = CourseSearchTool(mock_vector_store)

        # Mock search results with intentional duplicates
        duplicate_results = SearchResults(
            documents=[
                "First chunk from lesson 1",
                "Second chunk from lesson 1",
                "Third chunk from lesson 1",
                "Chunk from lesson 2"
            ],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 1},  # Duplicate
                {"course_title": "Test Course", "lesson_number": 1},  # Duplicate
                {"course_title": "Test Course", "lesson_number": 2}   # Different lesson
            ],
            distances=[0.1, 0.2, 0.3, 0.4]
        )

        # Override the mock to return our custom results
        mock_vector_store.search.side_effect = lambda *args, **kwargs: duplicate_results

        result = tool.execute("test query")

        # Should only have 2 unique sources (lesson 1 and lesson 2)
        assert len(tool.last_sources) == 2

        source_texts = [s["text"] for s in tool.last_sources]
        assert "Test Course - Lesson 1" in source_texts
        assert "Test Course - Lesson 2" in source_texts

    def test_format_results_with_lesson_links(self, mock_vector_store):
        """Test that lesson links are properly included in sources."""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("RAG systems")

        # Check that sources include lesson links
        sources_with_links = [s for s in tool.last_sources if s.get("url")]
        assert len(sources_with_links) > 0

        # Check that the URL is properly formatted
        for source in sources_with_links:
            assert source["url"].startswith("http")

    def test_empty_search_results(self, mock_vector_store):
        """Test handling of empty search results."""
        # Mock empty results without error - this should trigger the empty check
        empty_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None  # No error, just empty results
        )
        mock_vector_store.search.side_effect = lambda *args, **kwargs: empty_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("nonexistent query")

        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0

    def test_search_error_handling(self, mock_vector_store):
        """Test handling of search errors."""
        # Mock error results
        error_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        mock_vector_store.search.side_effect = lambda *args, **kwargs: error_results

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        assert "Database connection failed" in result

    def test_formatted_output_structure(self, mock_vector_store):
        """Test that output is properly formatted with headers."""
        tool = CourseSearchTool(mock_vector_store)

        result = tool.execute("RAG systems")

        # Should contain course and lesson headers in square brackets
        assert "[RAG Fundamentals - Lesson 1]" in result

        # Should contain actual content
        assert "Introduction to Retrieval-Augmented Generation" in result

    def test_last_sources_reset_between_searches(self, mock_vector_store):
        """Test that last_sources is properly updated between searches."""
        tool = CourseSearchTool(mock_vector_store)

        # First search - use default mock behavior
        tool.execute("RAG systems")
        first_sources = tool.last_sources.copy()

        # Mock different results for second search
        empty_results = SearchResults.empty("No results")
        mock_vector_store.search.side_effect = lambda *args, **kwargs: empty_results

        # Second search
        tool.execute("nonexistent")
        second_sources = tool.last_sources

        # Sources should be different (empty for second search)
        assert len(second_sources) == 0
        assert first_sources != second_sources


class TestToolManager:
    """Test suite for ToolManager functionality."""

    def test_tool_registration(self, mock_vector_store):
        """Test tool registration in ToolManager."""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)

        manager.register_tool(tool)

        # Check that tool is registered
        assert "search_course_content" in manager.tools

    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions."""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool(self, mock_vector_store):
        """Test tool execution through ToolManager."""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="RAG systems")

        assert "RAG Fundamentals" in result

    def test_execute_nonexistent_tool(self, mock_vector_store):
        """Test execution of nonexistent tool."""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "Tool 'nonexistent_tool' not found" in result

    def test_get_last_sources(self, mock_vector_store):
        """Test getting sources from last tool execution."""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute tool to generate sources
        manager.execute_tool("search_course_content", query="RAG systems")

        sources = manager.get_last_sources()
        assert len(sources) > 0

    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources from all tools."""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)

        # Execute tool to generate sources
        manager.execute_tool("search_course_content", query="RAG systems")

        # Reset sources
        manager.reset_sources()

        # Sources should be empty
        sources = manager.get_last_sources()
        assert len(sources) == 0