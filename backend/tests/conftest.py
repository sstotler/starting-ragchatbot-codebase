"""
Shared pytest fixtures for RAG system tests.
"""

import pytest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock(spec=Config)
    config.ANTHROPIC_API_KEY = "test_api_key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def sample_course():
    """Sample course for testing."""
    lessons = [
        Lesson(lesson_number=1, title="Introduction to RAG", lesson_link="https://example.com/lesson1"),
        Lesson(lesson_number=2, title="Advanced RAG Techniques", lesson_link="https://example.com/lesson2")
    ]
    return Course(
        title="RAG Fundamentals",
        course_link="https://example.com/course",
        instructor="Dr. AI",
        lessons=lessons
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing."""
    return [
        CourseChunk(
            content="Introduction to Retrieval-Augmented Generation systems.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="RAG systems combine retrieval and generation capabilities.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Advanced techniques include prompt compression and query optimization.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=0
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = Mock()

    # Mock search method as a proper Mock
    def mock_search(query, course_name=None, lesson_number=None):
        # Return sample search results
        if "RAG" in query:
            return SearchResults(
                documents=[
                    "Introduction to Retrieval-Augmented Generation systems.",
                    "RAG systems combine retrieval and generation capabilities.",
                    "RAG systems combine retrieval and generation capabilities."  # Duplicate for testing deduplication
                ],
                metadata=[
                    {"course_title": "RAG Fundamentals", "lesson_number": 1},
                    {"course_title": "RAG Fundamentals", "lesson_number": 1},
                    {"course_title": "RAG Fundamentals", "lesson_number": 1}  # Same lesson - should deduplicate
                ],
                distances=[0.1, 0.2, 0.3]
            )
        else:
            return SearchResults.empty("No results found")

    # Create a proper Mock object for search
    mock_store.search = Mock(side_effect=mock_search)

    # Mock get_lesson_link method
    def mock_get_lesson_link(course_title, lesson_number):
        if course_title == "RAG Fundamentals" and lesson_number == 1:
            return "https://example.com/lesson1"
        return None

    mock_store.get_lesson_link = Mock(side_effect=mock_get_lesson_link)

    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing."""
    mock_client = Mock()

    # Mock message creation
    def mock_create(**kwargs):
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response from Claude"
        mock_response.stop_reason = "end_turn"
        return mock_response

    mock_client.messages.create = mock_create
    return mock_client


@pytest.fixture
def mock_tool_manager():
    """Mock tool manager for testing."""
    mock_manager = Mock()

    # Mock tool definitions
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                    "lesson_number": {"type": "integer"}
                },
                "required": ["query"]
            }
        }
    ]

    # Mock tool execution
    mock_manager.execute_tool.return_value = "Search results from tool"

    # Mock sources
    mock_manager.get_last_sources.return_value = [
        {"text": "RAG Fundamentals - Lesson 1", "url": "https://example.com/lesson1"}
    ]

    mock_manager.reset_sources.return_value = None

    return mock_manager


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return SearchResults(
        documents=[
            "RAG systems are powerful for knowledge retrieval.",
            "Prompt compression reduces token usage.",
            "Query optimization improves search accuracy."
        ],
        metadata=[
            {"course_title": "RAG Fundamentals", "lesson_number": 1},
            {"course_title": "Prompt Compression", "lesson_number": 1},
            {"course_title": "RAG Fundamentals", "lesson_number": 2}
        ],
        distances=[0.1, 0.2, 0.3]
    )


@pytest.fixture
def duplicate_search_results():
    """Search results with duplicate sources for testing deduplication."""
    return SearchResults(
        documents=[
            "First chunk about RAG systems.",
            "Second chunk about RAG systems.",
            "Third chunk about RAG systems."
        ],
        metadata=[
            {"course_title": "RAG Fundamentals", "lesson_number": 1},
            {"course_title": "RAG Fundamentals", "lesson_number": 1},  # Same source
            {"course_title": "RAG Fundamentals", "lesson_number": 1}   # Same source
        ],
        distances=[0.1, 0.2, 0.3]
    )