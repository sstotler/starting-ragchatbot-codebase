"""
End-to-end tests for RAG System functionality.

Tests cover:
- Complete query processing pipeline
- Session management and conversation history
- Tool integration and coordination
- Source attribution and collection
- Error handling in the complete system
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


class TestRAGSystem:
    """Test suite for RAGSystem end-to-end functionality."""

    @pytest.fixture
    def mock_rag_system(self, mock_config):
        """Create a RAGSystem with mocked dependencies."""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_generator, \
             patch('rag_system.SessionManager') as mock_session_manager, \
             patch('rag_system.ToolManager') as mock_tool_manager, \
             patch('rag_system.CourseSearchTool') as mock_search_tool, \
             patch('rag_system.CourseOutlineTool') as mock_outline_tool:

            # Setup mock instances
            mock_vector_store_instance = Mock()
            mock_vector_store.return_value = mock_vector_store_instance

            mock_ai_generator_instance = Mock()
            mock_ai_generator.return_value = mock_ai_generator_instance

            mock_session_manager_instance = Mock()
            mock_session_manager.return_value = mock_session_manager_instance

            mock_tool_manager_instance = Mock()
            mock_tool_manager.return_value = mock_tool_manager_instance

            mock_search_tool_instance = Mock()
            mock_search_tool.return_value = mock_search_tool_instance

            mock_outline_tool_instance = Mock()
            mock_outline_tool.return_value = mock_outline_tool_instance

            # Create RAG system
            rag_system = RAGSystem(mock_config)

            # Store mocks for access in tests
            rag_system._mock_vector_store = mock_vector_store_instance
            rag_system._mock_ai_generator = mock_ai_generator_instance
            rag_system._mock_session_manager = mock_session_manager_instance
            rag_system._mock_tool_manager = mock_tool_manager_instance

            return rag_system

    def test_rag_system_initialization(self, mock_rag_system, mock_config):
        """Test that RAGSystem initializes with all required components."""
        assert mock_rag_system.config == mock_config
        assert hasattr(mock_rag_system, 'document_processor')
        assert hasattr(mock_rag_system, 'vector_store')
        assert hasattr(mock_rag_system, 'ai_generator')
        assert hasattr(mock_rag_system, 'session_manager')
        assert hasattr(mock_rag_system, 'tool_manager')
        assert hasattr(mock_rag_system, 'search_tool')
        assert hasattr(mock_rag_system, 'outline_tool')

    def test_query_without_session(self, mock_rag_system):
        """Test query processing without existing session."""
        # Setup mocks
        mock_rag_system._mock_ai_generator.generate_response.return_value = "Test response"
        mock_rag_system._mock_tool_manager.get_last_sources.return_value = [
            {"text": "Source 1", "url": "http://example.com/1"}
        ]

        # Execute query
        response, sources = mock_rag_system.query("What is RAG?")

        # Verify response
        assert response == "Test response"
        assert len(sources) == 1
        assert sources[0]["text"] == "Source 1"

        # Verify AI generator was called correctly
        mock_rag_system._mock_ai_generator.generate_response.assert_called_once()
        call_args = mock_rag_system._mock_ai_generator.generate_response.call_args

        # Check that query is properly formatted (use kwargs instead of positional args)
        call_kwargs = call_args[1] if len(call_args) > 1 else {}
        call_positional = call_args[0] if len(call_args) > 0 else []

        # The query should be in the first positional argument
        if len(call_positional) > 0:
            assert "What is RAG?" in call_positional[0]

        # Check that tools and tool_manager are passed
        assert 'tools' in call_kwargs
        assert 'tool_manager' in call_kwargs

    def test_query_with_session(self, mock_rag_system):
        """Test query processing with existing session."""
        session_id = "test_session_123"
        conversation_history = "Previous conversation context"

        # Setup mocks
        mock_rag_system._mock_session_manager.get_conversation_history.return_value = conversation_history
        mock_rag_system._mock_ai_generator.generate_response.return_value = "Follow-up response"
        mock_rag_system._mock_tool_manager.get_last_sources.return_value = []

        # Execute query
        response, sources = mock_rag_system.query("Follow-up question", session_id)

        # Verify response
        assert response == "Follow-up response"

        # Verify session manager interactions
        mock_rag_system._mock_session_manager.get_conversation_history.assert_called_once_with(session_id)
        mock_rag_system._mock_session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow-up question", "Follow-up response"
        )

        # Verify AI generator received conversation history
        call_args = mock_rag_system._mock_ai_generator.generate_response.call_args
        assert 'conversation_history' in call_args[1]
        assert call_args[1]['conversation_history'] == conversation_history

    def test_query_with_sources(self, mock_rag_system):
        """Test that sources are properly collected and reset."""
        # Setup mock sources
        test_sources = [
            {"text": "RAG Course - Lesson 1", "url": "http://example.com/lesson1"},
            {"text": "Prompt Optimization - Lesson 2", "url": "http://example.com/lesson2"}
        ]

        mock_rag_system._mock_ai_generator.generate_response.return_value = "Response with sources"
        mock_rag_system._mock_tool_manager.get_last_sources.return_value = test_sources

        # Execute query
        response, sources = mock_rag_system.query("Search query")

        # Verify sources are returned
        assert len(sources) == 2
        assert sources == test_sources

        # Verify sources are reset after retrieval
        mock_rag_system._mock_tool_manager.reset_sources.assert_called_once()

    def test_add_course_document_success(self, mock_rag_system, sample_course, sample_course_chunks):
        """Test successful addition of a course document."""
        # Setup mocks
        mock_rag_system.document_processor.process_course_document.return_value = (sample_course, sample_course_chunks)

        # Execute
        course, chunk_count = mock_rag_system.add_course_document("/path/to/course.pdf")

        # Verify
        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)

        # Verify vector store interactions
        mock_rag_system.vector_store.add_course_metadata.assert_called_once_with(sample_course)
        mock_rag_system.vector_store.add_course_content.assert_called_once_with(sample_course_chunks)

    def test_add_course_document_error(self, mock_rag_system):
        """Test error handling when adding course document fails."""
        # Setup mock to raise exception
        mock_rag_system.document_processor.process_course_document.side_effect = Exception("Processing failed")

        # Execute
        course, chunk_count = mock_rag_system.add_course_document("/invalid/path.pdf")

        # Verify error handling
        assert course is None
        assert chunk_count == 0

    def test_add_course_folder_success(self, mock_rag_system, sample_course, sample_course_chunks):
        """Test successful addition of course folder."""
        # Setup mocks
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['course1.pdf', 'course2.txt']), \
             patch('os.path.isfile', return_value=True):

            mock_rag_system.vector_store.get_existing_course_titles.return_value = []
            mock_rag_system.document_processor.process_course_document.return_value = (sample_course, sample_course_chunks)

            # Execute
            total_courses, total_chunks = mock_rag_system.add_course_folder("/path/to/courses")

            # Verify
            assert total_courses == 2  # Two files processed
            assert total_chunks == len(sample_course_chunks) * 2

    def test_add_course_folder_nonexistent(self, mock_rag_system):
        """Test handling of nonexistent course folder."""
        with patch('os.path.exists', return_value=False):
            total_courses, total_chunks = mock_rag_system.add_course_folder("/nonexistent/path")

            assert total_courses == 0
            assert total_chunks == 0

    def test_add_course_folder_with_existing_courses(self, mock_rag_system, sample_course, sample_course_chunks):
        """Test folder addition with some courses already existing."""
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=['course1.pdf', 'course2.txt']), \
             patch('os.path.isfile', return_value=True):

            # Mock existing course titles
            mock_rag_system.vector_store.get_existing_course_titles.return_value = [sample_course.title]
            mock_rag_system.document_processor.process_course_document.return_value = (sample_course, sample_course_chunks)

            # Execute
            total_courses, total_chunks = mock_rag_system.add_course_folder("/path/to/courses")

            # Should skip existing courses
            assert total_courses == 0  # All courses already exist
            assert total_chunks == 0

    def test_get_course_analytics(self, mock_rag_system):
        """Test course analytics retrieval."""
        # Setup mocks
        mock_rag_system.vector_store.get_course_count.return_value = 5
        mock_rag_system.vector_store.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3", "Course 4", "Course 5"
        ]

        # Execute
        analytics = mock_rag_system.get_course_analytics()

        # Verify
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]

    def test_query_error_handling(self, mock_rag_system):
        """Test error handling during query processing."""
        # Setup mock to raise exception
        mock_rag_system._mock_ai_generator.generate_response.side_effect = Exception("AI Generation failed")

        # Execute and expect exception to propagate
        with pytest.raises(Exception) as exc_info:
            mock_rag_system.query("Test query")

        assert "AI Generation failed" in str(exc_info.value)

    def test_session_management_integration(self, mock_rag_system):
        """Test integration between RAG system and session management."""
        session_id = "integration_test_session"

        # Setup mocks
        mock_rag_system._mock_session_manager.get_conversation_history.return_value = None  # New session
        mock_rag_system._mock_ai_generator.generate_response.return_value = "First response"
        mock_rag_system._mock_tool_manager.get_last_sources.return_value = []

        # First query
        response1, _ = mock_rag_system.query("First question", session_id)

        # Verify session exchange was added
        mock_rag_system._mock_session_manager.add_exchange.assert_called_with(
            session_id, "First question", "First response"
        )

        # Setup for second query with history
        mock_rag_system._mock_session_manager.get_conversation_history.return_value = "Previous context"
        mock_rag_system._mock_ai_generator.generate_response.return_value = "Second response"

        # Second query
        response2, _ = mock_rag_system.query("Second question", session_id)

        # Verify history was retrieved and new exchange added
        assert mock_rag_system._mock_session_manager.get_conversation_history.call_count == 2
        assert mock_rag_system._mock_session_manager.add_exchange.call_count == 2

    def test_tool_integration(self, mock_rag_system):
        """Test proper integration with tool system."""
        # Execute query
        mock_rag_system.query("Test query with tools")

        # Verify tool manager interactions
        mock_rag_system._mock_ai_generator.generate_response.assert_called_once()
        call_args = mock_rag_system._mock_ai_generator.generate_response.call_args[1]

        # Check that tools and tool_manager are passed
        assert 'tools' in call_args
        assert 'tool_manager' in call_args
        assert call_args['tool_manager'] == mock_rag_system.tool_manager

        # Verify sources are collected and reset
        mock_rag_system._mock_tool_manager.get_last_sources.assert_called_once()
        mock_rag_system._mock_tool_manager.reset_sources.assert_called_once()

    def test_prompt_formatting(self, mock_rag_system):
        """Test that user queries are properly formatted for AI."""
        user_query = "What is RAG?"

        mock_rag_system.query(user_query)

        # Verify AI generator received properly formatted prompt
        call_args = mock_rag_system._mock_ai_generator.generate_response.call_args
        call_positional = call_args[0] if len(call_args) > 0 else []

        # The formatted prompt should be in the first positional argument
        if len(call_positional) > 0:
            formatted_prompt = call_positional[0]
            assert user_query in formatted_prompt
            assert "Answer this question about course materials:" in formatted_prompt

    def test_concurrent_session_isolation(self, mock_rag_system):
        """Test that multiple sessions don't interfere with each other."""
        session1 = "session_1"
        session2 = "session_2"

        # Setup different histories for different sessions
        def mock_get_history(session_id):
            if session_id == session1:
                return "History for session 1"
            elif session_id == session2:
                return "History for session 2"
            return None

        mock_rag_system._mock_session_manager.get_conversation_history.side_effect = mock_get_history
        mock_rag_system._mock_ai_generator.generate_response.return_value = "Response"
        mock_rag_system._mock_tool_manager.get_last_sources.return_value = []

        # Execute queries for both sessions
        mock_rag_system.query("Query 1", session1)
        mock_rag_system.query("Query 2", session2)

        # Verify both sessions were handled correctly
        assert mock_rag_system._mock_session_manager.get_conversation_history.call_count == 2
        assert mock_rag_system._mock_session_manager.add_exchange.call_count == 2

        # Verify correct session IDs were used
        call_args_list = mock_rag_system._mock_session_manager.add_exchange.call_args_list
        assert call_args_list[0][0][0] == session1  # First call with session1
        assert call_args_list[1][0][0] == session2  # Second call with session2