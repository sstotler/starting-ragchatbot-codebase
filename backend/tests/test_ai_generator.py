"""
Integration tests for AIGenerator functionality.

Tests cover:
- Tool calling detection and execution
- Tool result processing and integration
- Response generation with and without tools
- System prompt and conversation history handling
- Error scenarios and edge cases
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test suite for AIGenerator class."""

    def test_initialization(self, mock_config):
        """Test AIGenerator initialization with proper parameters."""
        ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)

        assert ai_gen.model == mock_config.ANTHROPIC_MODEL
        assert ai_gen.base_params["model"] == mock_config.ANTHROPIC_MODEL
        assert ai_gen.base_params["temperature"] == 0
        assert ai_gen.base_params["max_tokens"] == 800

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic, mock_config):
        """Test response generation without tool use."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response without tools"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        ai_gen.client = mock_client

        result = ai_gen.generate_response("What is RAG?")

        assert result == "Test response without tools"
        mock_client.messages.create.assert_called_once()

        # Check that tools are not included in the call
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" not in call_args

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic, mock_config):
        """Test response generation with conversation history."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with history"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        ai_gen.client = mock_client

        history = "Previous conversation context"
        result = ai_gen.generate_response("Follow-up question", conversation_history=history)

        # Check that history is included in system content
        call_args = mock_client.messages.create.call_args[1]
        assert history in call_args["system"]
        assert result == "Response with history"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic, mock_config, mock_tool_manager):
        """Test response generation with tools available but no tool use."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Direct response without using tools"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        ai_gen.client = mock_client

        tools = mock_tool_manager.get_tool_definitions()
        result = ai_gen.generate_response("General question", tools=tools, tool_manager=mock_tool_manager)

        assert result == "Direct response without using tools"

        # Check that tools are included in the call
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tool_choice"] == {"type": "auto"}

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic, mock_config, mock_tool_manager):
        """Test response generation with actual tool use."""
        # Setup mock for initial response with tool use
        mock_client = Mock()

        # Mock tool use content block
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "RAG systems"}

        # Mock initial response (tool use)
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]
        mock_initial_response.stop_reason = "tool_use"

        # Mock final response (after tool execution)
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Final response after tool use"

        # Setup mock calls
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        ai_gen.client = mock_client

        tools = mock_tool_manager.get_tool_definitions()
        result = ai_gen.generate_response("Search for RAG", tools=tools, tool_manager=mock_tool_manager)

        assert result == "Final response after tool use"

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="RAG systems")

        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_single_tool(self, mock_anthropic, mock_config, mock_tool_manager):
        """Test tool execution handling with single tool call."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        ai_gen.client = mock_client

        # Mock tool content block
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test query"}

        # Mock initial response
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Tool execution result"
        mock_client.messages.create.return_value = mock_final_response

        # Mock base parameters
        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test system"
        }

        result = ai_gen._handle_tool_execution(mock_initial_response, base_params, mock_tool_manager)

        assert result == "Tool execution result"
        mock_tool_manager.execute_tool.assert_called_once_with("search_course_content", query="test query")

    @patch('ai_generator.anthropic.Anthropic')
    def test_handle_tool_execution_multiple_tools(self, mock_anthropic, mock_config, mock_tool_manager):
        """Test tool execution handling with multiple tool calls."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        ai_gen.client = mock_client

        # Mock multiple tool content blocks
        mock_tool_block1 = Mock()
        mock_tool_block1.type = "tool_use"
        mock_tool_block1.name = "search_course_content"
        mock_tool_block1.id = "tool_123"
        mock_tool_block1.input = {"query": "first query"}

        mock_tool_block2 = Mock()
        mock_tool_block2.type = "tool_use"
        mock_tool_block2.name = "get_course_outline"
        mock_tool_block2.id = "tool_456"
        mock_tool_block2.input = {"course_title": "RAG Course"}

        # Mock initial response with multiple tools
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block1, mock_tool_block2]

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Multiple tools result"
        mock_client.messages.create.return_value = mock_final_response

        # Mock base parameters
        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test system"
        }

        result = ai_gen._handle_tool_execution(mock_initial_response, base_params, mock_tool_manager)

        assert result == "Multiple tools result"

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="first query")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="RAG Course")

    @patch('ai_generator.anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic, mock_config):
        """Test handling of API errors."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        ai_gen.client = mock_client

        with pytest.raises(Exception) as exc_info:
            ai_gen.generate_response("Test query")

        assert "API Error" in str(exc_info.value)

    def test_system_prompt_content(self, mock_config):
        """Test that system prompt contains expected content."""
        ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)

        system_prompt = ai_gen.SYSTEM_PROMPT

        # Check for key phrases in system prompt
        assert "course materials" in system_prompt.lower()
        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt
        assert "tool call per query maximum" in system_prompt.lower()

    @patch('ai_generator.anthropic.Anthropic')
    def test_message_construction_with_tools(self, mock_anthropic, mock_config, mock_tool_manager):
        """Test proper message construction when tools are available."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        ai_gen.client = mock_client

        tools = mock_tool_manager.get_tool_definitions()
        ai_gen.generate_response("Test query", tools=tools, tool_manager=mock_tool_manager)

        # Check the actual call parameters
        call_args = mock_client.messages.create.call_args[1]

        assert call_args["model"] == mock_config.ANTHROPIC_MODEL
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert "tools" in call_args
        assert call_args["tool_choice"] == {"type": "auto"}
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "Test query"

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_result_message_format(self, mock_anthropic, mock_config, mock_tool_manager):
        """Test that tool results are properly formatted in messages."""
        mock_client = Mock()

        # Mock tool block
        mock_tool_block = Mock()
        mock_tool_block.type = "tool_use"
        mock_tool_block.name = "search_course_content"
        mock_tool_block.id = "tool_123"
        mock_tool_block.input = {"query": "test"}

        # Mock initial response
        mock_initial_response = Mock()
        mock_initial_response.content = [mock_tool_block]

        # Mock final response
        mock_final_response = Mock()
        mock_final_response.content = [Mock()]
        mock_final_response.content[0].text = "Final result"

        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        mock_anthropic.return_value = mock_client

        ai_gen = AIGenerator(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
        ai_gen.client = mock_client

        # Mock tool execution result
        mock_tool_manager.execute_tool.return_value = "Tool execution result"

        base_params = {
            "messages": [{"role": "user", "content": "original query"}],
            "system": "system prompt"
        }

        ai_gen._handle_tool_execution(mock_initial_response, base_params, mock_tool_manager)

        # Check the final API call for proper message structure
        final_call_args = mock_client.messages.create.call_args_list[1][1]
        messages = final_call_args["messages"]

        # Should have: original user message, assistant tool use, user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

        # Check tool result format
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_123"
        assert tool_results[0]["content"] == "Tool execution result"