import os
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from src.config.settings import Settings


def test_settings_defaults():
    with patch.dict(os.environ, {}, clear=True):
        s = Settings(openai_api_key="test-key")
        assert s.openai_model == "gpt-4o-mini"
        assert s.openai_base_url == "https://api.openai.com/v1"
        assert s.tavily_api_key == ""


def test_research_agent_init():
    with patch("src.agent.research_agent.settings.openai_api_key", "test-key"), \
         patch("src.agent.research_agent.settings.tavily_api_key", "test-tavily-key"):
        from src.agent.research_agent import ResearchAgent
        agent = ResearchAgent()
        assert agent.graph is not None
        assert len(agent.tools) == 2  # web_search + get_weather
        assert agent._thread_id == "session-0"


def test_reset_changes_thread_id():
    with patch("src.agent.research_agent.settings.openai_api_key", "test-key"), \
         patch("src.agent.research_agent.settings.tavily_api_key", "test-tavily-key"):
        from src.agent.research_agent import ResearchAgent
        agent = ResearchAgent()
        old = agent._thread_id
        agent.reset()
        assert agent._thread_id != old
        assert agent._thread_id == "session-1"


def test_chat_returns_final_message_content():
    fake_response = {"messages": [AIMessage(content="mocked answer")]}
    with patch("src.agent.research_agent.settings.openai_api_key", "test-key"), \
         patch("src.agent.research_agent.settings.tavily_api_key", "test-tavily-key"):
        from src.agent.research_agent import ResearchAgent
        agent = ResearchAgent()
        agent.graph = MagicMock()
        agent.graph.invoke.return_value = fake_response
        result = agent.chat("hello")
        assert result == "mocked answer"
        invoked_config = agent.graph.invoke.call_args.kwargs["config"]
        assert invoked_config["configurable"]["thread_id"] == "session-0"
