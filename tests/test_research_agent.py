import os
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, AIMessageChunk

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
        assert len(agent.tools) == 3  # web_search + get_weather + retrieve_annual_report
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


def test_chat_stream_yields_token_chunks():
    """验证 chat_stream 返回生成器，且每个元素都是字符串片段。"""
    # 构造模拟流：(msg, metadata) 元组的列表
    fake_stream = [
        # agent 节点的 AIMessage chunk（应该被 yield）
        (AIMessageChunk(content="你好"), {"langgraph_node": "agent"}),
        (AIMessageChunk(content="，世界"), {"langgraph_node": "agent"}),
        # 工具调用决策 chunk（带 tool_call_chunks，应被跳过）
        (AIMessageChunk(content="", tool_call_chunks=[{"name": "web_search"}]),
         {"langgraph_node": "agent"}),
        # 工具节点产出的 chunk（不是 AIMessage，应被跳过）
        (object(), {"langgraph_node": "tools"}),
    ]
    with patch("src.agent.research_agent.settings.openai_api_key", "test-key"), \
         patch("src.agent.research_agent.settings.tavily_api_key", "test-tavily-key"):
        from src.agent.research_agent import ResearchAgent
        agent = ResearchAgent()
        agent.graph = MagicMock()
        agent.graph.stream.return_value = fake_stream

        chunks = list(agent.chat_stream("hi"))
        assert chunks == ["你好", "，世界"]
        # 验证 stream_mode 参数正确传入
        call_kwargs = agent.graph.stream.call_args.kwargs
        assert call_kwargs["stream_mode"] == "messages"
        assert call_kwargs["config"]["configurable"]["thread_id"] == "session-0"
