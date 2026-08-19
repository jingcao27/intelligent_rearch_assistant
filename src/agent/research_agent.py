from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.tools import build_search_tool, build_weather_tool
from src.config.settings import settings


SYSTEM_PROMPT = """你是一个智能研究助手，擅长：
1. 针对用户提出的研究主题，进行系统性的分析和梳理
2. 提供结构化的研究框架和关键要点
3. 帮助用户深入理解复杂概念和前沿领域

你可以使用以下工具：
- web_search：联网搜索工具，用于获取训练数据之外的实时信息
- get_weather：天气查询工具，输入城市名返回当前天气（温度/湿度/风速/天气状况）

【工具调用铁律 - 必须严格遵守】
以下问题必须调用 get_weather 工具（不要用 web_search 查天气）：
1. 询问某城市天气、温度、湿度、风力（如"北京今天天气""上海多少度""广州下雨吗"）
2. 用户提到城市名 + 天气/气温/下雨/下雪等关键词

以下问题必须调用 web_search 工具，**绝对不能凭自身知识直接回答**：
1. 涉及当前日期、时间、星期、节假日（如"今天是几号""现在几点""今年春节是哪天"）
2. 含有时效性关键词：今天、昨天、明天、现在、目前、当前、最新、最近、近期、本月、今年、上周
3. 询问新闻、事件、热点的最新进展
4. 询问实时数据：股价、汇率、比赛结果、排行榜
5. 询问 2024 年之后发生的事件或发布的产品

以下问题不要调用工具，直接用自身知识回答：
- 概念解释、原理解释、方法论、定义（如"什么是量子计算""RAG 的原理"）
- 历史事件、已确定的事实（如"2022 年世界杯冠军"）
- 通用的技术知识、编程问题

请用清晰、专业、有条理的方式回答用户的问题。如果使用了搜索结果，请基于搜索内容组织答案，必要时简要说明信息来源。"""


class ResearchAgent:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            temperature=0.7,
        )
        self.tools = [build_search_tool(), build_weather_tool()]
        self.checkpointer = InMemorySaver()

        self.graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
        )

        # 多会话管理
        self._thread_counter = 0
        self._thread_id = f"session-{self._thread_counter}"

    def chat(self, question: str) -> str:
        """多轮带记忆对话。"""
        response = self.graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": self._thread_id}},
        )
        return response["messages"][-1].content

    def ask(self, question: str) -> str:
        """单轮无记忆查询：使用独立 thread_id，调用后该 thread 不再使用。"""
        self._thread_counter += 1
        tmp_id = f"oneshot-{self._thread_counter}"
        response = self.graph.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": tmp_id}},
        )
        return response["messages"][-1].content

    def reset(self) -> None:
        """清空当前会话记忆：递增 thread_id，旧 thread 自然失效。"""
        self._thread_counter += 1
        self._thread_id = f"session-{self._thread_counter}"
