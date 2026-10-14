from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.tools import (
    build_rag_tool,
    build_search_tool,
    build_weather_tool,
)
from src.config.settings import settings


SYSTEM_PROMPT = """你是一个智能研究助手，擅长：
1. 针对用户提出的研究主题，进行系统性的分析和梳理
2. 提供结构化的研究框架和关键要点
3. 帮助用户深入理解复杂概念和前沿领域

你可以使用以下工具：
- web_search：联网搜索工具，用于获取训练数据之外的实时信息
- get_weather：天气查询工具，输入城市名返回当前天气（温度/湿度/风速/天气状况）
- retrieve_annual_report：A 股上市公司年报检索（基于 RAG 向量检索）

【工具调用铁律 - 必须严格遵守】
1. 以下问题必须调用 retrieve_annual_report 工具（不要用 web_search 查年报）：
   - 询问 A 股上市公司年报内容（如"茅台 2024 年报""宁德时代营收""比亚迪净利润"）
   - 询问已覆盖公司的财务数据、业务战略、风险因素、研发投入、股东结构
   - 已覆盖公司：贵州茅台、宁德时代、比亚迪、中国平安、招商银行、工商银行、
     中国石油、中国石化、五粮液、恒瑞医药
   - 调用时传入 company 参数限定公司（如"茅台""宁德时代"），未提及公司时不传

2. 以下问题必须调用 get_weather 工具（不要用 web_search 查天气）：
   - 询问某城市天气、温度、湿度、风力（如"北京今天天气""上海多少度""广州下雨吗"）
   - 用户提到城市名 + 天气/气温/下雨/下雪等关键词

3. 以下问题必须调用 web_search 工具，**绝对不能凭自身知识直接回答**：
   - 涉及当前日期、时间、星期、节假日（如"今天是几号""现在几点""今年春节是哪天"）
   - 含有时效性关键词：今天、昨天、明天、现在、目前、当前、最新、最近、近期、本月、今年、上周
   - 询问新闻、事件、热点的最新进展
   - 询问实时数据：股价、汇率、比赛结果、排行榜
   - 询问 2024 年之后发生的事件或发布的产品

4. 以下问题不要调用工具，直接用自身知识回答：
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
        self.tools = [build_search_tool(), build_weather_tool(), build_rag_tool()]
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

    def chat_stream(self, question: str):
        """多轮带记忆对话的流式版本。

        逐 token 产出最终回复文本，跳过工具调用相关的中间消息。
        yield 的每个元素是一个 str 片段（可能为空字符串之外的任意长度）。
        """
        for msg, metadata in self.graph.stream(
            {"messages": [HumanMessage(content=question)]},
            config={"configurable": {"thread_id": self._thread_id}},
            stream_mode="messages",
        ):
            # 只关心 AIMessage 的文本 chunk；ToolMessage 和带 tool_call_chunks 的 chunk 跳过
            if not isinstance(msg, AIMessage):
                continue
            # 工具调用决策阶段的 chunk 通常 content 为空且带 tool_call_chunks，跳过
            if getattr(msg, "tool_call_chunks", None):
                continue
            content = msg.content if isinstance(msg.content, str) else ""
            if not content:
                continue
            # 多轮工具调用时，最终回复来自最后一个 agent 节点；
            # LangGraph 1.x 中模型调用节点叫 "model"，工具调用决策节点叫 "agent"
            node = metadata.get("langgraph_node", "")
            if node not in ("agent", "model"):
                continue
            yield content

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
