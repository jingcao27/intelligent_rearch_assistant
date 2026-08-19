# 添加联网功能实施计划

## Context

智能研究助手 Agent 当前只基于 LLM 自身知识回答问题，无法获取时效性信息（如近期新闻、最新数据）。需要为它接入联网搜索能力，让 Agent **自动判断**何时联网（问"最新""近期"时触发，问概念解释时不触发）。

技术选型：Tavily Search API + LangGraph `create_react_agent`（基于原生 function calling，非 prompt ReAct）。

## 关键技术验证结论

- **qwen-plus 兼容性**：DashScope 的 OpenAI 兼容协议原生支持 function calling，`ChatOpenAI.bind_tools()` 可直接用，无需切 SDK
- **多轮记忆实现**：`create_react_agent` + `InMemorySaver` checkpointer + `thread_id` 区分会话；SystemMessage 不要塞进 messages 列表（已在 `prompt=` 参数里注册）
- **reset 实现**：递增 `_thread_counter` 换 `thread_id`，旧 thread 自然失效
- **Tavily 工具**：用官方维护的 `langchain-tavily` 包的 `TavilySearch` 类，不用 `langchain-community` 里已 deprecated 的版本

## 文件改动清单

### 1. [pyproject.toml](file:///e:/java1/Intelligent_research_assistant/pyproject.toml) — 添加依赖

```toml
dependencies = [
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "langgraph>=0.2.0",
    "langgraph-checkpoint>=4.0.0",
    "langgraph-prebuilt>=1.0.0",
    "langchain-tavily>=0.2.0",
    "python-dotenv>=1.0.0",
]
```

### 2. [.env.example](file:///e:/java1/Intelligent_research_assistant/.env.example) — 新增 Tavily Key

末尾追加：
```
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. [src/config/settings.py](file:///e:/java1/Intelligent_research_assistant/src/config/settings.py) — 新增字段

```python
tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
```

### 4. [src/agent/tools.py](file:///e:/java1/Intelligent_research_assistant/src/agent/tools.py) — 新建工具模块

封装 `build_search_tool()` 返回 `TavilySearch` 实例：
- `tavily_api_key=settings.tavily_api_key` 显式传参（便于单测 mock）
- `max_results=5`, `topic="general"`, `include_answer="basic"`

### 5. [src/agent/research_agent.py](file:///e:/java1/Intelligent_research_assistant/src/agent/research_agent.py) — 核心重构

- 用 `create_react_agent(model, tools, prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver())` 创建 graph agent
- `chat(question)` → `graph.invoke({"messages": [HumanMessage(question)]}, config={"configurable": {"thread_id": self._thread_id}})`，返回 `response["messages"][-1].content`
- `ask(question)` → 用临时 `oneshot-{n}` thread_id，单轮无记忆
- `reset()` → 递增 `_thread_counter`，换 `_thread_id`
- `SYSTEM_PROMPT` 增加工具调用判断标准（中文说明何时该搜/不该搜）

### 6. [tests/test_research_agent.py](file:///e:/java1/Intelligent_research_assistant/tests/test_research_agent.py) — 测试改造

- 所有测试需同时 mock `openai_api_key` 和 `tavily_api_key`（否则 TavilySearch 实例化会抛 ValidationError）
- 断言改为 `agent.graph is not None`、`agent.tools` 长度、`_thread_id` 格式
- 新增 `test_reset_changes_thread_id` 和 `test_chat_returns_final_message_content`（mock graph.invoke）

### 7. [src/main.py](file:///e:/java1/Intelligent_research_assistant/src/main.py) — 无需改动

`chat(question) -> str` 接口签名保持兼容，CLI 循环不动。

## 实施顺序

1. 改 pyproject.toml 加依赖 → `.venv\Scripts\python.exe -m pip install -e ".[dev]"`
2. 改 settings.py + .env.example
3. 新建 src/agent/tools.py
4. 重构 src/agent/research_agent.py
5. 改 tests/test_research_agent.py → `pytest tests/ -v` 验证 mock 测试通过
6. 在 `.env` 填真实 `TAVILY_API_KEY`（需用户去 https://app.tavily.com/ 注册）

## 验证方式

### 单元测试
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```
应全部通过（不依赖真实 API 调用）。

### 端到端手动测试
```powershell
.venv\Scripts\python.exe -m src.main
```

输入两个对比问题观察行为：
- **不应触发搜索**：`什么是量子计算？` — Agent 直接用知识回答
- **应触发搜索**：`2026 年 8 月最新 AI 新闻有哪些？` — Agent 调用 Tavily 搜索后基于结果回答

如果两类问题行为符合预期，则联网功能生效。

## 注意事项

- Tavily 免费版每月 1000 次搜索额度，开发期足够
- InMemorySaver 在 CLI 退出后丢失历史（CLI 场景可接受，未来服务化需换 SqliteSaver/PostgresSaver）
- `create_react_agent` 默认 `recursion_limit=25`，若遇 LLM 反复调用搜索循环，可在 invoke 时显式提高
- SYSTEM_PROMPT 已用中文显式说明工具调用判断标准，提升 qwen-plus 工具选择稳定性
