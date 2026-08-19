# 智能研究助手 (Intelligent Research Assistant)

基于 Python + LangChain 构建的智能研究助手 Agent。

## 技术栈

- **Python 3.14**
- **LangChain** — Agent 框架
- **OpenAI API** — LLM 后端

## 快速开始

### 1. 创建虚拟环境并安装依赖

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

### 2. 配置环境变量

```powershell
copy .env.example .env
```

然后编辑 `.env` 文件，填入你的 OpenAI API Key：

```
OPENAI_API_KEY=sk-你的真实key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

### 3. 运行

```powershell
.venv\Scripts\python.exe -m src.main
```

## 项目结构

```
├── src/
│   ├── agent/           # Agent 实现
│   │   └── research_agent.py
│   ├── config/          # 配置管理
│   │   └── settings.py
│   └── main.py          # 入口文件
├── tests/               # 测试
├── pyproject.toml       # 项目配置
└── .env.example         # 环境变量模板
```
