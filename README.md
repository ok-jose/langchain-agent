# LangChain + Ollama 本地 Agent

基于 LangChain 和 Ollama 的本地 AI Agent 项目，支持工具调用和多轮对话。

## 环境要求

- Python 3.10+
- [Ollama](https://ollama.ai) 已安装并运行

## 快速开始

### 1. 安装 Ollama 和模型

```bash
# 安装 Ollama 后，拉取所需模型
ollama pull llama3.1
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，修改 Ollama 地址和模型名称
```

### 4. 运行

```bash
# 交互式对话
python agent.py

# 分步教程示例
python tutorial_examples.py
```

## 项目结构

```
langchain-demo/
├── agent.py              # 主程序 - 完整的 Agent 对话
├── tutorial_examples.py  # 分步教程 - 5 个独立示例
├── requirements.txt      # Python 依赖
├── .env                  # 环境变量
├── .env.example          # 环境变量模板
└── README.md
```

## 可用工具

- `get_weather` - 查询城市天气
- `get_current_time` - 获取当前时间
- `calculate` - 数学计算

## 自定义工具

在 `agent.py` 中使用 `@tool` 装饰器添加新工具：

```python
@tool
def my_tool(param: str) -> str:
    """工具描述（必须写清楚，LLM 依赖它来决定是否调用）"""
    # 实现逻辑
    return result
```

然后将工具添加到 `tools` 列表中即可。
