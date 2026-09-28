# MCP Server Boilerplate

FastMCP server (add/subtract tools, file resources, a prompt) + a LangGraph agent
that uses the tools through a **local Ollama model** (no OpenAI key needed).

## Setup (Windows, PowerShell)

```powershell
ollama pull qwen3:0.6b                 # ~520 MB, supports tool calling
py -3.13 -m venv .venv-win
.venv-win\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Setup (WSL / Linux)

```bash
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Optional env vars: `OLLAMA_MODEL` (default `qwen3:0.6b`),
`OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`).
