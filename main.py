"""
MCP Server Boilerplate
----------------------
1. Build a FastMCP server with tools, resources and a prompt.
2. Test it with an in-memory client.
3. Serve it over HTTP and test it with an HTTP client.
4. Let a LangGraph ReAct agent (backed by a local Ollama model) use the MCP tools.

Run:  python main.py
"""
import asyncio
import logging
import os
import socket
import urllib.request

from fastmcp import FastMCP, Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

PORT = 8000
MCP_URL = f"http://127.0.0.1:{PORT}/mcp"

# Small local model with tool-calling support (pull with: ollama pull qwen3:0.6b)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
# Set explicitly: a system-wide OLLAMA_HOST=0.0.0.0 is fine for the server
# but the Python client can't connect to 0.0.0.0 on Windows.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

DOCS = [
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/aNE__JjH4DLNEibuNpfDlg/examples.txt",
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/tfoeGPInNoajVS0DSohdVg/README.txt",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_dir():
    if os.path.exists("path"):
        print("✓ Path directory already exists")
    else:
        print("✗ Path directory doesn't exist - creating it...")
        os.makedirs("path")
        print("✓ Path directory created")


def download_docs():
    """Replacement for the notebook's `!wget -P path/ ...` cells."""
    for url in DOCS:
        dest = os.path.join("path", url.rsplit("/", 1)[-1])
        if os.path.exists(dest):
            continue
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"✓ Downloaded {dest}")
        except Exception as e:
            print(f"✗ Could not download {url}: {e}")


def test_port(port=PORT):
    """Return True if the port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return False
        except socket.error:
            return True


def print_stream_info(read, write, _sid, verbose=False):
    """Print information about the read/write streams and session ID."""
    if verbose:
        print("READ (receives FROM server):")
        print(read)
        print()
        print("WRITE (sends TO server):")
        print(write)
        print()
        print("SESSION ID:")
        print(_sid())


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
# MCP server definition
# ---------------------------------------------------------------------------
mcp = FastMCP(
    name="CalculatorMCPServer",
    instructions="""
        This server provides calculator tools and document resources.
        Call add() or subtract() to do arithmetic.
    """,
)


@mcp.tool
def add(a: int, b: int) -> int:
    """
    Add two integers together.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The sum of `a` and `b`.
    """
    return a + b


@mcp.tool
def subtract(a: int, b: int) -> int:
    """
    Subtract one integer from another.

    Args:
        a (int): The number to subtract from.
        b (int): The number to subtract.

    Returns:
        int: The result of `a - b`.
    """
    return a - b


@mcp.resource("file:///endpoint/{name}")
def return_template_document(name: str) -> str:
    """Read a document by name"""
    return f"Document contents of {name}"


@mcp.resource("file://endpoint2/{name}")
def read_document(name: str) -> str:
    """Read a document by name from the path directory"""
    try:
        with open(f"path/{name}", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Document '{name}' not found in path directory"
    except Exception as e:
        return f"Error reading document: {str(e)}"


@mcp.prompt(title="Code Review")
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"


# ---------------------------------------------------------------------------
# Part 1: in-memory client (talks to the server object directly)
# ---------------------------------------------------------------------------
async def demo_in_memory_client():
    section("Part 1: In-memory client")
    client = Client(mcp)

    async with client:
        response = await client.call_tool("add", {"a": 4, "b": 5})
        print("Result .data:              ", response.data)
        print("Content (as text):         ", response.content[0].text)
        print("Structured content:        ", response.structured_content)

        response = await client.call_tool("subtract", {"a": 4, "b": 5})
        print("subtract(4, 5):            ", response.content[0].text)

        tools = await client.list_tools()
        print("\nAvailable tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description.strip().splitlines()[0]}")
        print("\nInput schema of first tool: ", tools[0].inputSchema)
        print("Output schema of first tool:", tools[0].outputSchema)

        result = await client.read_resource("file:///endpoint/README.txt")
        print("\nTemplate resource:", result[0].text)

        result = await client.read_resource("file://endpoint2/README.txt")
        print("File resource (README.txt, first 200 chars):")
        print(result[0].text[:200])

        result = await client.read_resource("file://endpoint2/random.txt")
        resource = result[0]
        print(f"\nuri:      {resource.uri}")
        print(f"mimeType: {resource.mimeType}")
        print(f"meta:     {resource.meta}")
        print(f"text:     {resource.text}")

        result = await client.get_prompt("review_code", {"code": "CODE TO BE REVIEWED"})
        message = result.messages[0]
        print(f"\nPrompt Role:    {message.role}")
        print(f"Prompt Content: {message.content.text}")


# ---------------------------------------------------------------------------
# Part 2: HTTP client (server running on localhost:PORT)
# ---------------------------------------------------------------------------
async def demo_http_client():
    section("Part 2: HTTP client")
    http_client = Client(StreamableHttpTransport(url=MCP_URL))
    async with http_client:
        response = await http_client.call_tool("add", {"a": 4, "b": 5})
        print("add(4, 5) over HTTP:", response.content[0].text)
        tool_list = await http_client.list_tools()
        print("Tools over HTTP:    ", [t.name for t in tool_list])


# ---------------------------------------------------------------------------
# Part 3: LangGraph agent using MCP tools + local Ollama model
# ---------------------------------------------------------------------------
def print_messages(messages):
    for m in messages:
        if isinstance(m, HumanMessage):
            kind = "HUMAN"
        elif isinstance(m, AIMessage):
            kind = "AI"
        elif isinstance(m, ToolMessage):
            kind = "TOOL"
        else:
            kind = "OTHER"
        content = m.content
        if not content and getattr(m, "tool_calls", None):
            content = f"tool call -> {[(c['name'], c['args']) for c in m.tool_calls]}"
        print(f"[{kind}] {content}")


async def demo_agent():
    section(f"Part 3: LangGraph agent with Ollama ({OLLAMA_MODEL})")
    # reasoning=False turns off qwen3's <think> output
    llm = ChatOllama(
        model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0, reasoning=False
    )

    async with streamablehttp_client(MCP_URL) as (read, write, _sid):
        print_stream_info(read, write, _sid, verbose=False)

        async with ClientSession(read, write) as session:
            await session.initialize()

            # Load LangChain tools from the LIVE MCP session; they only work
            # while the session is open, so the agent must run in here too.
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model=llm, tools=tools)

            for question in [
                "Use the add tool to add 2 and 1 and let me know if you used a tool.",
                "What's 8 + 7? Use tools.",
            ]:
                print(f"\n--- {question}")
                response = await agent.ainvoke({"messages": question})
                print_messages(response["messages"])


# ---------------------------------------------------------------------------
async def main():
    make_dir()
    download_docs()

    await demo_in_memory_client()

    if test_port():
        raise SystemExit(f"Port {PORT} is already in use - stop the other process first.")

    server_task = asyncio.create_task(mcp.run_http_async(port=PORT, show_banner=False))
    while not test_port():  # wait until the server is listening
        await asyncio.sleep(0.1)
    print(f"\nHTTP MCP Server started in background on port {PORT}")

    try:
        await demo_http_client()
        await demo_agent()
    finally:
        # Cancelling uvicorn makes it log a harmless CancelledError traceback
        logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
        server_task.cancel()
        try:
            await server_task
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    asyncio.run(main())
