"""Test Gemini provider - basic response + tool calling via MCP."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import run_agent


async def test_basic_response():
    """Test 1: Simple text response, no tools."""
    print("\n" + "=" * 60)
    print("Test 1: Gemini basic response (no tools)")
    print("=" * 60)

    messages = []
    async for msg in run_agent(
        prompt="You are a helpful assistant. Be very brief.",
        mcp_config={},
        user_message="What is 2 + 2? Reply with just the number.",
        provider="gemini",
        model="gemini-2.0-flash",
    ):
        messages.append(msg)

    print(f"\n✓ Got {len(messages)} messages")
    assert len(messages) >= 1, "Should get at least init + text message"
    print("✓ Test 1 passed!\n")


async def test_with_mcp_tools():
    """Test 2: Tool calling via a simple MCP server."""
    print("=" * 60)
    print("Test 2: Gemini with MCP tool calling")
    print("=" * 60)

    # Use a minimal MCP server that provides a single tool
    mcp_server_code = '''
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
from mcp.server.stdio import stdio_server
from mcp.server import Server
import mcp.types as types
import asyncio

server = Server("test-server")

@server.list_tools()
async def list_tools():
    return [types.Tool(
        name="add_numbers",
        description="Add two numbers together",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    )]

@server.call_tool()
async def call_tool(name, arguments):
    if name == "add_numbers":
        result = arguments["a"] + arguments["b"]
        return [types.TextContent(type="text", text=str(result))]
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

asyncio.run(main())
'''

    # Write the temp MCP server
    server_path = Path(__file__).parent / "_temp_mcp_server.py"
    server_path.write_text(mcp_server_code)

    try:
        mcp_config = {
            "test-server": {
                "command": sys.executable,
                "args": [str(server_path)],
            }
        }

        messages = []
        tool_calls_seen = False
        tool_results_seen = False

        async for msg in run_agent(
            prompt="You have an add_numbers tool. Use it to answer the user's question. Be brief.",
            mcp_config=mcp_config,
            user_message="What is 17 + 25? Use the add_numbers tool.",
            provider="gemini",
            model="gemini-2.0-flash",
        ):
            messages.append(msg)
            # Check for adapter messages with tool calls/results
            if hasattr(msg, 'content'):
                for block in msg.content:
                    if hasattr(block, 'name') and hasattr(block, 'id'):
                        tool_calls_seen = True
                        print(f"  → Tool called: {block.name}({block.input})")
                    if hasattr(block, 'tool_use_id'):
                        tool_results_seen = True
                        print(f"  → Tool result: {block.content}")

        print(f"\n✓ Got {len(messages)} messages")
        assert tool_calls_seen, "Should have seen tool calls"
        assert tool_results_seen, "Should have seen tool results"
        print("✓ Tool call + result logged correctly")
        print("✓ Test 2 passed!\n")

    finally:
        server_path.unlink(missing_ok=True)


async def main():
    await test_basic_response()
    await test_with_mcp_tools()
    print("=" * 60)
    print("All Gemini provider tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
