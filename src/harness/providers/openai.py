"""OpenAI provider using openai-agents SDK."""

import json
from contextlib import AsyncExitStack
from typing import AsyncIterator, Union

from agents import Agent, Runner  # type: ignore[import-untyped]
from agents.mcp import MCPServerStdio  # type: ignore[import-untyped]

from .adapters import AdapterMessage, TextBlock, ToolCallBlock, ToolResultBlock


def _build_mcp_servers(mcp_config: dict) -> list[MCPServerStdio]:
    """Convert Seer mcp_config dict to MCPServerStdio objects."""
    servers = []
    for name, config in mcp_config.items():
        params = {
            "command": config["command"],
            "args": config.get("args", []),
        }
        if "env" in config:
            params["env"] = config["env"]
        servers.append(MCPServerStdio(
            name=name,
            params=params,
            client_session_timeout_seconds=300,
        ))
    return servers


async def run_openai(
    mcp_config: dict,
    system_prompt: str,
    task: str,
    model: str = "gpt-5",
    verbose: bool = False,
) -> AsyncIterator[Union[dict, AdapterMessage]]:
    """
    Run OpenAI agent with MCP.

    Args:
        mcp_config: MCP server configuration
        system_prompt: System prompt for agent
        task: User task/query
        model: OpenAI model to use
        verbose: Print debug info

    Yields:
        Agent messages (adapter objects compatible with logging layer)
    """
    mcp_servers = _build_mcp_servers(mcp_config)

    async with AsyncExitStack() as stack:
        # Connect all MCP servers
        for server in mcp_servers:
            await stack.enter_async_context(server)

        # Create agent
        agent = Agent(
            name="seer-agent",
            instructions=system_prompt,
            mcp_servers=mcp_servers,
            model=model,
        )

        # List tools from connected servers for init message
        tool_names = []
        for server in mcp_servers:
            tools = await server.list_tools()
            tool_names.extend(tools)

        yield {
            "type": "init",
            "provider": "openai",
            "model": model,
            "tools": [{"name": t.name} for t in tool_names],
        }

        # Run agent with streaming
        result = Runner.run_streamed(
            starting_agent=agent,
            input=task,
            max_turns=50,
        )

        async for event in result.stream_events():
            if event.type != "run_item_stream_event":
                continue

            if event.name == "tool_called":
                item = event.item
                raw = item.raw_item
                # Handle both function calls and MCP calls
                name = getattr(raw, "name", None) or ""
                call_id = getattr(raw, "call_id", None) or getattr(raw, "id", "") or ""
                args_str = getattr(raw, "arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {}

                yield AdapterMessage([ToolCallBlock(
                    name=name,
                    id=call_id,
                    input=args,
                )])

            elif event.name == "tool_output":
                item = event.item
                raw = item.raw_item
                call_id = getattr(raw, "call_id", "") or ""
                output = item.output
                # output can be a string or a structured object
                if isinstance(output, str):
                    content = output
                else:
                    content = str(output)

                yield AdapterMessage([ToolResultBlock(
                    tool_use_id=call_id,
                    content=content,
                    is_error=False,
                )])

            elif event.name == "message_output_created":
                item = event.item
                raw = item.raw_item
                # Extract text from all content parts
                text_parts = []
                if hasattr(raw, "content"):
                    for part in raw.content:
                        if hasattr(part, "text"):
                            text_parts.append(part.text)
                if text_parts:
                    yield AdapterMessage([TextBlock(text="\n".join(text_parts))])
