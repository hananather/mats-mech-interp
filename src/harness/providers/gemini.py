"""Gemini provider using google-genai SDK with manual agentic loop."""

import asyncio
import os
import uuid
from contextlib import AsyncExitStack
from typing import AsyncIterator

from google import genai
from google.genai import types


def _get_api_key() -> str:
    """Get Gemini API key from environment."""
    for var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        key = os.environ.get(var)
        if key:
            return key
    raise ValueError(
        "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
    )
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .adapters import AdapterMessage, TextBlock, ToolCallBlock, ToolResultBlock

MAX_TURNS = 50


def _mcp_tool_to_function_declaration(tool) -> dict:
    """Convert an MCP tool to a Gemini function declaration dict."""
    schema = tool.inputSchema or {"type": "object", "properties": {}}
    return {
        "name": tool.name,
        "description": tool.description or "",
        "parameters": schema,
    }


async def run_gemini(
    mcp_config: dict,
    system_prompt: str,
    task: str,
    model: str = "gemini-3-pro-preview",
    verbose: bool = False,
) -> AsyncIterator:
    """
    Run Gemini agent with MCP.

    Uses a manual agentic loop: call generate_content, check for function
    calls, execute them via MCP, feed results back, repeat.

    Args:
        mcp_config: MCP server configuration
        system_prompt: System prompt for agent
        task: User task/query
        model: Gemini model to use
        verbose: Print debug info

    Yields:
        Agent messages (adapter objects compatible with logging layer)
    """
    client = genai.Client(api_key=_get_api_key())

    async with AsyncExitStack() as stack:
        # Connect MCP servers and collect tools
        tool_name_to_session: dict[str, ClientSession] = {}
        function_declarations = []

        for _server_name, config in mcp_config.items():
            server_params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env"),
            )
            # stdio_client returns (read_stream, write_stream)
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(server_params)
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()

            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                tool_name_to_session[tool.name] = session
                function_declarations.append(_mcp_tool_to_function_declaration(tool))

        # Build Gemini tools config
        gemini_tools = None
        if function_declarations:
            gemini_tools = [types.Tool(function_declarations=function_declarations)]

        # Yield init message
        yield {
            "type": "init",
            "provider": "gemini",
            "model": model,
            "tools": [{"name": fd["name"]} for fd in function_declarations],
        }

        # Build initial conversation
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=task)])]

        # Agentic loop
        for turn in range(MAX_TURNS):
            # Rate limit: avoid 429s on free-tier APIs
            if turn > 0:
                await asyncio.sleep(8)

            config_obj = types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=gemini_tools,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True,
                ),
            )

            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
                config=config_obj,
            )

            # Check if response has function calls
            function_calls = []
            text_parts = []

            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        function_calls.append(part.function_call)
                    elif part.text:
                        text_parts.append(part.text)

            if not function_calls:
                # No function calls - emit text and stop
                if text_parts:
                    yield AdapterMessage([TextBlock(text="\n".join(text_parts))])
                break

            # Append model response to conversation
            contents.append(response.candidates[0].content)

            # Process function calls
            function_responses = []
            for fc in function_calls:
                call_id = str(uuid.uuid4())
                args = dict(fc.args) if fc.args else {}

                # Yield tool call for logging
                yield AdapterMessage([ToolCallBlock(
                    name=fc.name,
                    id=call_id,
                    input=args,
                )])

                # Execute via MCP
                session = tool_name_to_session.get(fc.name)
                if session is None:
                    error_content = f"Unknown tool: {fc.name}"
                    yield AdapterMessage([ToolResultBlock(
                        tool_use_id=call_id,
                        content=error_content,
                        is_error=True,
                    )])
                    function_responses.append(
                        types.Part.from_function_response(
                            name=fc.name,
                            response={"error": error_content},
                        )
                    )
                    continue

                try:
                    result = await session.call_tool(fc.name, arguments=args)
                    # Extract text from result content
                    result_text_parts = []
                    for item in result.content:
                        if hasattr(item, "text"):
                            result_text_parts.append(item.text)
                        else:
                            result_text_parts.append(str(item))
                    result_text = "\n".join(result_text_parts)
                    is_error = bool(result.isError)
                except Exception as e:
                    result_text = str(e)
                    is_error = True

                yield AdapterMessage([ToolResultBlock(
                    tool_use_id=call_id,
                    content=result_text,
                    is_error=is_error,
                )])

                function_responses.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result_text} if not is_error else {"error": result_text},
                    )
                )

            # Add function responses to conversation
            contents.append(types.Content(role="user", parts=function_responses))
