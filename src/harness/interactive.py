"""Simple interactive agent - ESC to interrupt (background monitoring)."""

import asyncio
import sys
from typing import Optional

from .agent import DEFAULT_MODELS
from .logging import run_agent_with_logging


class EscapeMonitor:
    """Monitor for ESC key in background."""

    def __init__(self):
        self.pressed = False
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._fd = None
        self._old_settings = None

    async def start(self):
        """Start monitoring."""
        import tty
        import termios

        self.pressed = False
        self._running = True
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

        self._task = asyncio.create_task(self._monitor())

    async def stop(self):
        """Stop monitoring and restore terminal."""
        import termios
        import select

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Restore terminal
        if self._old_settings and self._fd is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

        # Drain any buffered input
        try:
            while select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.read(1)
        except Exception:
            pass

    async def _monitor(self):
        """Background loop checking for ESC."""
        import select

        loop = asyncio.get_event_loop()

        while self._running:
            try:
                # Use run_in_executor to avoid blocking the event loop
                ready = await loop.run_in_executor(
                    None,
                    lambda: select.select([sys.stdin], [], [], 0.1)[0]
                )

                if ready and self._running:
                    char = sys.stdin.read(1)
                    if char == '\x1b':  # ESC
                        self.pressed = True
                        print("\n\n⚠️  ESC pressed - interrupting...", flush=True)
                        return
            except Exception:
                pass

            await asyncio.sleep(0.01)  # Small yield to event loop


async def _run_claude_interactive(
    prompt: str,
    mcp_config: dict,
    user_message: str,
    model: str,
    kwargs: Optional[dict],
) -> None:
    """Claude interactive session using ClaudeSDKClient."""
    from claude_agent_sdk import (
        ClaudeSDKClient,
        ClaudeAgentOptions,
        ResultMessage,
    )

    options = ClaudeAgentOptions(
        system_prompt=prompt,
        model=model,
        mcp_servers=mcp_config,
        permission_mode="bypassPermissions",
        include_partial_messages=True,
        **(kwargs or {})
    )

    print("\n💬 Press ESC to interrupt • Type 'exit' to quit\n", flush=True)

    client = ClaudeSDKClient(options=options)
    await client.connect()

    next_prompt = user_message
    monitor = EscapeMonitor()
    first_query = True

    try:
        while True:
            if not next_prompt:
                try:
                    next_prompt = input("\n→ You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\n👋 Exiting...")
                    break

                if next_prompt.lower() in ['exit', 'quit']:
                    break
                if not next_prompt:
                    continue

            await client.query(next_prompt)
            next_prompt = None

            # Start ESC monitoring
            await monitor.start()

            try:
                # Create agent stream generator
                async def agent_stream():
                    # Yield init message on first query
                    if first_query and hasattr(client, 'tools') and client.tools:
                        yield {
                            "type": "init",
                            "provider": "claude",
                            "model": model,
                            "tools": [{"name": getattr(tool, 'name', str(tool))} for tool in client.tools],
                        }

                    async for message in client.receive_response():
                        # Check if ESC was pressed
                        if monitor.pressed:
                            await client.interrupt()
                            monitor.pressed = False
                            continue

                        yield message

                        if isinstance(message, ResultMessage):
                            break

                # Wrap with logging adapter
                async for message in run_agent_with_logging(agent_stream()):
                    pass  # Logging adapter handles output

                first_query = False

            finally:
                await monitor.stop()

    finally:
        await client.disconnect()


async def _run_openai_interactive(
    prompt: str,
    mcp_config: dict,
    user_message: str,
    model: str,
) -> None:
    """OpenAI interactive session using openai-agents SDK."""
    from contextlib import AsyncExitStack
    from agents import Agent, Runner  # type: ignore[import-untyped]
    from .providers.openai import _build_mcp_servers
    from .providers.adapters import AdapterMessage, TextBlock, ToolCallBlock, ToolResultBlock
    import json

    mcp_servers = _build_mcp_servers(mcp_config)

    print("\n💬 Press ESC to interrupt • Type 'exit' to quit\n", flush=True)

    async with AsyncExitStack() as stack:
        for server in mcp_servers:
            await stack.enter_async_context(server)

        agent = Agent(
            name="seer-agent",
            instructions=prompt,
            mcp_servers=mcp_servers,
            model=model,
        )

        # List tools for init message
        tool_names = []
        for server in mcp_servers:
            tools = await server.list_tools()
            tool_names.extend(tools)

        next_prompt = user_message
        monitor = EscapeMonitor()
        first_query = True

        try:
            while True:
                if not next_prompt:
                    try:
                        next_prompt = input("\n→ You: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n👋 Exiting...")
                        break

                    if next_prompt.lower() in ['exit', 'quit']:
                        break
                    if not next_prompt:
                        continue

                current_input = next_prompt
                next_prompt = None

                await monitor.start()

                try:
                    async def agent_stream():
                        if first_query:
                            yield {
                                "type": "init",
                                "provider": "openai",
                                "model": model,
                                "tools": [{"name": t.name} for t in tool_names],
                            }

                        result = Runner.run_streamed(
                            starting_agent=agent,
                            input=current_input,
                            max_turns=50,
                        )

                        async for event in result.stream_events():
                            if monitor.pressed:
                                monitor.pressed = False
                                break

                            if event.type != "run_item_stream_event":
                                continue

                            if event.name == "tool_called":
                                raw = event.item.raw_item
                                name = getattr(raw, "name", None) or ""
                                call_id = getattr(raw, "call_id", None) or getattr(raw, "id", "") or ""
                                args_str = getattr(raw, "arguments", "{}")
                                try:
                                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                                except (json.JSONDecodeError, TypeError):
                                    args = {}
                                yield AdapterMessage([ToolCallBlock(name=name, id=call_id, input=args)])

                            elif event.name == "tool_output":
                                raw = event.item.raw_item
                                call_id = getattr(raw, "call_id", "") or ""
                                output = event.item.output
                                content = output if isinstance(output, str) else str(output)
                                yield AdapterMessage([ToolResultBlock(tool_use_id=call_id, content=content)])

                            elif event.name == "message_output_created":
                                raw = event.item.raw_item
                                text_parts = []
                                if hasattr(raw, "content"):
                                    for part in raw.content:
                                        if hasattr(part, "text"):
                                            text_parts.append(part.text)
                                if text_parts:
                                    yield AdapterMessage([TextBlock(text="\n".join(text_parts))])

                    async for message in run_agent_with_logging(agent_stream()):
                        pass

                    first_query = False

                finally:
                    await monitor.stop()

        except Exception:
            raise


async def _run_gemini_interactive(
    prompt: str,
    mcp_config: dict,
    user_message: str,
    model: str,
) -> None:
    """Gemini interactive session with manual agentic loop."""
    import uuid
    from contextlib import AsyncExitStack
    from google import genai
    from google.genai import types
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from .providers.adapters import AdapterMessage, TextBlock, ToolCallBlock, ToolResultBlock
    from .providers.gemini import _mcp_tool_to_function_declaration, _get_api_key, MAX_TURNS

    client = genai.Client(api_key=_get_api_key())

    print("\n💬 Press ESC to interrupt • Type 'exit' to quit\n", flush=True)

    async with AsyncExitStack() as stack:
        # Connect MCP servers
        tool_name_to_session: dict[str, ClientSession] = {}
        function_declarations = []

        for _server_name, config in mcp_config.items():
            server_params = StdioServerParameters(
                command=config["command"],
                args=config.get("args", []),
                env=config.get("env"),
            )
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

        gemini_tools = None
        if function_declarations:
            gemini_tools = [types.Tool(function_declarations=function_declarations)]

        next_prompt = user_message
        monitor = EscapeMonitor()
        first_query = True
        # Maintain conversation history across turns
        contents = []

        try:
            while True:
                if not next_prompt:
                    try:
                        next_prompt = input("\n→ You: ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print("\n👋 Exiting...")
                        break

                    if next_prompt.lower() in ['exit', 'quit']:
                        break
                    if not next_prompt:
                        continue

                current_input = next_prompt
                next_prompt = None

                # Add user message to conversation
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=current_input)],
                ))

                await monitor.start()

                try:
                    async def agent_stream():
                        if first_query:
                            yield {
                                "type": "init",
                                "provider": "gemini",
                                "model": model,
                                "tools": [{"name": fd["name"]} for fd in function_declarations],
                            }

                        # Agentic loop for this turn
                        for _ in range(MAX_TURNS):
                            if monitor.pressed:
                                monitor.pressed = False
                                break

                            config_obj = types.GenerateContentConfig(
                                system_instruction=prompt,
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

                            function_calls = []
                            text_parts = []

                            if response.candidates:
                                for part in response.candidates[0].content.parts:
                                    if part.function_call:
                                        function_calls.append(part.function_call)
                                    elif part.text:
                                        text_parts.append(part.text)

                            if not function_calls:
                                if text_parts:
                                    yield AdapterMessage([TextBlock(text="\n".join(text_parts))])
                                # Append model response to history
                                if response.candidates:
                                    contents.append(response.candidates[0].content)
                                break

                            # Append model response
                            contents.append(response.candidates[0].content)

                            # Process function calls
                            function_responses = []
                            for fc in function_calls:
                                if monitor.pressed:
                                    monitor.pressed = False
                                    return

                                call_id = str(uuid.uuid4())
                                args = dict(fc.args) if fc.args else {}

                                yield AdapterMessage([ToolCallBlock(name=fc.name, id=call_id, input=args)])

                                session = tool_name_to_session.get(fc.name)
                                if session is None:
                                    error_content = f"Unknown tool: {fc.name}"
                                    yield AdapterMessage([ToolResultBlock(
                                        tool_use_id=call_id, content=error_content, is_error=True,
                                    )])
                                    function_responses.append(
                                        types.Part.from_function_response(
                                            name=fc.name, response={"error": error_content},
                                        )
                                    )
                                    continue

                                try:
                                    result = await session.call_tool(fc.name, arguments=args)
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
                                    tool_use_id=call_id, content=result_text, is_error=is_error,
                                )])

                                function_responses.append(
                                    types.Part.from_function_response(
                                        name=fc.name,
                                        response={"result": result_text} if not is_error else {"error": result_text},
                                    )
                                )

                            contents.append(types.Content(role="user", parts=function_responses))

                    async for message in run_agent_with_logging(agent_stream()):
                        pass

                    first_query = False

                finally:
                    await monitor.stop()

        except Exception:
            raise


async def run_agent_interactive(
    prompt: str,
    mcp_config: dict,
    user_message: str = "",
    provider: str = "claude",
    model: Optional[str] = None,
    kwargs: Optional[dict] = None,
) -> None:
    """Interactive session - ESC to interrupt. Supports claude/openai/gemini."""
    model = model or DEFAULT_MODELS[provider]

    if provider == "claude":
        await _run_claude_interactive(prompt, mcp_config, user_message, model, kwargs)
    elif provider == "openai":
        await _run_openai_interactive(prompt, mcp_config, user_message, model)
    elif provider == "gemini":
        await _run_gemini_interactive(prompt, mcp_config, user_message, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
