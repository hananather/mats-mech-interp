"""Test Gemini provider with a real Modal sandbox + Jupyter notebook.

This is a minimal test: spin up a sandbox (no GPU needed), give Gemini
a simple notebook task, verify it uses the MCP tools, then tear down.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import Sandbox, SandboxConfig, ExecutionMode
from src.workspace import Workspace
from src.execution import create_notebook_session
from src.harness import run_agent


async def main():
    print("=" * 60)
    print("Test: Gemini + Modal Sandbox + Jupyter Notebook")
    print("=" * 60)

    # Minimal sandbox - no GPU, no model download (saves cost)
    config = SandboxConfig(
        gpu=None,
        execution_mode=ExecutionMode.NOTEBOOK,
        python_packages=["numpy"],
    )

    print("\n⏳ Starting Modal sandbox...")
    sandbox = Sandbox(config).start()
    print("✓ Sandbox started")

    workspace = Workspace(libraries=[])
    session = create_notebook_session(sandbox, workspace)
    print(f"✓ Notebook session ready")

    task = """Use the Jupyter notebook to:
1. Run: import numpy as np; print(np.array([1,2,3]) * 2)
2. Run: print("hello from modal sandbox")

That's it. Be brief."""

    prompt = f"{session.model_info_text}\n\n{task}"

    tool_calls = []
    try:
        print("\n⏳ Running Gemini agent...\n")
        async for msg in run_agent(
            prompt=prompt,
            mcp_config=session.mcp_config,
            user_message=task,
            provider="gemini",
            model="gemini-2.0-flash",
        ):
            if hasattr(msg, 'content'):
                for block in msg.content:
                    if hasattr(block, 'name') and hasattr(block, 'id'):
                        tool_calls.append(block.name)

        print(f"\n\n✓ Agent made {len(tool_calls)} tool calls: {tool_calls}")
        assert len(tool_calls) > 0, "Gemini should have used notebook tools"
        print("✓ Test passed!")

    finally:
        print("\n⏳ Terminating sandbox...")
        sandbox.terminate()
        print("✓ Sandbox terminated")


if __name__ == "__main__":
    asyncio.run(main())
