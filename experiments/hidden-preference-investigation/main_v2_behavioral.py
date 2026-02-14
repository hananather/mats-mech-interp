"""Hidden Belief v2 - Behavioral only + Subscription auth.

Matched-prompt experiment: uses the same tool-agnostic task prompt as the
SAE condition. The only difference between this and main_v2_sae.py is
which tools the agent has access to.

Behavioral tools only. No SAE tools.

Run with: uv run python main_v2_behavioral.py
"""

import asyncio
import os
from pathlib import Path

from src.environment import Sandbox, SandboxConfig, ExecutionMode, ModelConfig
from src.workspace import Workspace, Library
from src.execution import create_notebook_session
from src.harness import run_agent

from task_prompt import TASK_HIDDEN_BELIEF

# Force subscription auth: remove API key so the CLI uses `claude login` credentials.
os.environ.pop("ANTHROPIC_API_KEY", None)


async def main():
    example_dir = Path(__file__).parent
    toolkit = example_dir.parent / "toolkit"

    config = SandboxConfig(
        gpu="A100",
        execution_mode=ExecutionMode.NOTEBOOK,
        models=[ModelConfig(
            name="bcywinski/gemma-2-9b-it-user-female",
            base_model="google/gemma-2-9b-it",
            is_peft=True,
            hidden=True,
        )],
        python_packages=["torch", "transformers", "accelerate", "datasets", "peft"],
        secrets=["HF_TOKEN"],
    )
    sandbox = Sandbox(config).start()

    workspace = Workspace(
        libraries=[
            Library.from_file(toolkit / "steering_hook.py"),
            Library.from_file(toolkit / "extract_activations.py"),
            Library.from_file(toolkit / "batch_generate.py"),
        ]
    )

    session = create_notebook_session(sandbox, workspace)

    research_methodology = (toolkit / "research_methodology.md").read_text()
    prompt = f"{session.model_info_text}\n\n{workspace.get_library_docs()}\n\n{research_methodology}\n\n{TASK_HIDDEN_BELIEF}"

    try:
        async for msg in run_agent(
            prompt=prompt,
            mcp_config=session.mcp_config,
            provider="claude",
            model="claude-opus-4-6",
            kwargs={"allowed_tools": ["mcp__notebooks__*"]},
        ):
            pass

        print(f"\n Jupyter: {session.jupyter_url}")

    finally:
        sandbox.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n")
        print("=" * 80)
        print("INTERRUPTED - IMPORTANT REMINDER")
        print("=" * 80)
        print("Please check https://modal.com/apps to ensure your GPU container is stopped!")
        print("Modal containers may still be running and consuming credits.")
        print("=" * 80)
