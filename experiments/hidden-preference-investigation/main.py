"""Hidden Preference Investigation - default entry point.

Runs the SAE-enabled experiment by default.
For the behavioral-only baseline, use main_behavioral_only.py.

Run with: uv run python main.py
"""

from main_with_sae import main
import asyncio

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
