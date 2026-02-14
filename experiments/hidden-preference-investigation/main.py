"""Hidden Preference Investigation - default entry point (API auth).

Runs the SAE-enabled experiment using API key auth.
Prefer main_subscription.py instead (uses Pro Max plan, no API cost).

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
