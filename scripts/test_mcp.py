"""Verify the MCP server: discovery, both capabilities, and failure handling.

    python scripts/test_mcp.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.mcp_client import call_tool, list_tools  # noqa: E402
from config import DESTINATION  # noqa: E402


def show(title: str, payload: dict) -> None:
    print(f"\n--- {title} ---")
    print(json.dumps(payload, indent=2)[:1400])


def main() -> int:
    tools = list_tools()
    print(f"Discovered {len(tools)} MCP tools:")
    for tool in tools:
        print(f"  - {tool['name']}: {(tool['description'] or '').splitlines()[0]}")

    if not tools:
        print(
            "\nNo tools discovered. Check that `mcp` is installed and that "
            "`python -m mcp_server.travel_tools` starts without error."
        )
        return 1

    show(
        "get_current_weather",
        call_tool("get_current_weather", {"location": DESTINATION}),
    )
    show(
        "get_weather_forecast (3 days)",
        call_tool("get_weather_forecast", {"location": DESTINATION, "days": 3}),
    )
    show(
        "convert_currency INR 50000 -> SGD",
        call_tool(
            "convert_currency",
            {"amount": 50000, "from_currency": "INR", "to_currency": "SGD"},
        ),
    )

    # Failure paths — these must return status="error", not raise.
    show(
        "failure: unknown location",
        call_tool("get_weather_forecast", {"location": "Zzzyxnowhere", "days": 2}),
    )
    show(
        "failure: unsupported currency",
        call_tool(
            "convert_currency",
            {"amount": 100, "from_currency": "XYZ", "to_currency": "SGD"},
        ),
    )
    print("\nAll calls returned a structured result. Next: python scripts/demo_cli.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
