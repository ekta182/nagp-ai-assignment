"""MCP client: discovers and calls the tools exposed by mcp_server/.

Uses the official MCP Python SDK over stdio. A short-lived session is created
per call (`asyncio.run`) rather than holding one open, because Streamlit reruns
scripts on every interaction and a long-lived event loop owned by a background
thread is a reliable source of "attached to a different loop" errors. The cost
is ~0.5s of process spawn per tool call, which is invisible next to local LLM
inference.

The discovered tool schemas are fed to the planner prompt, which is how the
tools become "available to the AI application".
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from functools import lru_cache
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import MCP_SERVER_MODULE, MCP_TIMEOUT_SECONDS, PROJECT_ROOT


def _server_params() -> StdioServerParameters:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    kwargs = {"command": sys.executable, "args": ["-m", MCP_SERVER_MODULE], "env": env}
    try:
        return StdioServerParameters(cwd=str(PROJECT_ROOT), **kwargs)
    except Exception:  # noqa: BLE001  # older SDKs have no `cwd` field
        return StdioServerParameters(**kwargs)


async def _run(coro_factory) -> Any:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=MCP_TIMEOUT_SECONDS)
            return await asyncio.wait_for(
                coro_factory(session), timeout=MCP_TIMEOUT_SECONDS
            )


async def _list_tools_async() -> list[dict]:
    result = await _run(lambda s: s.list_tools())
    return [
        {
            "name": tool.name,
            "description": (tool.description or "").strip(),
            "input_schema": tool.inputSchema,
        }
        for tool in result.tools
    ]


async def _call_tool_async(name: str, arguments: dict) -> dict:
    result = await _run(lambda s: s.call_tool(name, arguments))

    # FastMCP serialises a dict return value as JSON inside a text content block.
    texts = [
        block.text
        for block in result.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    payload = "\n".join(texts).strip()

    if not payload:
        return {"status": "error", "tool": name, "message": "empty tool response"}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return {"status": "ok", "tool": name, "raw_text": payload}

    if isinstance(parsed, dict):
        parsed.setdefault("tool", name)
        parsed.setdefault("status", "error" if result.isError else "ok")
        return parsed
    return {"status": "ok", "tool": name, "result": parsed}


@lru_cache(maxsize=1)
def list_tools() -> list[dict]:
    """Tool catalogue from the MCP server. Cached for the process lifetime."""
    try:
        return asyncio.run(_list_tools_async())
    except Exception as exc:  # noqa: BLE001
        print(f"[mcp_client] tool discovery failed: {exc}", file=sys.stderr)
        return []


def call_tool(name: str, arguments: dict | None = None) -> dict:
    """Invoke one MCP tool. Never raises — failures come back as status=error."""
    arguments = arguments or {}
    try:
        return asyncio.run(_call_tool_async(name, arguments))
    except asyncio.TimeoutError:
        return {
            "status": "error",
            "tool": name,
            "message": f"tool timed out after {MCP_TIMEOUT_SECONDS}s",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "tool": name,
            "message": f"MCP call failed: {exc}",
        }


def tools_for_prompt() -> str:
    """Render the tool catalogue for the planner prompt."""
    tools = list_tools()
    if not tools:
        return "(no MCP tools are currently available)"
    lines = []
    for tool in tools:
        props = (tool.get("input_schema") or {}).get("properties", {}) or {}
        required = set((tool.get("input_schema") or {}).get("required", []) or [])
        args = ", ".join(
            f"{key}: {value.get('type', 'any')}"
            + ("" if key in required else " (optional)")
            for key, value in props.items()
        )
        summary = (tool["description"] or "").split("\n")[0]
        lines.append(f"- {tool['name']}({args})\n    {summary}")
    return "\n".join(lines)
