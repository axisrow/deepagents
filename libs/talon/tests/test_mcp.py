from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import pytest

from deepagents_code.mcp_tools import MCPServerInfo as CodeMCPServerInfo, MCPToolInfo
from deepagents_talon.config import TalonConfig
from deepagents_talon.mcp import MCPConfigError, discover_mcp_config_paths, load_mcp_tools


@dataclass(frozen=True)
class DummyTool:
    name: str


FakeCodeLoaderResult: TypeAlias = tuple[list[DummyTool], None, list[CodeMCPServerInfo]]


def _fake_code_loader(path: str) -> FakeCodeLoaderResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tools = [
        DummyTool("files_read"),
        DummyTool("files_write"),
        DummyTool("search"),
    ]
    infos = [
        CodeMCPServerInfo(
            name=name,
            transport=str(server.get("type") or server.get("transport") or "stdio"),
            tools=tuple(MCPToolInfo(name=tool.name, description="") for tool in tools),
        )
        for name, server in data["mcpServers"].items()
        if isinstance(server, dict)
    ]
    return tools, None, infos


def _write_mcp_config(path: Path, server_name: str = "server") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"mcpServers": {server_name: {"type": "stdio", "command": "server"}}}),
        encoding="utf-8",
    )


async def test_load_mcp_tools_reads_manifest_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Path] = []

    async def fake_loader(path: str) -> FakeCodeLoaderResult:
        seen.append(Path(path))
        return _fake_code_loader(path)

    monkeypatch.setattr("deepagents_talon.mcp.get_mcp_tools", fake_loader)
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    config.ensure_home()
    tools_path = config.manifest_dir / ".mcp.json"
    tools_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote": {
                        "transport": "sse",
                        "url": "https://tools.example/sse",
                        "headers": {"Authorization": "Bearer ${TOKEN}"},
                    },
                },
            },
        ),
    )

    result = await load_mcp_tools(config)

    assert [tool.name for tool in result.tools] == ["files_read", "files_write", "search"]
    assert seen == [tools_path]


async def test_load_mcp_tools_prefers_env_config_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Path] = []

    async def fake_loader(path: str) -> FakeCodeLoaderResult:
        seen.append(Path(path))
        return _fake_code_loader(path)

    monkeypatch.setattr("deepagents_talon.mcp.get_mcp_tools", fake_loader)
    env_path = tmp_path / "custom-tools.json"
    env_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "custom": {
                        "type": "stdio",
                        "command": "server",
                    },
                },
            },
        ),
    )
    config = TalonConfig.from_env(
        {
            "AGENT_ASSISTANT_ID": "test",
            "DEEPAGENTS_TALON_MCP_CONFIG": str(env_path),
        },
        base_home=tmp_path,
    )
    config.ensure_home()
    (config.manifest_dir / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "manifest": {
                        "type": "stdio",
                        "command": "server",
                    },
                },
            },
        ),
    )

    await load_mcp_tools(config)

    assert seen == [env_path]


async def test_load_mcp_tools_rejects_malformed_assistant_config_over_global(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Path] = []

    async def fake_loader(path: str) -> FakeCodeLoaderResult:
        seen.append(Path(path))
        return _fake_code_loader(path)

    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr("deepagents_talon.mcp.get_mcp_tools", fake_loader)
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    config.ensure_home()
    global_path = home / ".deepagents" / ".mcp.json"
    global_path.parent.mkdir(parents=True)
    global_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "global": {
                        "type": "stdio",
                        "command": "server",
                    },
                },
            },
        ),
        encoding="utf-8",
    )
    local_path = config.manifest_dir / ".mcp.json"
    local_path.write_text("{", encoding="utf-8")

    with pytest.raises(MCPConfigError):
        await load_mcp_tools(config)

    assert seen == [local_path]


def test_discover_mcp_config_paths_ignores_legacy_tools_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    config.ensure_home()
    (config.manifest_dir / "tools.json").write_text("{}", encoding="utf-8")
    config_path = config.manifest_dir / ".mcp.json"
    _write_mcp_config(config_path)

    assert discover_mcp_config_paths(config) == [config_path]


@pytest.mark.parametrize(
    ("global_content", "assistant_content", "expected"),
    [
        (
            json.dumps({"mcpServers": {"global": {"type": "stdio", "command": "server"}}}),
            json.dumps({"mcpServers": {"assistant": {"type": "stdio", "command": "server"}}}),
            ("global", "assistant"),
        ),
        (
            json.dumps({"mcpServers": {"global": {"type": "stdio", "command": "server"}}}),
            None,
            ("global",),
        ),
        (None, '{"mcpServers": {}}', ("assistant",)),
        (None, "{bad", ("assistant",)),
    ],
)
def test_discover_mcp_config_paths_returns_existing_configs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    global_content: str | None,
    assistant_content: str | None,
    expected: tuple[str, ...],
) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    config = TalonConfig.from_env({"AGENT_ASSISTANT_ID": "test"}, base_home=tmp_path)
    config.ensure_home()
    global_path = home / ".deepagents" / ".mcp.json"
    assistant_path = config.manifest_dir / ".mcp.json"
    paths = {"global": global_path, "assistant": assistant_path}
    if global_content is not None:
        global_path.parent.mkdir(parents=True)
        global_path.write_text(global_content, encoding="utf-8")
    if assistant_content is not None:
        assistant_path.write_text(assistant_content, encoding="utf-8")

    assert discover_mcp_config_paths(config) == [paths[name] for name in expected]
