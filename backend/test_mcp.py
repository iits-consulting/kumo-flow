"""Smoke tests for the MCP server (mcp_server.py): tool registration/schemas
and validate parity with the CLI. Everything else is covered by test_cli.py —
the logic is shared by import.

    uv run --extra agent pytest test_mcp.py
"""

import anyio

import cli
import mcp_server

# unknown kind -> error; 'v' is off every target path -> warning
BAD = {
    "nodes": [{"id": "x", "kind": "nope"}, {"id": "v", "kind": "view"}],
    "edges": [],
    "targets": ["x"],
}


def test_registers_eight_tools_with_expected_schemas():
    tools = {t.name: t for t in anyio.run(mcp_server.mcp.list_tools)}
    assert set(tools) == {
        "catalog", "validate_workflow", "run_workflow", "get_progress",
        "stop_run", "fetch_media", "get_canvas", "set_canvas",
    }

    expected = {
        "catalog": ({"input_kind"}, set()),
        "validate_workflow": ({"workflow", "targets"}, {"workflow"}),
        "run_workflow": ({"workflow", "targets", "run_id"}, {"workflow"}),
        "get_progress": ({"run_id"}, {"run_id"}),
        "stop_run": ({"run_id", "target"}, {"run_id"}),
        "fetch_media": ({"media_url", "out_path"}, {"media_url", "out_path"}),
        "get_canvas": (set(), set()),
        "set_canvas": ({"workflow", "rev"}, {"workflow"}),
    }
    for name, (props, required) in expected.items():
        schema = tools[name].input_schema
        assert set(schema["properties"]) == props, name
        assert set(schema.get("required", [])) == required, name
        assert tools[name].description  # docstring became the tool description


def test_validate_workflow_matches_cli():
    out = mcp_server.validate_workflow(BAD)
    errors, warnings = cli.validate_graph(cli.parse_workflow(BAD), ["x"])
    assert errors and warnings  # the fixture exercises both lists
    assert out == {"errors": errors, "warnings": warnings}
    assert "unknown kind 'nope'" in out["errors"][0]


def test_instructions_are_agents_md_minus_cli_and_mcp_sections():
    text = mcp_server._instructions()
    assert "## Port contract" in text and "## Hard rules" in text
    assert "## Recipes" in text  # single-sourced compositional recipes flow through
    assert "map every clause" in text  # the verify-by-looking rule survives the section filter
    assert "## CLI" not in text and "## MCP server" not in text
    assert mcp_server.mcp.instructions == text  # actually wired into the server


def test_set_canvas_rejects_invalid_workflow_before_any_http():
    try:
        mcp_server.set_canvas(BAD)  # unknown kind -> refused locally, no backend needed
        assert False, "expected ValueError"
    except ValueError as e:
        assert "unknown kind 'nope'" in str(e)
