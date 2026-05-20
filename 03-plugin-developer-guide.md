# Forge CLI — Plugin Developer Guide

> **Audience:** Engineering teams building commands for the Forge CLI platform.

---

## Overview

Forge is a centralized CLI platform. You write Click commands, wrap them in a small plugin class, publish a Python package, and the core discovers your commands automatically at runtime. You never touch the core code, the core team never touches yours.

The contract has three parts:

1. **A plugin class** — tells the core your metadata and registers your commands
2. **An entry point** — one line in `pyproject.toml` so the core discovers you
3. **SDK decorators** — `@require_auth` and `@require_permission(...)` on your commands

Everything else — login flows, token refresh, secure storage, RBAC evaluation, output formatting, error handling — is handled by the core.

---

## What You Get for Free

```python
# Authentication — one decorator
@require_auth
# Ensures valid token, handles refresh, prompts re-login if needed

# Authorization — one decorator
@require_permission("data:pipelines", "execute")
# Checks RBAC, shows clear denial message

# Token access — one method call
token = ctx.obj.get_token()
# Always valid, already refreshed if needed

# Output formatting — respects user's --output flag
ctx.obj.render(data, columns=["name", "status", "owner"])
# --output json → JSON    --output table → aligned table    --output yaml → YAML

# Config access — read forge-wide or plugin-scoped config
region = ctx.obj.config.get("data.default_warehouse", fallback="primary")

# Error handling — throw, the core catches and formats
raise click.ClickException("Pipeline 'foo' not found")
# Core renders it cleanly, respects --verbose for stack traces
```

---

## Step-by-Step: Building a Plugin

### Step 1 — Scaffold the Package

Create a standard Python package. (A `forge plugin init <name>` template generator will be available.)

```
forge-data-plugin/
├── src/
│   └── forge_data/
│       ├── __init__.py
│       ├── plugin.py          # Plugin registration class
│       ├── commands/
│       │   ├── __init__.py
│       │   ├── pipelines.py   # forge data pipeline ...
│       │   ├── warehouses.py  # forge data warehouse ...
│       │   └── jobs.py        # forge data job ...
│       └── client.py          # Your API client (your business logic)
├── pyproject.toml
├── tests/
│   ├── test_pipelines.py
│   └── test_warehouses.py
└── README.md
```

### Step 2 — Write Your Commands

Write plain Click commands. Nothing special beyond the SDK decorators:

```python
# forge_data/commands/pipelines.py
import click
from forge.plugins.sdk import require_auth, require_permission
from forge_data.client import DataAPIClient


@click.group()
def pipeline():
    """Manage data pipelines."""
    pass


@pipeline.command()
@require_auth
@require_permission("data:pipelines", "read")
@click.option(
    "--status",
    type=click.Choice(["active", "paused", "failed", "all"]),
    default="all",
)
@click.pass_context
def list(ctx, status):
    """List all pipelines."""
    token = ctx.obj.get_token()
    client = DataAPIClient(token=token)
    pipelines = client.list_pipelines(status=status)

    # Return structured data — the core formats it
    ctx.obj.render(pipelines, columns=["name", "status", "last_run", "owner"])


@pipeline.command()
@require_auth
@require_permission("data:pipelines", "execute")
@click.argument("name")
@click.option("--params", "-p", multiple=True, help="Key=value parameters")
@click.option("--wait/--no-wait", default=False, help="Wait for completion")
@click.pass_context
def run(ctx, name, params, wait):
    """Trigger a pipeline run."""
    token = ctx.obj.get_token()
    client = DataAPIClient(token=token)

    parsed_params = dict(p.split("=", 1) for p in params)
    result = client.trigger_run(name, parsed_params)

    click.echo(f"Pipeline '{name}' triggered. Run ID: {result['run_id']}")

    if wait:
        click.echo("Waiting for completion...")
        final = client.wait_for_completion(result["run_id"])
        click.echo(f"Completed with status: {final['status']}")


@pipeline.command()
@require_auth
@require_permission("data:pipelines", "read")
@click.argument("name")
@click.option("--lines", "-n", default=50)
@click.pass_context
def logs(ctx, name, lines):
    """Tail logs for a pipeline."""
    token = ctx.obj.get_token()
    client = DataAPIClient(token=token)
    for line in client.stream_logs(name, tail=lines):
        click.echo(line)
```

Another command group in the same plugin:

```python
# forge_data/commands/warehouses.py
import click
from forge.plugins.sdk import require_auth, require_permission
from forge_data.client import DataAPIClient


@click.group()
def warehouse():
    """Manage data warehouses."""
    pass


@warehouse.command()
@require_auth
@require_permission("data:warehouses", "read")
@click.argument("query")
@click.option("--warehouse", "-w", default="primary")
@click.option("--limit", default=100)
@click.pass_context
def query(ctx, query, warehouse, limit):
    """Run a SQL query against a warehouse."""
    token = ctx.obj.get_token()
    client = DataAPIClient(token=token)
    results = client.run_query(warehouse, query, limit=limit)
    ctx.obj.render(results["rows"], columns=results["columns"])
```

### Step 3 — Write the Plugin Class

This is the glue. It tells forge your metadata and attaches your command groups:

```python
# forge_data/plugin.py
import click
from forge.plugins.sdk import ForgePlugin, PluginMeta
from forge_data.commands.pipelines import pipeline
from forge_data.commands.warehouses import warehouse
from forge_data.commands.jobs import job


class DataPlugin(ForgePlugin):
    meta = PluginMeta(
        name="data",
        version="1.2.0",
        description="Data pipeline and warehouse management",
        team="data-engineering",
        min_core_version="1.0.0",
    )

    def register(self, group: click.Group):
        """Register all command groups under 'forge data ...'"""

        @click.group()
        def data():
            """Data engineering tools."""
            pass

        data.add_command(pipeline)
        data.add_command(warehouse)
        data.add_command(job)

        group.add_command(data)
```

After registration, the command tree looks like:

```
forge
├── auth          (builtin)
├── config        (builtin)
├── plugins       (builtin)
└── data          ← Your plugin
    ├── pipeline
    │   ├── list
    │   ├── run
    │   └── logs
    ├── warehouse
    │   └── query
    └── job
        ├── list
        └── cancel
```

### Step 4 — Declare the Entry Point

In your `pyproject.toml`, add one line under `[project.entry-points]`:

```toml
[project]
name = "forge-data-plugin"
version = "1.2.0"
description = "Data engineering commands for Forge CLI"
requires-python = ">=3.10"
dependencies = [
    "forge-cli-sdk>=1.0.0,<2.0.0",  # The SDK package (slim, stable)
    "httpx>=0.27",                    # Your own dependencies
]

[project.entry-points."forge.plugins"]
data = "forge_data.plugin:DataPlugin"
```

The key line is:

```
data = "forge_data.plugin:DataPlugin"
 ^           ^                ^
 |           |                └── Class inside the module
 |           └── Dotted path to the module
 └── Plugin name (used for logging/debugging)
```

This is a standard Python packaging mechanism (`importlib.metadata` entry points). When your package is installed, the forge core discovers it automatically. **No config file to edit, no registration command to run.**

### Step 5 — Install and Use

```bash
pip install forge-data-plugin

# Commands are immediately available
forge data pipeline list --status active
forge data pipeline run etl-daily -p date=2026-05-19 --wait
forge data warehouse query "SELECT count(*) FROM events" -w analytics
```

---

## Plugin SDK Reference

The SDK package (`forge-cli-sdk`) exposes these public APIs:

```python
from forge.plugins.sdk import (
    # Plugin registration
    ForgePlugin,          # Base class to subclass
    PluginMeta,           # Metadata dataclass

    # Auth decorators
    require_auth,         # Ensure user is authenticated
    require_permission,   # Check RBAC before command runs
)
```

### ForgePlugin (Base Class)

```python
from dataclasses import dataclass

@dataclass
class PluginMeta:
    name: str              # Top-level command name (e.g., "data")
    version: str           # Semver (e.g., "1.2.0")
    description: str       # One-line description
    team: str              # Owning team name
    min_core_version: str = "1.0.0"  # Minimum forge-cli version required

class ForgePlugin:
    """Base class. Subclass this and implement register()."""
    meta: PluginMeta

    def register(self, group: click.Group):
        """Required: attach your commands to the CLI group."""
        raise NotImplementedError

    def on_activate(self, context):
        """Optional: called once when the plugin loads.
        Use for: checking dependencies, verifying connectivity."""
        pass

    def health_check(self) -> dict:
        """Optional: called by 'forge plugins health'.
        Return status dict for diagnostics."""
        return {"status": "ok"}
```

### @require_auth

Ensures the user is authenticated before the command runs. Handles token refresh transparently. If the session is expired and can't be refreshed, shows a clear message directing the user to `forge auth login`.

```python
@pipeline.command()
@require_auth
@click.pass_context
def list(ctx):
    token = ctx.obj.get_token()  # Guaranteed to be valid here
    ...
```

### @require_permission(resource, action)

Checks RBAC before the command runs. If denied, shows a clear message with the required permission.

```python
@pipeline.command()
@require_auth
@require_permission("data:pipelines", "execute")
@click.pass_context
def run(ctx, name):
    ...  # Only reached if user has data:pipelines:execute permission
```

Permission string format: `"plugin:entity"` + `"action"`

| Plugin | Entity | Actions |
|--------|--------|---------|
| `data` | `pipelines` | `read`, `execute`, `write`, `delete` |
| `data` | `warehouses` | `read`, `write`, `query` |
| `infra` | `stacks` | `read`, `deploy`, `delete` |
| `security` | `scans` | `read`, `write`, `execute` |

You define what permissions your commands need. The platform team maps IdP groups to those permissions in the RBAC policy.

---

## ForgeContext — What's Available on `ctx.obj`

Every command that uses `@click.pass_context` gets a `ForgeContext` instance at `ctx.obj`:

| Method/Property | Description |
|----------------|-------------|
| `ctx.obj.get_token()` | Returns a valid access token (refreshes if needed) |
| `ctx.obj.auth` | The `AuthManager` instance |
| `ctx.obj.config` | The `ConfigStore` instance |
| `ctx.obj.render(data, columns)` | Format data per the user's `--output` flag |
| `ctx.obj.check_permission(resource, action)` | Returns `bool` |
| `ctx.obj.require_permission(resource, action)` | Raises on denial |
| `ctx.obj.output_format` | Current output format (`"table"`, `"json"`, `"yaml"`) |
| `ctx.obj.verbose` | Whether `--verbose` was passed |
| `ctx.obj.profile` | Current auth profile name |

### Reading Plugin-Scoped Config

```python
# User's ~/.forge/config.toml:
# [data]
# default_warehouse = "analytics"
# query_timeout = 30

warehouse = ctx.obj.config.get("data.default_warehouse", fallback="primary")
timeout = ctx.obj.config.get("data.query_timeout", fallback=60)
```

---

## Testing Your Plugin

Use the test harness to test commands in isolation:

```python
# tests/test_pipelines.py
from forge.testing import CliRunner, mock_auth


def test_pipeline_list():
    runner = CliRunner()
    with mock_auth(user="engineer@company.com", groups=["data-eng"]):
        result = runner.invoke(["data", "pipeline", "list"])
        assert result.exit_code == 0
        assert "etl-daily" in result.output


def test_pipeline_list_json_output():
    runner = CliRunner()
    with mock_auth(user="engineer@company.com", groups=["data-eng"]):
        result = runner.invoke(["data", "pipeline", "list", "--output", "json"])
        assert result.exit_code == 0
        import json
        data = json.loads(result.output)
        assert isinstance(data, list)


def test_pipeline_run_permission_denied():
    runner = CliRunner()
    with mock_auth(user="readonly@company.com", groups=["viewer"]):
        result = runner.invoke(["data", "pipeline", "run", "etl-daily"])
        assert result.exit_code == 1
        assert "Permission denied" in result.output


def test_pipeline_run_unauthenticated():
    runner = CliRunner()
    # No mock_auth → simulates unauthenticated user
    result = runner.invoke(["data", "pipeline", "run", "etl-daily"])
    assert result.exit_code == 1
    assert "forge auth login" in result.output
```

---

## Packaging Conventions

| Convention | Rule |
|-----------|------|
| Package name | `forge-<plugin-name>-plugin` (e.g., `forge-data-plugin`) |
| Module name | `forge_<plugin_name>` (e.g., `forge_data`) |
| Entry point group | `forge.plugins` |
| SDK dependency | `forge-cli-sdk>=1.0.0,<2.0.0` |
| Top-level command | One word, lowercase (e.g., `data`, `infra`, `security`) |
| Permission format | `plugin:entity:action` (e.g., `data:pipelines:execute`) |

### Version Compatibility

Pin the SDK within a major version. The platform team commits to backward compatibility within major versions:

```toml
dependencies = [
    "forge-cli-sdk>=1.0.0,<2.0.0",
]
```

### Namespace Ownership

Each team claims a top-level command name. No collisions. The platform team maintains a registry. Claim yours before starting development.

| Command | Team | Description |
|---------|------|-------------|
| `infra` | Platform Engineering | Infrastructure management |
| `data` | Data Engineering | Pipelines and warehouses |
| `security` | Security | Scanning and compliance |
| `deploy` | DevOps | Deployment orchestration |
| `tools` | Platform Engineering | Internal tool integrations |

---

## Common Patterns

### Calling Your Backend API

```python
# forge_data/client.py
import httpx
from typing import Optional

class DataAPIClient:
    """Your team's API client. The token comes from forge core."""

    def __init__(self, token: str, base_url: str = "https://data-api.internal.company.com"):
        self.http = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )

    def list_pipelines(self, status: str = "all") -> list[dict]:
        params = {} if status == "all" else {"status": status}
        resp = self.http.get("/v1/pipelines", params=params)
        resp.raise_for_status()
        return resp.json()["pipelines"]

    def trigger_run(self, name: str, params: dict) -> dict:
        resp = self.http.post(f"/v1/pipelines/{name}/runs", json={"params": params})
        resp.raise_for_status()
        return resp.json()
```

### Progress Indicators

```python
@pipeline.command()
@require_auth
@click.pass_context
def sync(ctx, source, target):
    """Sync data between sources."""
    token = ctx.obj.get_token()
    client = DataAPIClient(token=token)

    with click.progressbar(length=100, label="Syncing") as bar:
        for progress in client.sync_with_progress(source, target):
            bar.update(progress.percent - bar.pos)

    click.echo("Sync complete.")
```

### Confirmation Prompts for Destructive Actions

```python
@pipeline.command()
@require_auth
@require_permission("data:pipelines", "delete")
@click.argument("name")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation")
@click.pass_context
def delete(ctx, name, yes):
    """Delete a pipeline permanently."""
    if not yes:
        click.confirm(f"Delete pipeline '{name}'? This cannot be undone", abort=True)

    token = ctx.obj.get_token()
    client = DataAPIClient(token=token)
    client.delete_pipeline(name)
    click.echo(f"Pipeline '{name}' deleted.")
```

### Subcommand with Nested Groups

```python
# forge data pipeline schedule list
# forge data pipeline schedule set <name> --cron "0 * * * *"

@pipeline.group()
def schedule():
    """Manage pipeline schedules."""
    pass

@schedule.command()
@require_auth
@require_permission("data:pipelines", "read")
@click.pass_context
def list(ctx):
    ...

@schedule.command()
@require_auth
@require_permission("data:pipelines", "write")
@click.argument("name")
@click.option("--cron", required=True)
@click.pass_context
def set(ctx, name, cron):
    ...
```

---

## Checklist Before Publishing

- [ ] Plugin class extends `ForgePlugin` with complete `PluginMeta`
- [ ] Entry point declared in `pyproject.toml` under `forge.plugins`
- [ ] SDK dependency pinned: `forge-cli-sdk>=1.0.0,<2.0.0`
- [ ] All commands use `@require_auth` where needed
- [ ] All commands use `@require_permission(...)` with correct resource/action
- [ ] Commands use `ctx.obj.render()` for structured output (not raw `print`)
- [ ] Destructive commands have `--yes` flag or `click.confirm()`
- [ ] Tests use `forge.testing.CliRunner` and `mock_auth`
- [ ] Top-level command name registered with the platform team
- [ ] README documents available commands and required permissions
- [ ] `forge plugins health` reports correctly via `health_check()` (optional)
