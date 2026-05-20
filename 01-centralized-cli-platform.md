# Forge CLI — Centralized CLI Platform Proposal

## The Problem

Multiple teams are independently building CLI tools for their internal integrations (AIDLC / Kiro workflows, infrastructure, data, security, etc.). This creates:

- **Fragmented authentication** — each tool has its own login, its own credentials, its own token management
- **Inconsistent UX** — different output formats, error messages, flag conventions, help text styles
- **Duplicated infrastructure** — every team re-implements config, logging, output formatting, distribution
- **Painful onboarding** — new engineers install 5 different tools, manage 5 sets of credentials, learn 5 different patterns

## The Vision

A single CLI binary — `forge` — that acts as a **platform**. Teams register their functionality as **plugins** (commands/subcommands), while the core handles auth, RBAC, config, output formatting, logging, and distribution.

```bash
forge <plugin> <command> [args] [flags]

forge infra deploy --env staging
forge data pipeline run --name etl-daily
forge security scan --repo my-service
forge tools list
```

Users install one tool, authenticate once, and get a consistent experience across every team's commands.

---

## Architecture Overview

There are three layers:

| Layer | Owner | Responsibility |
|-------|-------|----------------|
| **Core Shell** | Platform team (you) | CLI parsing, authentication, authorization, configuration, output formatting, telemetry, error handling, plugin discovery/loading |
| **Plugin System** | Platform team (you) | The contract that other teams build against. Entry point discovery, lifecycle hooks, registration API |
| **Developer SDK** | Platform team (you) | A small library plugin authors import. Decorators, base classes, utilities |

Plugins are what other teams build. Each plugin is a Python package that exposes commands through the SDK interface. Plugins never handle auth themselves — they receive an authenticated context from the core.

---

## Plugin Architecture: In-Process Plugins (Recommended)

### Why In-Process Over Subprocess

There are two common approaches:

**Option A: In-process plugins (recommended).** Plugins are Python packages installed into the same environment. They register commands via entry points. Simpler, faster, easier to debug. The tradeoff is that a bad plugin can crash the CLI, but for internal tooling with trusted teams, this is acceptable.

**Option B: Subprocess plugins (the kubectl/git model).** Plugins are standalone binaries named `forge-<plugin>`. The core discovers them on PATH and shells out. This gives language independence and process isolation, but auth passing gets harder (serializing tokens to env vars or temp files), and the UX for errors and output formatting degrades.

For an internal Python organization, **Option A is the right call**.

---

## Project Structure

```
forge-cli/
├── src/
│   └── forge/
│       ├── __init__.py
│       ├── main.py              # Entry point
│       ├── core/
│       │   ├── cli.py           # Click group, plugin loading
│       │   ├── config.py        # Config management (~/.forge/)
│       │   ├── context.py       # ForgeContext passed to all commands
│       │   ├── output.py        # Formatters (table, json, yaml, plain)
│       │   └── errors.py        # Standardized error handling
│       ├── auth/
│       │   ├── manager.py       # Token lifecycle
│       │   ├── oauth.py         # OAuth/OIDC device flow
│       │   ├── token_store.py   # Secure token storage
│       │   └── rbac.py          # Permission checks
│       ├── plugins/
│       │   ├── loader.py        # Discovery and loading
│       │   ├── registry.py      # Plugin metadata registry
│       │   └── sdk.py           # Base classes for plugin authors
│       └── builtin/
│           ├── auth_commands.py  # forge auth login/logout/status
│           ├── config_commands.py# forge config set/get/list
│           └── plugin_commands.py# forge plugins list/info
├── pyproject.toml
└── README.md
```

---

## Core Implementation

### CLI Entry Point

```python
# forge/core/cli.py
import click
from forge.core.context import ForgeContext
from forge.plugins.loader import load_plugins

@click.group()
@click.option('--profile', default='default', help='Auth profile to use')
@click.option('--output', '-o', type=click.Choice(['table', 'json', 'yaml']), default='table')
@click.option('--verbose', '-v', is_flag=True)
@click.pass_context
def cli(ctx, profile, output, verbose):
    """Forge — unified internal tools CLI."""
    ctx.ensure_object(dict)
    ctx.obj = ForgeContext(
        profile=profile,
        output_format=output,
        verbose=verbose,
    )

def main():
    load_plugins(cli)  # Discover and attach plugin command groups
    cli()
```

### Context Object

The `ForgeContext` is the central object threaded through every command. It provides lazy access to auth, config, and output formatting — plugins never instantiate these themselves.

```python
# forge/core/context.py
from dataclasses import dataclass, field
from typing import Optional
from forge.auth.manager import AuthManager
from forge.core.config import ConfigStore

@dataclass
class ForgeContext:
    profile: str = 'default'
    output_format: str = 'table'
    verbose: bool = False
    _auth: Optional[AuthManager] = field(default=None, repr=False)
    _config: Optional[ConfigStore] = field(default=None, repr=False)

    @property
    def auth(self) -> AuthManager:
        if self._auth is None:
            self._auth = AuthManager(profile=self.profile)
        return self._auth

    @property
    def config(self) -> ConfigStore:
        if self._config is None:
            self._config = ConfigStore(profile=self.profile)
        return self._config

    def get_token(self) -> str:
        """Get a valid access token, refreshing if needed."""
        return self.auth.get_valid_token()

    def check_permission(self, resource: str, action: str) -> bool:
        """Check if current user has permission for resource:action."""
        return self.auth.check_permission(resource, action)

    def require_permission(self, resource: str, action: str):
        """Raise if current user lacks permission."""
        if not self.check_permission(resource, action):
            raise click.ClickException(
                f"Permission denied: {action} on {resource}. "
                f"Contact your admin to request access."
            )

    def render(self, data, columns=None):
        """Render data in the user's chosen output format."""
        from forge.core.output import render
        render(data, format=self.output_format, columns=columns)
```

### Plugin Discovery and Loading

```python
# forge/plugins/loader.py
import importlib.metadata
import click
import logging

log = logging.getLogger(__name__)

def load_plugins(cli_group: click.Group):
    """Discover and load all installed forge plugins via entry points."""
    entry_points = importlib.metadata.entry_points(group="forge.plugins")

    for ep in entry_points:
        try:
            plugin_cls = ep.load()
            plugin = plugin_cls()

            # Version compatibility check
            # (compare plugin.meta.min_core_version against core version)

            plugin.register(cli_group)
            log.debug(f"Loaded plugin: {plugin.meta.name} v{plugin.meta.version}")
        except Exception as e:
            log.warning(f"Failed to load plugin '{ep.name}': {e}")
            # Don't crash the whole CLI for one bad plugin
```

### Output Formatting

Every command returns structured data. The core formats it based on the `--output` flag. This is non-negotiable for scriptability.

```python
# forge/core/output.py
import json
import click

try:
    import yaml
except ImportError:
    yaml = None

def render(data, format: str = "table", columns: list = None):
    """Render structured data in the requested format."""
    if format == "json":
        click.echo(json.dumps(data, indent=2, default=str))
    elif format == "yaml":
        if yaml is None:
            raise click.ClickException("PyYAML not installed. Use --output json.")
        click.echo(yaml.dump(data, default_flow_style=False))
    elif format == "table":
        _render_table(data, columns)

def _render_table(data, columns=None):
    """Render a list of dicts as an aligned table."""
    if not data:
        click.echo("No results.")
        return

    if isinstance(data, dict):
        data = [data]

    if columns is None:
        columns = list(data[0].keys())

    # Calculate column widths
    widths = {col: len(col) for col in columns}
    for row in data:
        for col in columns:
            widths[col] = max(widths[col], len(str(row.get(col, ""))))

    # Header
    header = "  ".join(col.upper().ljust(widths[col]) for col in columns)
    click.echo(header)
    click.echo("  ".join("─" * widths[col] for col in columns))

    # Rows
    for row in data:
        line = "  ".join(str(row.get(col, "")).ljust(widths[col]) for col in columns)
        click.echo(line)
```

### Error Handling

```python
# forge/core/errors.py
import click
import sys
import traceback

class ForgeError(click.ClickException):
    """Base error for all forge errors."""
    pass

class AuthError(ForgeError):
    """Authentication failed or expired."""
    def format_message(self):
        return f"Auth error: {self.message}\nRun `forge auth login` to authenticate."

class PermissionError(ForgeError):
    """User lacks required permission."""
    def format_message(self):
        return f"Permission denied: {self.message}\nContact your admin to request access."

class PluginError(ForgeError):
    """A plugin encountered an error."""
    pass

def handle_exception(e: Exception, verbose: bool = False):
    """Top-level exception handler."""
    if isinstance(e, click.ClickException):
        e.show()
        sys.exit(e.exit_code)
    elif isinstance(e, KeyboardInterrupt):
        click.echo("\nAborted.")
        sys.exit(130)
    else:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            traceback.print_exc()
        else:
            click.echo("Run with --verbose for full traceback.", err=True)
        sys.exit(1)
```

---

## Configuration Management

Use `~/.forge/` as the home directory. Store config in TOML (human-editable), tokens in a separate directory with restricted permissions.

```
~/.forge/
├── config.toml        # User preferences, default profile
├── profiles/
│   ├── default.toml   # Profile-specific config
│   └── staging.toml
└── credentials/       # 0600 permissions
    ├── default.json   # Tokens for default profile
    └── staging.json
```

```python
# forge/core/config.py
import tomllib
from pathlib import Path
from typing import Any, Optional

FORGE_HOME = Path.home() / ".forge"

class ConfigStore:
    def __init__(self, profile: str = "default"):
        self.profile = profile
        self._global = self._load(FORGE_HOME / "config.toml")
        self._profile = self._load(FORGE_HOME / "profiles" / f"{profile}.toml")

    def get(self, key: str, fallback: Any = None) -> Any:
        """Get config value. Dot-separated keys. Profile overrides global."""
        parts = key.split(".")
        # Check profile config first, then global
        val = self._resolve(self._profile, parts)
        if val is not None:
            return val
        val = self._resolve(self._global, parts)
        return val if val is not None else fallback

    def _load(self, path: Path) -> dict:
        if not path.exists():
            return {}
        with open(path, "rb") as f:
            return tomllib.load(f)

    def _resolve(self, data: dict, parts: list) -> Optional[Any]:
        for part in parts:
            if isinstance(data, dict) and part in data:
                data = data[part]
            else:
                return None
        return data
```

---

## Built-in Commands

### Auth Commands

```python
# forge/builtin/auth_commands.py
import click

@click.group()
def auth():
    """Authentication management."""
    pass

@auth.command()
@click.option('--force', is_flag=True, help='Force re-login even if session is valid')
@click.pass_context
def login(ctx, force):
    """Authenticate with your company identity."""
    ctx.obj.auth.login(force=force)

@auth.command()
@click.pass_context
def logout(ctx):
    """Clear local credentials."""
    ctx.obj.auth.logout()

@auth.command()
@click.pass_context
def status(ctx):
    """Show current authentication status."""
    info = ctx.obj.auth.status()
    if not info["authenticated"]:
        click.echo("Not authenticated. Run `forge auth login`.")
        return
    click.echo(f"User:       {info['user']}")
    click.echo(f"Profile:    {info['profile']}")
    click.echo(f"Groups:     {', '.join(info['groups'])}")
    click.echo(f"Expires in: {info['expires_in']}")

@auth.command()
@click.pass_context
def token(ctx):
    """Print current access token (for debugging/piping)."""
    click.echo(ctx.obj.get_token())
```

### Plugin Management Commands

```python
# forge/builtin/plugin_commands.py
import click
import importlib.metadata

@click.group()
def plugins():
    """Manage installed plugins."""
    pass

@plugins.command(name="list")
def list_plugins():
    """List all installed forge plugins."""
    entry_points = importlib.metadata.entry_points(group="forge.plugins")
    for ep in entry_points:
        try:
            plugin_cls = ep.load()
            plugin = plugin_cls()
            click.echo(f"  {plugin.meta.name:20s} v{plugin.meta.version:10s} {plugin.meta.description}")
        except Exception as e:
            click.echo(f"  {ep.name:20s} (failed to load: {e})")

@plugins.command()
@click.argument("name")
def info(name):
    """Show detailed info about a plugin."""
    entry_points = importlib.metadata.entry_points(group="forge.plugins")
    for ep in entry_points:
        try:
            plugin_cls = ep.load()
            plugin = plugin_cls()
            if plugin.meta.name == name:
                click.echo(f"Name:        {plugin.meta.name}")
                click.echo(f"Version:     {plugin.meta.version}")
                click.echo(f"Description: {plugin.meta.description}")
                click.echo(f"Team:        {plugin.meta.team}")
                click.echo(f"Min Core:    {plugin.meta.min_core_version}")
                return
        except Exception:
            pass
    click.echo(f"Plugin '{name}' not found.")
```

---

## Telemetry

Log command usage for adoption tracking and debugging. Make it opt-out and transparent.

```python
# forge/core/telemetry.py
import time
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

@dataclass
class CommandEvent:
    command: str           # e.g. "data.pipeline.run"
    user: Optional[str]
    duration_ms: int
    exit_code: int
    plugin: Optional[str]
    timestamp: float

class Telemetry:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def track(self, event: CommandEvent):
        if not self.enabled:
            return
        # Send to your internal analytics endpoint
        # Could be CloudWatch, Datadog, a simple HTTP POST, etc.
        log.debug(f"Telemetry: {event.command} by {event.user} took {event.duration_ms}ms")
```

---

## Distribution

Publish the core and SDK as internal PyPI packages. Plugins are also packages.

```bash
# End user installation
pip install forge-cli                  # Core CLI
pip install forge-infra-plugin         # Infra team's plugin
pip install forge-data-plugin          # Data team's plugin

# Or a meta-package that bundles everything
pip install forge-all
```

Consider also providing:

- A `forge plugins install <name>` command that wraps pip for convenience
- Homebrew tap or apt repo for the core binary
- Docker image with all plugins pre-installed for CI/CD

---

## Versioning Strategy

The core and plugins version independently. Plugins declare a minimum core version. The core checks this at load time.

| Package | Versioning | Compatibility |
|---------|-----------|---------------|
| `forge-cli` | semver, core team controls | Breaking changes = major bump |
| `forge-cli-sdk` | semver, strict backward compat | The contract — breaking changes are rare |
| `forge-*-plugin` | semver, each team controls | Declares `min_core_version` |

```
forge-cli-sdk   ← Plugin authors depend on this (slim, stable API)
forge-cli       ← End users install this (full runtime, depends on SDK)
forge-*-plugin  ← Each team's plugin (depends on SDK)
```

---

## Prioritized Build Sequence

1. **Core CLI shell + plugin loader + SDK** — get the skeleton working with one example plugin. Ship to one friendly team and iterate on the SDK contract.
2. **Auth (device flow + file storage)** — get login/logout/status working against your IdP. File permissions are fine for v1.
3. **RBAC (claims-based)** — parse groups from JWTs, ship a policy file.
4. **Output formatting + error handling** — make the UX consistent across plugins.
5. **CI/CD auth (service accounts, FORGE_TOKEN)** — unblocks pipeline adoption.
6. **Telemetry + plugin install command** — polish and operational visibility.

---

## Key Selling Points to Other Teams

> "You write a Click command group and a 10-line plugin class. We handle auth, RBAC, config, distribution, and output formatting. Your users get a consistent experience and you delete your auth code."

What plugin authors get for free:

- Authentication — one decorator (`@require_auth`)
- Authorization — one decorator (`@require_permission(...)`)
- Token access — one method call (`ctx.obj.get_token()`)
- Output formatting — respects `--output` flag automatically
- Config management — profile-aware, plugin-namespaced
- Error handling — consistent, verbose-mode aware
- Distribution — standard Python packaging
- Telemetry — usage tracking without plugin code
