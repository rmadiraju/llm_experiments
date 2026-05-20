# Forge CLI — SDK Reference (`forge-cli-sdk`)

> **Package:** `forge-cli-sdk`
> **Purpose:** The slim, stable contract that plugin authors depend on.

---

## Installation

```bash
pip install forge-cli-sdk
```

---

## Full SDK Source

```python
# forge/plugins/sdk.py
"""
Forge Plugin SDK — the contract between the core CLI and plugin authors.

Plugin authors import from this module. Keep it stable — breaking changes
require a major version bump and affect all plugin teams.
"""

import functools
import click
from dataclasses import dataclass
from typing import Optional


# ─── Plugin Metadata ──────────────────────────────────────────────

@dataclass
class PluginMeta:
    """Metadata about a plugin. Required for all plugins."""

    name: str              # Top-level command name (e.g., "data")
    version: str           # Semver string (e.g., "1.2.0")
    description: str       # One-line description for help text
    team: str              # Owning team name for attribution
    min_core_version: str = "1.0.0"  # Minimum forge-cli version required


# ─── Plugin Base Class ────────────────────────────────────────────

class ForgePlugin:
    """Base class for all forge plugins.

    Subclass this and implement register(). The core loader will:
    1. Discover your plugin via entry points
    2. Instantiate it
    3. Call register() to attach your commands
    """

    meta: PluginMeta  # Must be set as a class attribute

    def register(self, group: click.Group):
        """Required: attach your Click command groups to the CLI.

        Args:
            group: The root CLI group. Call group.add_command() to register.
        """
        raise NotImplementedError(
            f"Plugin {self.__class__.__name__} must implement register()"
        )

    def on_activate(self, context):
        """Optional: called once when the plugin loads.

        Use for:
        - Checking that required external dependencies are available
        - Verifying backend API connectivity
        - One-time setup
        """
        pass

    def health_check(self) -> dict:
        """Optional: called by 'forge plugins health'.

        Return a dict with diagnostic information. Example:
            {"status": "ok", "api_reachable": True, "version_match": True}
        """
        return {"status": "ok"}


# ─── Auth Decorators ──────────────────────────────────────────────

def require_auth(f):
    """Decorator: ensures user is authenticated before command runs.

    Usage:
        @my_command.command()
        @require_auth
        @click.pass_context
        def do_thing(ctx):
            token = ctx.obj.get_token()  # Guaranteed valid
            ...

    Behavior:
    - Checks for a valid access token
    - Transparently refreshes if expired (using refresh token)
    - If refresh fails, shows: "Session expired. Run `forge auth login`"
    - Never prompts for login interactively during a command
    """

    @functools.wraps(f)
    @click.pass_context
    def wrapper(ctx, *args, **kwargs):
        forge_ctx = ctx.obj
        forge_ctx.auth.ensure_authenticated()
        return ctx.invoke(f, *args, **kwargs)

    return wrapper


def require_permission(resource: str, action: str):
    """Decorator: checks RBAC before command runs.

    Args:
        resource: "plugin:entity" format (e.g., "data:pipelines")
        action: The action being performed (e.g., "read", "execute", "delete")

    Usage:
        @my_command.command()
        @require_auth
        @require_permission("data:pipelines", "execute")
        @click.pass_context
        def run_pipeline(ctx, name):
            ...  # Only reached if authorized

    Behavior:
    - Evaluates the user's groups against the RBAC policy
    - If denied, shows: "Permission denied: execute on data:pipelines.
      Contact your admin to request access."
    - Always place AFTER @require_auth (auth must happen first)
    """

    def decorator(f):
        @functools.wraps(f)
        @click.pass_context
        def wrapper(ctx, *args, **kwargs):
            ctx.obj.require_permission(resource, action)
            return ctx.invoke(f, *args, **kwargs)

        return wrapper

    return decorator
```

---

## Decorator Ordering

Decorators are applied bottom-up. The correct order is:

```python
@my_group.command()      # 1. Register as a Click command
@require_auth            # 2. Check authentication
@require_permission(...) # 3. Check authorization
@click.option(...)       # 4. Parse options
@click.argument(...)     # 5. Parse arguments
@click.pass_context      # 6. Inject context
def my_command(ctx, ...):
    ...
```

---

## Publishing the SDK

The SDK is published separately from the core CLI to keep plugin dependencies light:

```toml
# forge-cli-sdk/pyproject.toml
[project]
name = "forge-cli-sdk"
version = "1.0.0"
description = "SDK for building Forge CLI plugins"
requires-python = ">=3.10"
dependencies = [
    "click>=8.0,<9.0",
]

# No dependency on forge-cli itself!
# Plugins can develop and test against the SDK alone.
```
