# Forge CLI — Authentication & Token Management Proposal

## Overview

Authentication is the make-or-break component of the centralized CLI. Get it wrong and teams won't adopt. This document covers the full auth design: OAuth flows, token lifecycle, secure storage, RBAC, and CI/CD support.

---

## Auth Flow: OAuth 2.0 Device Authorization Grant (RFC 8628)

The **Device Authorization Grant** is purpose-built for CLIs:

- No need to run a local HTTP server
- Works in SSH sessions
- Works in containers and remote machines
- Your IdP (Okta, Azure AD, AWS IAM Identity Center) almost certainly supports it

### User Experience

```
$ forge auth login

Opening browser for authentication...
If the browser doesn't open, visit: https://auth.yourcompany.com/device
Enter code: ABCD-1234

Waiting for authorization... ✓
Authenticated as raghu@company.com
Token expires in 8 hours. Auto-refresh is enabled.
```

### Flow Diagram

```
┌─────────┐    ┌──────────────┐    ┌─────────────┐
│  Login   │───>│ Device Flow  │───>│ Store Tokens│
│ Command  │    │ (IdP)        │    │ (Encrypted) │
└─────────┘    └──────────────┘    └─────────────┘
                                          │
                                          v
┌─────────┐    ┌──────────────┐    ┌─────────────┐
│  Any     │───>│ Get Token    │───>│ Valid?      │
│ Command  │    │ from Store   │    │             │
└─────────┘    └──────────────┘    └─────────────┘
                                     │yes     │no
                                     v        v
                               ┌────────┐ ┌──────────┐
                               │ Use It │ │ Refresh  │
                               └────────┘ │ Token    │
                                          └──────────┘
                                           │ok    │fail
                                           v      v
                                     ┌────────┐ ┌────────┐
                                     │Use New │ │Re-login│
                                     │ Token  │ │ Prompt │
                                     └────────┘ └────────┘
```

---

## OAuth Device Flow Implementation

```python
# forge/auth/oauth.py
import time
import httpx
from dataclasses import dataclass
from typing import Optional

@dataclass
class DeviceCodeResponse:
    device_code: str
    user_code: str
    verification_uri: str
    verification_uri_complete: Optional[str]
    expires_in: int
    interval: int  # Polling interval in seconds

@dataclass
class TokenResponse:
    access_token: str
    refresh_token: Optional[str]
    expires_in: int
    expires_at: float  # Computed: time.time() + expires_in
    id_token: Optional[str]
    token_type: str = "Bearer"

class OAuthError(Exception):
    pass

class OAuthDeviceFlow:
    def __init__(self, client_id: str, idp_base_url: str):
        self.client_id = client_id
        self.idp_base_url = idp_base_url.rstrip("/")
        self.http = httpx.Client(timeout=30)

    def request_device_code(self, scopes: list[str]) -> DeviceCodeResponse:
        """Step 1: Request a device code from the IdP."""
        resp = self.http.post(
            f"{self.idp_base_url}/oauth2/device/authorize",
            data={
                "client_id": self.client_id,
                "scope": " ".join(scopes),
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return DeviceCodeResponse(
            device_code=data["device_code"],
            user_code=data["user_code"],
            verification_uri=data["verification_uri"],
            verification_uri_complete=data.get("verification_uri_complete"),
            expires_in=data["expires_in"],
            interval=data.get("interval", 5),
        )

    def poll_for_token(
        self,
        device_code: str,
        interval: int = 5,
        timeout: int = 300,
    ) -> TokenResponse:
        """Step 2: Poll the token endpoint until user authorizes."""
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(interval)
            resp = self.http.post(
                f"{self.idp_base_url}/oauth2/token",
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "client_id": self.client_id,
                    "device_code": device_code,
                },
            )
            data = resp.json()

            if resp.status_code == 200:
                return TokenResponse(
                    access_token=data["access_token"],
                    refresh_token=data.get("refresh_token"),
                    expires_in=data["expires_in"],
                    expires_at=time.time() + data["expires_in"],
                    id_token=data.get("id_token"),
                    token_type=data.get("token_type", "Bearer"),
                )
            elif data.get("error") == "authorization_pending":
                continue  # User hasn't authorized yet
            elif data.get("error") == "slow_down":
                interval += 5  # Back off
            elif data.get("error") == "expired_token":
                raise OAuthError("Device code expired. Please try again.")
            elif data.get("error") == "access_denied":
                raise OAuthError("Authorization denied by user.")
            else:
                raise OAuthError(f"OAuth error: {data.get('error_description', data.get('error'))}")

        raise OAuthError("Authorization timed out. Please try again.")

    def refresh(self, refresh_token: str) -> TokenResponse:
        """Refresh an access token using a refresh token."""
        resp = self.http.post(
            f"{self.idp_base_url}/oauth2/token",
            data={
                "grant_type": "refresh_token",
                "client_id": self.client_id,
                "refresh_token": refresh_token,
            },
        )
        if resp.status_code != 200:
            raise OAuthError("Token refresh failed.")
        data = resp.json()
        return TokenResponse(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", refresh_token),
            expires_in=data["expires_in"],
            expires_at=time.time() + data["expires_in"],
            id_token=data.get("id_token"),
            token_type=data.get("token_type", "Bearer"),
        )
```

---

## Auth Manager

The `AuthManager` is the single point of contact for all auth operations. Commands never interact with OAuth or token storage directly.

```python
# forge/auth/manager.py
import os
import time
import webbrowser
import click
from typing import Optional
from forge.auth.token_store import TokenStore, StoredTokens
from forge.auth.oauth import OAuthDeviceFlow, OAuthError

class AuthExpiredError(Exception):
    pass

class AuthManager:
    TOKEN_REFRESH_BUFFER = 300  # Refresh 5 min before expiry

    def __init__(self, profile: str = "default"):
        self.profile = profile
        self.store = TokenStore(profile)
        self.oauth = OAuthDeviceFlow(
            client_id=self._get_client_id(),
            idp_base_url=self._get_idp_url(),
        )
        self._cached_claims = None

    def _get_client_id(self) -> str:
        """Read client ID from config or environment."""
        return os.environ.get(
            "FORGE_OAUTH_CLIENT_ID",
            "your-registered-client-id"  # From config in production
        )

    def _get_idp_url(self) -> str:
        """Read IdP URL from config or environment."""
        return os.environ.get(
            "FORGE_IDP_URL",
            "https://auth.yourcompany.com"  # From config in production
        )

    def ensure_authenticated(self):
        """Ensure we have a valid token, or raise."""
        try:
            self.get_valid_token()
        except AuthExpiredError:
            raise click.ClickException(
                "Session expired. Run `forge auth login` to re-authenticate."
            )

    def get_valid_token(self) -> str:
        """Return a valid access token, refreshing transparently if needed."""

        # Priority 1: Environment variable (CI/CD)
        env_token = os.environ.get("FORGE_TOKEN")
        if env_token:
            claims = self._decode_claims(env_token)
            if claims.get("exp", 0) < time.time():
                raise AuthExpiredError("FORGE_TOKEN is expired.")
            return env_token

        # Priority 2: Service account file (CI/CD)
        sa_path = os.environ.get("FORGE_SERVICE_ACCOUNT")
        if sa_path:
            return self._service_account_flow(sa_path)

        # Priority 3: Stored interactive tokens
        tokens = self.store.load()
        if not tokens:
            raise AuthExpiredError("Not authenticated.")

        # Token still valid (with buffer)?
        if tokens.expires_at > time.time() + self.TOKEN_REFRESH_BUFFER:
            return tokens.access_token

        # Try refresh
        if tokens.refresh_token:
            try:
                new_tokens = self.oauth.refresh(tokens.refresh_token)
                self.store.save(StoredTokens(
                    access_token=new_tokens.access_token,
                    refresh_token=new_tokens.refresh_token,
                    expires_at=new_tokens.expires_at,
                    id_token=new_tokens.id_token,
                    token_type=new_tokens.token_type,
                ))
                return new_tokens.access_token
            except OAuthError:
                pass  # Refresh failed, fall through

        raise AuthExpiredError("Session expired and refresh failed.")

    def login(self, force: bool = False):
        """Interactive login via device code flow."""
        if not force:
            try:
                self.get_valid_token()
                click.echo("Already authenticated. Use --force to re-login.")
                return
            except AuthExpiredError:
                pass

        # Start device flow
        device_resp = self.oauth.request_device_code(
            scopes=["openid", "profile", "email", "offline_access"]
        )

        click.echo(f"\nOpening browser for authentication...")
        click.echo(
            f"If the browser doesn't open, visit: {device_resp.verification_uri}"
        )
        click.echo(f"Enter code: {device_resp.user_code}\n")

        webbrowser.open(
            device_resp.verification_uri_complete or device_resp.verification_uri
        )

        # Poll for completion
        click.echo("Waiting for authorization... ", nl=False)
        token_resp = self.oauth.poll_for_token(
            device_code=device_resp.device_code,
            interval=device_resp.interval,
            timeout=300,
        )

        self.store.save(StoredTokens(
            access_token=token_resp.access_token,
            refresh_token=token_resp.refresh_token,
            expires_at=token_resp.expires_at,
            id_token=token_resp.id_token,
            token_type=token_resp.token_type,
        ))

        claims = self._decode_claims(token_resp.access_token)
        click.echo(f"✓\nAuthenticated as {claims.get('email', 'unknown')}")

    def logout(self):
        """Clear stored tokens."""
        self.store.clear()
        self._cached_claims = None
        click.echo("Logged out. Tokens cleared.")

    def status(self) -> dict:
        """Return current auth status for display."""
        tokens = self.store.load()
        if not tokens:
            return {"authenticated": False}

        claims = self._decode_claims(tokens.access_token)
        remaining = tokens.expires_at - time.time()

        return {
            "authenticated": True,
            "user": claims.get("email"),
            "groups": claims.get("groups", []),
            "expires_in": f"{int(remaining // 60)} minutes",
            "profile": self.profile,
        }

    def get_user_claims(self) -> dict:
        """Decoded JWT claims for RBAC decisions."""
        if self._cached_claims is None:
            token = self.get_valid_token()
            self._cached_claims = self._decode_claims(token)
        return self._cached_claims

    def check_permission(self, resource: str, action: str) -> bool:
        """Check RBAC permission."""
        from forge.auth.rbac import evaluate_permission
        claims = self.get_user_claims()
        return evaluate_permission(claims, resource, action)

    def _decode_claims(self, token: str) -> dict:
        """Decode JWT claims without verification (verification done by IdP)."""
        import base64
        import json
        # JWT is header.payload.signature — we need the payload
        parts = token.split(".")
        if len(parts) != 3:
            return {}
        # Add padding
        payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))

    def _service_account_flow(self, sa_path: str) -> str:
        """Authenticate using a service account key file (for CI/CD)."""
        import json
        with open(sa_path) as f:
            sa = json.load(f)
        # Implementation depends on your IdP:
        # - client_credentials grant with client cert
        # - JWT bearer assertion
        # - AWS STS AssumeRole
        resp = self.oauth.http.post(
            f"{self.oauth.idp_base_url}/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": sa["client_id"],
                "client_secret": sa["client_secret"],
                "scope": "openid profile email",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["access_token"]
```

---

## Secure Token Storage

Tokens in plaintext JSON files are a real risk. The storage backend should use the OS keychain when available and fall back to file-based storage with restricted permissions.

```python
# forge/auth/token_store.py
import os
import json
import stat
import platform
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class StoredTokens:
    access_token: str
    refresh_token: Optional[str]
    expires_at: float   # Unix timestamp
    id_token: Optional[str] = None
    token_type: str = "Bearer"


class BackendUnavailable(Exception):
    pass


class TokenStore:
    """Secure token storage with platform-appropriate backends."""

    def __init__(self, profile: str = "default"):
        self.profile = profile
        self._backend = self._pick_backend()

    def _pick_backend(self):
        """Use OS keychain if available, fall back to encrypted file."""
        system = platform.system()
        try:
            if system == "Darwin":
                return KeychainBackend(self.profile)
            elif system == "Linux":
                return SecretServiceBackend(self.profile)
            elif system == "Windows":
                return WinCredBackend(self.profile)
        except (BackendUnavailable, ImportError):
            pass
        return EncryptedFileBackend(self.profile)

    def save(self, tokens: StoredTokens):
        self._backend.write(json.dumps(asdict(tokens)))

    def load(self) -> Optional[StoredTokens]:
        data = self._backend.read()
        if data is None:
            return None
        return StoredTokens(**json.loads(data))

    def clear(self):
        self._backend.delete()


# ─── Backend: macOS Keychain ──────────────────────────────────────

class KeychainBackend:
    SERVICE = "forge-cli"

    def __init__(self, profile: str):
        try:
            import keyring
            self.keyring = keyring
        except ImportError:
            raise BackendUnavailable("keyring package not installed")
        self.account = f"forge-{profile}"

    def write(self, data: str):
        self.keyring.set_password(self.SERVICE, self.account, data)

    def read(self) -> Optional[str]:
        return self.keyring.get_password(self.SERVICE, self.account)

    def delete(self):
        try:
            self.keyring.delete_password(self.SERVICE, self.account)
        except self.keyring.errors.PasswordDeleteError:
            pass


# ─── Backend: Linux Secret Service (GNOME Keyring / KWallet) ─────

class SecretServiceBackend:
    """Uses the same keyring library, which auto-detects GNOME Keyring or KWallet."""

    SERVICE = "forge-cli"

    def __init__(self, profile: str):
        try:
            import keyring
            # Test that a real backend is available (not the null backend)
            backend = keyring.get_keyring()
            if "fail" in type(backend).__name__.lower():
                raise BackendUnavailable("No keyring backend available")
            self.keyring = keyring
        except ImportError:
            raise BackendUnavailable("keyring package not installed")
        self.account = f"forge-{profile}"

    def write(self, data: str):
        self.keyring.set_password(self.SERVICE, self.account, data)

    def read(self) -> Optional[str]:
        return self.keyring.get_password(self.SERVICE, self.account)

    def delete(self):
        try:
            self.keyring.delete_password(self.SERVICE, self.account)
        except self.keyring.errors.PasswordDeleteError:
            pass


# ─── Backend: Windows Credential Locker ───────────────────────────

class WinCredBackend:
    SERVICE = "forge-cli"

    def __init__(self, profile: str):
        try:
            import keyring
            self.keyring = keyring
        except ImportError:
            raise BackendUnavailable("keyring package not installed")
        self.account = f"forge-{profile}"

    def write(self, data: str):
        self.keyring.set_password(self.SERVICE, self.account, data)

    def read(self) -> Optional[str]:
        return self.keyring.get_password(self.SERVICE, self.account)

    def delete(self):
        try:
            self.keyring.delete_password(self.SERVICE, self.account)
        except self.keyring.errors.PasswordDeleteError:
            pass


# ─── Backend: Encrypted File (Fallback) ──────────────────────────

class EncryptedFileBackend:
    """Fallback: file-based storage with restrictive permissions.

    For production, consider layering DPAPI (Windows) or libsecret
    encryption on top. At minimum, restrict file permissions to owner.
    """

    def __init__(self, profile: str):
        self.path = Path.home() / ".forge" / "credentials" / f"{profile}.json"

    def write(self, data: str):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(data)
        # Restrict to owner only (0600)
        os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)

    def read(self) -> Optional[str]:
        if not self.path.exists():
            return None
        return self.path.read_text()

    def delete(self):
        if self.path.exists():
            self.path.unlink()
```

---

## RBAC (Role-Based Access Control)

### Approach: Claims-Based (Recommended for v1)

Your IdP puts group memberships in the JWT. The CLI maps groups to permissions locally using a policy file. This is simple, fast (no network calls for auth checks), and sufficient for most internal tooling.

### Permission Model

Permissions follow the pattern: `plugin:entity:action`

```
infra:stacks:deploy      → deploy infrastructure stacks
data:pipelines:execute   → trigger a pipeline run
data:pipelines:read      → list/view pipelines
security:scans:write     → create security scans
*:*:read                 → read access to everything
```

### Policy Definition

```python
# forge/auth/rbac.py
from typing import Dict, List

# This can be:
# 1. Bundled as a config file shipped with forge-cli
# 2. Fetched from a central config service at startup
# 3. Stored in ~/.forge/policy.toml and synced periodically
#
# Structure: role/group -> list of "resource:action" permissions

DEFAULT_POLICY: Dict[str, List[str]] = {
    "platform-eng": [
        "infra:stacks:*",          # All actions on infra stacks
        "infra:clusters:read",
        "data:pipelines:read",
    ],
    "data-eng": [
        "data:pipelines:*",        # Full pipeline access
        "data:warehouses:*",       # Full warehouse access
        "infra:stacks:read",       # Read-only infra
    ],
    "security": [
        "security:*:*",            # Full access to security plugin
        "*:*:read",                # Read access to everything
    ],
    "oncall": [
        "infra:stacks:deploy",     # Can deploy during incidents
        "infra:clusters:restart",
        "data:pipelines:execute",
    ],
    "admin": [
        "*:*:*",                   # Superuser
    ],
    "viewer": [
        "*:*:read",                # Read-only across everything
    ],
}


def evaluate_permission(claims: dict, resource: str, action: str) -> bool:
    """
    Check if user's groups grant permission for resource:action.

    Args:
        claims: Decoded JWT claims (must include 'groups' list)
        resource: "plugin:entity" (e.g., "infra:stacks")
        action: "read", "write", "deploy", "delete", "execute", etc.
    """
    user_groups = claims.get("groups", [])

    for group in user_groups:
        allowed = DEFAULT_POLICY.get(group, [])
        for permission in allowed:
            if _matches(permission, f"{resource}:{action}"):
                return True
    return False


def _matches(pattern: str, target: str) -> bool:
    """Wildcard matching: 'infra:*:*' matches 'infra:stacks:deploy'."""
    pattern_parts = pattern.split(":")
    target_parts = target.split(":")

    # Pad shorter list (handles 2-part vs 3-part patterns)
    while len(pattern_parts) < len(target_parts):
        pattern_parts.append("*")
    while len(target_parts) < len(pattern_parts):
        target_parts.append("*")

    for p, t in zip(pattern_parts, target_parts):
        if p == "*":
            continue
        if p != t:
            return False
    return True


def list_user_permissions(claims: dict) -> list[str]:
    """List all permissions granted to a user (for debugging/display)."""
    user_groups = claims.get("groups", [])
    permissions = set()
    for group in user_groups:
        for perm in DEFAULT_POLICY.get(group, []):
            permissions.add(perm)
    return sorted(permissions)
```

### How Plugins Declare Permissions

Plugin authors use the `@require_permission` decorator. They don't define permissions in a registry — they just use them, and you (the platform team) map groups to those permissions in the policy.

```python
# In a plugin command:
@infra.command()
@require_auth
@require_permission("infra:stacks", "deploy")
@click.pass_context
def deploy(ctx, stack, env):
    """Deploy an infrastructure stack."""
    # If we reach here, user is authenticated AND authorized
    ...
```

### Upgrading to Server-Side RBAC (v2)

When claims-based outgrows your needs (fine-grained resource-level checks, real-time permission changes, audit logging), move to a centralized authorization service:

```python
# forge/auth/rbac_server.py (v2 — replaces local evaluation)
import httpx

AUTHZ_SERVICE_URL = "https://authz.internal.yourcompany.com"

def evaluate_permission(claims: dict, resource: str, action: str) -> bool:
    """Server-side permission check."""
    resp = httpx.post(
        f"{AUTHZ_SERVICE_URL}/v1/check",
        json={
            "subject": claims.get("sub"),
            "resource": resource,
            "action": action,
        },
        headers={"Authorization": f"Bearer {claims.get('raw_token')}"},
        timeout=2,
    )
    if resp.status_code == 200:
        return resp.json().get("allowed", False)
    # Fail closed on errors
    return False
```

---

## CI/CD and Non-Interactive Authentication

Teams will use forge in pipelines. Support three methods, in priority order:

### Method 1: Environment Variable Token

Simplest approach. Inject a pre-obtained token as an env var.

```yaml
# GitHub Actions example
jobs:
  deploy:
    steps:
      - name: Deploy via Forge
        env:
          FORGE_TOKEN: ${{ secrets.FORGE_DEPLOY_TOKEN }}
        run: forge infra deploy my-stack --env staging
```

### Method 2: Service Account Key File

For long-running automation that needs its own identity.

```yaml
# GitHub Actions example
jobs:
  deploy:
    steps:
      - name: Deploy via Forge
        env:
          FORGE_SERVICE_ACCOUNT: /path/to/sa-key.json
        run: forge infra deploy my-stack --env staging
```

Service account key file format:

```json
{
  "type": "service_account",
  "client_id": "forge-ci-deploy",
  "client_secret": "...",
  "description": "CI/CD deploy service account for platform-eng"
}
```

### Method 3: AWS IAM Role (for AWS-native environments)

If running in AWS (ECS, Lambda, CodeBuild), use IAM role assumption to get forge tokens:

```python
# forge/auth/aws.py
import boto3

def get_token_via_iam(role_arn: str, idp_url: str) -> str:
    """Exchange AWS IAM credentials for a forge access token."""
    sts = boto3.client("sts")
    # Get AWS credentials proof
    identity = sts.get_caller_identity()

    # Exchange with your token service that validates AWS identity
    resp = httpx.post(
        f"{idp_url}/oauth2/token",
        data={
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "subject_token": identity["Arn"],
            "subject_token_type": "urn:aws:iam",
        },
    )
    return resp.json()["access_token"]
```

---

## Multi-Profile Support

Users working across environments (dev, staging, prod) need separate auth profiles:

```bash
# Login to different profiles
forge auth login --profile staging
forge auth login --profile prod

# Use a profile for a command
forge --profile staging infra deploy my-stack --env staging

# Set default profile
forge config set default_profile staging

# Check which profile is active
forge auth status
forge auth status --profile prod
```

Each profile gets its own token storage, config overrides, and IdP endpoint.

---

## Token Security Checklist

| Concern | Mitigation |
|---------|-----------|
| Token in plaintext on disk | OS keychain preferred; file fallback uses `0600` permissions |
| Token leaked in logs | Auth manager never logs token values; only logs metadata |
| Token in shell history | `forge auth token` pipes to stdout, not logged in history if piped |
| Token in environment | `FORGE_TOKEN` is standard pattern; document risk in CI/CD guide |
| Token lifetime too long | Configure IdP for 8-hour access tokens, 30-day refresh tokens |
| Stale refresh token | Refresh tokens are rotated on each use (IdP-side config) |
| Token used after logout | `forge auth logout` clears all stored tokens immediately |
| Man-in-the-middle | All IdP communication over HTTPS; pin certificates if paranoid |
| Shared workstation | Profiles isolate credentials; `forge auth logout --all` clears everything |

---

## Auth Configuration Reference

```toml
# ~/.forge/config.toml

[auth]
# IdP configuration
idp_url = "https://auth.yourcompany.com"
client_id = "forge-cli-public"

# Token behavior
auto_refresh = true          # Transparently refresh tokens
refresh_buffer_seconds = 300 # Refresh 5 min before expiry

# Default scopes requested during login
scopes = ["openid", "profile", "email", "offline_access"]

[auth.storage]
# "auto" (detect OS keychain), "keychain", "file"
backend = "auto"

[auth.rbac]
# "local" (claims-based) or "server" (centralized service)
mode = "local"
# If mode = "server":
# authz_url = "https://authz.internal.yourcompany.com"
# Policy file location (for local mode)
policy_path = ""  # Empty = use bundled default policy
```

---

## Summary: Auth Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Auth protocol | OAuth 2.0 Device Authorization Grant | Built for CLI, works everywhere |
| Token format | JWT (from IdP) | Self-contained claims, no lookup needed for RBAC |
| Token storage | OS keychain with file fallback | Best security without external dependencies |
| RBAC v1 | Claims-based (local policy) | Fast, no service dependency, good enough for internal tooling |
| RBAC v2 | Server-side authorization service | When fine-grained resource-level checks are needed |
| CI/CD auth | FORGE_TOKEN env var + service account files | Standard patterns, easy to integrate |
| Multi-account | Profile-based | Separate credentials per environment |
| Token lifecycle | 8h access / 30d refresh / auto-refresh | Balance security and convenience |
