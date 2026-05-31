## Option 1

The CLI pattern is solid, but with a nuance. Having a central CLI that talks to the MCP Gateway is a strong choice because it gives you one control plane for both humans and agents. The key is positioning: the CLI should be a peer interface to the MCP Gateway, not a layer the agents call via subprocess (shelling out to a CLI from an agent is fragile and loses structured I/O). The right pattern is:

Agents connect to the MCP Gateway via MCP protocol (native, structured)
CLI connects to the same MCP Gateway via its API (human-friendly, scriptable)
Both go through the same policy engine, approval routing, and audit trail

This means the CLI lives alongside the agents as another "client" of the gateway — not inside the rules layer. Developers can test any flow manually via CLI before the agent automates it, and CI/CD pipelines can use the CLI too. I'll place it as a peer to the IDE layer, both feeding into the gateway.

<img width="507" height="622" alt="image" src="https://github.com/user-attachments/assets/b086880b-d92d-45d1-ac42-32f1bc1409bd" />


## Pattern A: Agents shell out to CLI (your original idea)

```
Agent → bash: org-cli create-service --name order-svc
                    │
                    ▼
              MCP Gateway → downstream
```

This actually has real strengths. Every agent — Kiro, Copilot, Claude Code, Cursor, Q Developer — can run bash commands today. MCP client support varies and is still maturing across agents. A CLI in the steering rules is self-documenting: the agent reads the .aidlc/ rules, sees the CLI commands, and knows exactly what to call. The downside is that the agent parses text output (fragile) and you lose typed inputs/outputs. Error handling becomes "parse stderr" instead of structured error objects.

## Pattern B: Agents use MCP directly (my earlier suggestion)

```
Agent → MCP protocol → MCP Gateway → downstream
```
Cleaner integration, structured I/O, better error handling. But MCP client configuration differs per IDE, and not every agent handles MCP equally well yet.

## Pattern C: Hybrid — the CLI is the MCP server (best of both)

This is what I'd actually recommend now that I think about it more carefully:

```
Human/CI    → org-cli create-service --name foo     (CLI mode)
Agent       → MCP protocol → org-cli mcp-serve      (MCP server mode)
                    │
                    ▼ (both go through same core)
              Policy engine → downstream
```

You build one binary that runs in two modes. When a human types `org-cli create-service`, it runs as a CLI. When an agent connects, the same binary runs as `org-cli mcp-serve` and speaks MCP protocol. Both modes share the same core logic, policy engine, and audit trail.
This is actually a well-established pattern — the GitHub CLI (`gh`) and AWS CLI are moving this direction. Your steering rules then document both interfaces:

### In .aidlc/org/guardrails/tool-usage.md

#### For agents with MCP support (preferred)
Connect to org-cli MCP server. Tools available:
- create_service, provision_env, register_in_cmdb ...

#### Fallback: CLI commands
If MCP is unavailable, use these CLI commands:
- `org-cli service create --name <name> --template <template>`
- `org-cli env provision --service <name> --env dev`
- `org-cli cmdb register --service <name> --owner <team>`

All commands route through the same gateway and audit trail.

<img width="537" height="670" alt="image" src="https://github.com/user-attachments/assets/9ee80ba2-feba-4fb1-a5e4-02922e2bb599" />


The hybrid wins because you build one thing (`org-cli`) that serves both audiences. Agents with mature MCP support (Kiro, Claude Code) connect via `org-cli mcp-serve`. Agents where MCP is flaky or you haven't configured it yet fall back to `org-cli <command>` via bash. Humans and CI/CD pipelines always use the CLI mode. Same binary, same policy engine, same audit trail — two interfaces.
The key shift in v3 of the diagram: `org-cli` sits between the steering rules and the MCP Gateway. It's not just "another client" — it's the interface layer that both agents and humans go through.

## What is org-cli?
It's a single binary (or Node/Python package) you build that serves as the unified interface to your entire AIDLC platform. It has two entry points but one core engine inside:

<img width="523" height="536" alt="image" src="https://github.com/user-attachments/assets/589573f0-0ea8-4c38-8043-272a7dccbd04" />

The key insight: both modes call the exact same functions inside. The difference is only in how input arrives and how output is returned.

## How CLI mode works (the simple part)
When a human types `org-cli service create --name order-svc --template spring-boot`, the CLI parser converts that into a function call: `serviceHandler.create({ name: "order-svc", template: "spring-boot" })`. The result is printed as formatted text to the terminal. This is standard CLI design — nothing unusual.

## How mcp-serve works (the part you're asking about)
org-cli mcp-serve starts a Model Context Protocol server — a long-running process that speaks MCP's JSON-RPC protocol. AI agents connect to it the same way they connect to any MCP server (like the Jira or GitHub MCPs you're already familiar with).

Here's what happens step by step:

### Step 1: Agent discovers org-cli as an MCP server
The agent's MCP configuration (in the IDE or project config) points to your org-cli. For example in Kiro's .kiro/mcp.json:
```json
{
  "mcpServers": {
    "org-platform": {
      "command": "org-cli",
      "args": ["mcp-serve"],
      "env": {
        "AIDLC_PERSONA": "developer",
        "ORG_GATEWAY_URL": "https://gateway.internal.yourorg.com"
      }
    }
  }
}
```
For Claude Code in `.mcp.json`, for Copilot in its MCP config, for Cursor in its settings — the same binary, same args, different config file locations. The agent starts the process and opens a stdio or SSE connection to it.

### Step 2: Agent asks "what tools do you have?"
When the MCP connection starts, the agent sends an `initialize` request followed by `tools/list`.  `mcp-serve` responds with a tool registry — a list of every operation the agent can call, with typed input schemas. This is the MCP equivalent of `--help`:

```json
{
  "tools": [
    {
      "name": "service_create",
      "description": "Create a new service from a golden path template. Requires tech-lead approval.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "name": { "type": "string", "description": "Service name (kebab-case)" },
          "template": { "type": "string", "enum": ["spring-boot", "node-express", "react-app"] },
          "owner_team": { "type": "string" }
        },
        "required": ["name", "template", "owner_team"]
      }
    },
    {
      "name": "service_list",
      "description": "List services in the catalog. No approval needed.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "team": { "type": "string", "description": "Filter by team (optional)" }
        }
      }
    },
    {
      "name": "jira_search",
      "description": "Search Jira issues. No approval needed.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "jql": { "type": "string", "description": "JQL query" }
        },
        "required": ["jql"]
      }
    },
    {
      "name": "jira_create_issue",
      "description": "Create a Jira issue. Requires tech-lead approval.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "project": { "type": "string" },
          "summary": { "type": "string" },
          "type": { "type": "string", "enum": ["Story", "Task", "Bug"] }
        },
        "required": ["project", "summary", "type"]
      }
    },
    {
      "name": "approval_status",
      "description": "Check status of a pending approval request.",
      "inputSchema": {
        "type": "object",
        "properties": {
          "request_id": { "type": "string" }
        },
        "required": ["request_id"]
      }
    }
  ]
}
```
The agent now knows every operation available, the exact input types, and which ones need approval (from the descriptions). The steering rules in .aidlc/ reinforce this — they tell the agent the same permission matrix in natural language so it understands when to call each tool.

### Step 3: Agent calls a tool
When the agent decides to call `service_create`, it sends:
```json
{
  "method": "tools/call",
  "params": {
    "name": "service_create",
    "arguments": {
      "name": "order-processor",
      "template": "spring-boot",
      "owner_team": "commerce"
    }
  }
}
```
Inside `mcp-serve`, this hits the exact same handler as the CLI command `org-cli service create --name order-processor --template spring-boot --owner commerce`. The shared core runs it through the policy engine, which checks the persona's permissions. Since `service_create` requires approval for the `developer` persona, the core creates an approval request and returns:
```json
{
  "content": [{
    "type": "text",
    "text": "Approval required. Request APR-2026-0042 created.\nApprover: tech-lead\nReview at: https://idp.yourorg.com/approvals/APR-2026-0042\nUse approval_status tool with request_id 'APR-2026-0042' to check progress."
  }]
}
```
The agent reads this structured response, understands the action is pending, and tells the user about the approval link. No text parsing, no fragile regex — the MCP response is typed and predictable.

### Step 4: For ALLOW operations, it just works

When the agent calls jira_search (which is ALLOW for all personas), the core skips the approval flow, calls the downstream jira-mcp or Jira REST API directly, and returns results immediately.

Now here's what the flow looks like when everything connects:
<img width="578" height="476" alt="image" src="https://github.com/user-attachments/assets/e8fadc35-4a53-451b-a147-91f51b38e2d0" />

Code structure of org-cli
Here's what you'd actually build. I'll use TypeScript since MCP has a solid TypeScript SDK, but this works in Python too:

```
org-cli/
├── src/
│   ├── index.ts                  # Entry point: detects mode (mcp-serve vs CLI)
│   │
│   ├── mcp-server/               # MCP SERVER MODE
│   │   ├── server.ts             # MCP server setup (uses @modelcontextprotocol/sdk)
│   │   └── tool-registry.ts      # Converts command handlers → MCP tool definitions
│   │
│   ├── cli/                      # CLI MODE
│   │   ├── parser.ts             # Command parser (uses commander/yargs)
│   │   └── formatter.ts          # Formats results as terminal output (tables, colors)
│   │
│   ├── core/                     # SHARED CORE (both modes call this)
│   │   ├── handlers/
│   │   │   ├── service.ts        # service.create(), service.list(), etc.
│   │   │   ├── jira.ts           # jira.search(), jira.createIssue(), etc.
│   │   │   ├── confluence.ts     # confluence.getPage(), confluence.createPage()
│   │   │   ├── github.ts         # github.createPR(), github.listRepos()
│   │   │   ├── environment.ts    # env.provision(), env.list()
│   │   │   └── approval.ts       # approval.create(), approval.status()
│   │   ├── policy/
│   │   │   ├── engine.ts         # Reads persona + permission matrix, returns ALLOW/APPROVE/DENY
│   │   │   └── permissions.yaml  # The permission matrix (persona × tool → action)
│   │   ├── gateway/
│   │   │   └── client.ts         # HTTP/gRPC client to downstream MCP servers
│   │   └── audit/
│   │       └── logger.ts         # Logs every action to CloudWatch
│   │
│   └── config/
│       └── loader.ts             # Loads persona, gateway URL, auth from env/file
│
├── package.json
└── bin/
    └── org-cli                   # Executable entry point
```

The `index.ts` entry point is dead simple:

```typescript
const command = process.argv[2];

if (command === 'mcp-serve') {
  // Start as MCP server — long-running process
  await startMCPServer();
} else {
  // Run as CLI — execute command and exit
  await runCLI(process.argv.slice(2));
}
```
#### What mcp-serve actually does at the code level
Using the official `@modelcontextprotocol/sdk`, the server setup looks like this:
```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

async function startMCPServer() {
  const server = new McpServer({
    name: "org-platform",
    version: "1.0.0"
  });

  // Register each handler as an MCP tool
  // The SAME handler function used by CLI mode
  server.tool(
    "service_create",
    "Create a new service from a golden path. Requires approval.",
    {
      name: z.string(),
      template: z.enum(["spring-boot", "node-express", "react-app"]),
      owner_team: z.string()
    },
    async ({ name, template, owner_team }) => {
      // This calls the SAME core function as:
      // $ org-cli service create --name X --template Y --owner Z
      const result = await serviceHandler.create({ name, template, owner_team });
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    }
  );

  server.tool(
    "jira_search",
    "Search Jira issues. No approval needed.",
    { jql: z.string() },
    async ({ jql }) => {
      const result = await jiraHandler.search({ jql });
      return { content: [{ type: "text", text: JSON.stringify(result) }] };
    }
  );

  // ... register all other tools

  // Start listening on stdio (agent connects here)
  const transport = new StdioServerTransport();
  await server.connect(transport);
}
```

The `serviceHandler.create()` function is shared — it doesn't know or care whether it was called from MCP or CLI:

```typescript
// core/handlers/service.ts
async function create(params: { name: string; template: string; owner_team: string }) {
  // 1. Policy check
  const persona = config.getPersona();  // "developer", "architect", etc.
  const permission = policyEngine.check(persona, "service_create");

  if (permission === "DENY") {
    return { error: "Not permitted for persona: " + persona };
  }

  if (permission === "APPROVE") {
    // Don't execute — create approval request instead
    const request = await approvalHandler.create({
      action: "service_create",
      payload: params,
      approver_role: "tech-lead",
      justification: `New service ${params.name} using ${params.template} template`
    });
    return {
      status: "approval_required",
      request_id: request.id,
      review_url: request.url,
      message: `Approval needed. Tech lead can review at: ${request.url}`
    };
  }

  // permission === "ALLOW" — execute directly
  // 2. Call downstream IDP to scaffold
  const scaffoldResult = await gatewayClient.call("idp-mcp", "scaffold_service", params);

  // 3. Audit log
  await auditLogger.log({
    persona, action: "service_create",
    params, result: scaffoldResult
  });

  return scaffoldResult;
}
```

#### How agents connect to it in each IDE
The configuration is different per IDE, but the binary is the same:

**Kiro** `(.kiro/mcp.json)`:
```json
{ "mcpServers": { "org": { "command": "org-cli", "args": ["mcp-serve"] } } }
```

**Claude Code** (`.mcp.json`):
```json
{ "mcpServers": { "org": { "command": "org-cli", "args": ["mcp-serve"] } } }
```

**GitHub Copilot** (VS Code `settings.json`):
```json
{ "mcp": { "servers": { "org": { "command": "org-cli", "args": ["mcp-serve"] } } } }
```

Every IDE launches the same `org-cli mcp-serve` process, connects via stdio, and gets the same tool list. The persona comes from the environment variable `AIDLC_PERSONA` which is set per developer or per project.

#### Why this design is strong
Three things make it work well for your use case:

**One codebase = one policy**. When you update the permission matrix, it takes effect for both agents and humans immediately. There's no drift between "what the CLI allows" and "what the MCP allows."

**Agents get structured I/O**. When the agent calls `service_create` via MCP, it gets back a typed JSON response with `status, request_id, review_url`. It doesn't have to parse terminal text like "Approval needed — see https://...". The agent can programmatically check `if result.status === "approval_required"` and present the URL to the user.

**Humans get a familiar interface**. Your developers don't need to learn MCP or think about protocols. They type `org-cli service create --name foo` and get a nice formatted terminal output. Same operation, same audit trail, different interface.

The `tool-registry.ts` file is the bridge — it takes your handler definitions and auto-generates both the MCP tool schemas and the CLI command definitions from a single source of truth. You define each command once, and both interfaces are generated from it.




