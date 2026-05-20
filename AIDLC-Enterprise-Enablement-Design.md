# Enterprise AI-DLC Enablement: High-Level Design & Implementation Plan

## Executive Summary

This document outlines the architecture, tools, MCP servers, and implementation plan to bring the AI-Driven Development Lifecycle (AI-DLC) into your organization with four pillars: **persona-based SOPs**, **agent guardrails with approval workflows**, **IDP onboarding skills**, and **PDLC tool integration** — all with full audit and control, and IDE/agent agnostic.

The design uses the open-source AWS AI-DLC steering files as a starting point and extends them with organization-specific layers. The entire system is built on portable markdown rules + MCP servers, meaning the same configuration works across Kiro, GitHub Copilot, Claude Code, Cursor, Amazon Q Developer, and any other agent that supports project-level rules.

---

## 1. Architecture Overview

### 1.1 Layered Rule Architecture (IDE-Agnostic)

The core idea is a **three-tier rule system** that stacks on top of the standard AI-DLC workflow:

```
┌─────────────────────────────────────────────────┐
│  TIER 3: Project-Specific Rules                 │
│  (per-repo overrides, feature flags, tech stack)│
├─────────────────────────────────────────────────┤
│  TIER 2: Org Standards & Guardrails             │
│  (SOPs, persona rules, approval policies,       │
│   IDP skills, PDLC integrations)                │
├─────────────────────────────────────────────────┤
│  TIER 1: AI-DLC Core Workflow                   │
│  (aws-aidlc-rules from awslabs/aidlc-workflows) │
└─────────────────────────────────────────────────┘
```

### 1.2 Portable File Structure

All rules live in the repository. Each IDE maps them to its native rules location via symlinks or copy scripts:

```
<project-root>/
├── .aidlc/                              # CANONICAL source of truth
│   ├── core/                            # Tier 1: AI-DLC core (from awslabs)
│   │   ├── aws-aidlc-rules/
│   │   └── aws-aidlc-rule-details/
│   ├── org/                             # Tier 2: Your organization layer
│   │   ├── personas/
│   │   │   ├── architect.md
│   │   │   ├── developer.md
│   │   │   ├── tech-lead.md
│   │   │   ├── sre.md
│   │   │   └── qa.md
│   │   ├── guardrails/
│   │   │   ├── tool-permissions.md
│   │   │   ├── approval-workflow.md
│   │   │   └── data-classification.md
│   │   ├── standards/
│   │   │   ├── architecture-standards.md
│   │   │   ├── coding-standards.md
│   │   │   ├── security-baseline.md
│   │   │   ├── api-design-standards.md
│   │   │   └── infra-patterns.md
│   │   └── idp/
│   │       ├── service-onboarding.md
│   │       ├── golden-paths.md
│   │       └── environment-provisioning.md
│   ├── project/                         # Tier 3: Project overrides
│   │   └── project-context.md
│   └── sync.sh                          # Syncs .aidlc/ → IDE-specific dirs
├── .kiro/steering/                      # Kiro (auto-synced)
├── .amazonq/rules/                      # Amazon Q (auto-synced)
├── .github/copilot-instructions.md      # GitHub Copilot (auto-synced)
├── CLAUDE.md                            # Claude Code (auto-synced)
├── .cursor/rules/                       # Cursor (auto-synced)
└── .clinerules/                         # Cline (auto-synced)
```

### 1.3 Sync Script (IDE Agnostic Distribution)

A simple shell script (`sync.sh`) composes the three tiers into a single merged rule file and distributes it to each IDE's native location. This runs as a git hook (post-checkout, post-merge) or manually:

```bash
#!/bin/bash
# .aidlc/sync.sh — Compose & distribute rules to all IDE targets
MERGED=$(mktemp)
cat .aidlc/core/aws-aidlc-rules/core-workflow.md >> "$MERGED"
echo -e "\n---\n" >> "$MERGED"
# Append active persona (from .aidlc/active-persona or env var)
PERSONA=${AIDLC_PERSONA:-developer}
cat ".aidlc/org/personas/${PERSONA}.md" >> "$MERGED"
echo -e "\n---\n" >> "$MERGED"
# Append guardrails, standards, IDP rules
for f in .aidlc/org/guardrails/*.md .aidlc/org/standards/*.md .aidlc/org/idp/*.md; do
  [ -f "$f" ] && cat "$f" >> "$MERGED" && echo -e "\n---\n" >> "$MERGED"
done
# Append project overrides
[ -f .aidlc/project/project-context.md ] && cat .aidlc/project/project-context.md >> "$MERGED"

# Distribute to IDE-specific locations
mkdir -p .kiro/steering/aws-aidlc-rules && cp "$MERGED" .kiro/steering/aws-aidlc-rules/core-workflow.md
mkdir -p .amazonq/rules/aws-aidlc-rules && cp "$MERGED" .amazonq/rules/aws-aidlc-rules/core-workflow.md
cp "$MERGED" CLAUDE.md
mkdir -p .github && cp "$MERGED" .github/copilot-instructions.md
mkdir -p .cursor/rules && cp "$MERGED" .cursor/rules/ai-dlc-workflow.mdc
mkdir -p .clinerules && cp "$MERGED" .clinerules/core-workflow.md
# Copy rule details
for target in .kiro .amazonq; do
  cp -R .aidlc/core/aws-aidlc-rule-details "$target/"
done
cp -R .aidlc/core/aws-aidlc-rule-details .aidlc-rule-details
rm "$MERGED"
echo "✓ Rules synced for persona: $PERSONA"
```

---

## 2. Goal 1: Persona-Based SOPs

### 2.1 Persona Rule Files

Each persona file is a steering/rules markdown that instructs the agent on which standards to enforce. The agent reads the active persona file and adjusts its behavior accordingly.

**Example: `.aidlc/org/personas/architect.md`**

```markdown
# Persona: Solution Architect

## Active Standards
You MUST enforce and reference these standards in all outputs:
- Architecture Standards (.aidlc/org/standards/architecture-standards.md)
- Security Baseline (.aidlc/org/standards/security-baseline.md)
- API Design Standards (.aidlc/org/standards/api-design-standards.md)

## Behavioral Rules
- During AI-DLC Inception phase, produce architecture decision records (ADRs)
- All designs must reference the approved technology radar
- Validate against the org's reference architectures before proposing new patterns
- Use the CMDB MCP to check existing services before proposing new ones
- Produce C4 diagrams (context, container, component) for any new system

## Approval Authority
- Can approve: Design documents, ADRs, tech stack selections
- Cannot approve: Production deployments, budget allocations, data schema changes

## Tool Permissions
- READ: Jira, Confluence, GitHub, CMDB, Service Catalog, Cost Explorer
- WRITE (direct): Confluence (design docs), GitHub (branches, PRs)
- WRITE (approval required): Jira (epic creation), CMDB (new service records)
```

**Example: `.aidlc/org/personas/developer.md`**

```markdown
# Persona: Developer

## Active Standards
- Coding Standards (.aidlc/org/standards/coding-standards.md)
- Security Baseline (.aidlc/org/standards/security-baseline.md)

## Behavioral Rules
- During AI-DLC Construction phase, follow the building blocks library
- All code must pass linting rules defined in the project
- Follow the branching strategy: feature/* → develop → main
- Use approved libraries only (check org tech radar via MCP)
- Write unit tests for all business logic (min 80% coverage target)

## Tool Permissions
- READ: Jira (assigned tickets), Confluence, GitHub, Service Catalog
- WRITE (direct): GitHub (commits, PRs on feature branches)
- WRITE (approval required): Jira (status transitions beyond "In Progress"),
  CMDB (any write), infrastructure provisioning
```

### 2.2 Persona Selection Mechanism

Persona is selected via:
1. **Environment variable**: `AIDLC_PERSONA=architect` (CI/CD or shell)
2. **File**: `.aidlc/active-persona` containing the persona name
3. **Git config**: `git config aidlc.persona architect`
4. **Default**: Falls back to `developer`

### 2.3 Standards Files

Each standards file is a self-contained markdown the agent treats as authoritative policy:

```
.aidlc/org/standards/
├── architecture-standards.md    # Reference architectures, patterns, anti-patterns
├── coding-standards.md          # Language-specific rules, linting, naming conventions
├── security-baseline.md         # OWASP top 10, auth patterns, secret management
├── api-design-standards.md      # REST conventions, versioning, pagination, error formats
└── infra-patterns.md            # Approved cloud patterns, IaC conventions, tagging
```

These are version-controlled in a central `org-standards` repo and pulled into projects as a git submodule or via the sync script from an artifact registry.

---

## 3. Goal 2: Agent Guardrails & Approval Workflows

### 3.1 Guardrail Architecture

The guardrail system works at two levels:

**Level 1 — Rule-Based (In-Context Steering)**
The `tool-permissions.md` file tells the agent what it can and cannot do directly. This is the "first line of defense" — the agent's own instructions.

**Level 2 — MCP Gateway (Runtime Enforcement)**
An MCP Gateway proxy sits between the agent and downstream MCP servers. It enforces permissions at the protocol level, regardless of what the agent tries to do.

```
┌──────────┐    ┌──────────────────┐    ┌────────────────┐
│  Agent    │───▶│  MCP Gateway     │───▶│  Downstream    │
│ (any IDE) │    │  (Policy Engine) │    │  MCP Servers   │
└──────────┘    │                  │    │  (Jira, CMDB,  │
                │  • Auth/AuthZ    │    │   GitHub, IDP) │
                │  • Tool-level    │    └────────────────┘
                │    permissions   │
                │  • Approval      │    ┌────────────────┐
                │    routing       │───▶│  Audit Log     │
                │  • Audit logging │    │  (S3/ELK/      │
                └──────────────────┘    │   CloudWatch)  │
                         │              └────────────────┘
                         ▼
                ┌──────────────────┐
                │  Approval        │
                │  Workflow Engine  │
                │  (ServiceNow /   │
                │   custom / Slack) │
                └──────────────────┘
```

### 3.2 Tool Permission Matrix

Define in `.aidlc/org/guardrails/tool-permissions.md`:

```markdown
# Tool Permission Matrix

## Permission Levels
- ALLOW: Agent can call directly, logged for audit
- APPROVE: Agent generates a request, human approves before execution
- DENY: Agent cannot call, even with approval

## Matrix

| MCP Server     | Tool/Operation              | Architect | Developer | Tech Lead | SRE   |
|----------------|-----------------------------|-----------|-----------|-----------|-------|
| jira-mcp       | search_issues (GET)         | ALLOW     | ALLOW     | ALLOW     | ALLOW |
| jira-mcp       | get_issue (GET)             | ALLOW     | ALLOW     | ALLOW     | ALLOW |
| jira-mcp       | create_issue (POST)         | APPROVE   | APPROVE   | ALLOW     | APPROVE|
| jira-mcp       | transition_issue (PUT)      | APPROVE   | APPROVE   | ALLOW     | APPROVE|
| github-mcp     | list_repos (GET)            | ALLOW     | ALLOW     | ALLOW     | ALLOW |
| github-mcp     | create_pr (POST)            | ALLOW     | ALLOW     | ALLOW     | ALLOW |
| github-mcp     | merge_pr (POST)             | DENY      | DENY      | APPROVE   | DENY  |
| cmdb-mcp       | search_services (GET)       | ALLOW     | ALLOW     | ALLOW     | ALLOW |
| cmdb-mcp       | create_service (POST)       | APPROVE   | APPROVE   | APPROVE   | APPROVE|
| cmdb-mcp       | update_service (PUT)        | APPROVE   | DENY      | APPROVE   | APPROVE|
| confluence-mcp | get_page (GET)              | ALLOW     | ALLOW     | ALLOW     | ALLOW |
| confluence-mcp | create_page (POST)          | ALLOW     | APPROVE   | ALLOW     | APPROVE|
| idp-mcp        | list_templates (GET)        | ALLOW     | ALLOW     | ALLOW     | ALLOW |
| idp-mcp        | scaffold_service (POST)     | APPROVE   | APPROVE   | APPROVE   | APPROVE|
| idp-mcp        | provision_env (POST)        | DENY      | DENY      | APPROVE   | ALLOW |
```

### 3.3 Approval Workflow Design

When the agent hits an "APPROVE" action, it follows this flow:

```markdown
# Approval Workflow Rule

When you need to perform an operation marked APPROVE in the tool permission matrix:

1. DO NOT call the target MCP tool directly
2. Instead, call the `approval-mcp` server's `create_approval_request` tool with:
   - `action`: The tool and operation you want to perform
   - `target_mcp`: The downstream MCP server name
   - `payload`: The exact parameters you would have sent
   - `justification`: Why this action is needed (reference the AI-DLC stage)
   - `persona`: Your active persona
   - `approver_role`: Who should approve (from permission matrix)
3. The approval-mcp returns a request URL and ID
4. Present the URL to the user: "I've created approval request [ID].
   [Approver] can review it at: [URL]"
5. Do NOT proceed with the action until the approval is granted
6. You can check status via `approval-mcp.get_request_status(id)`
```

### 3.4 MCP Servers Needed for Guardrails

| MCP Server | Purpose | Build vs Buy |
|---|---|---|
| **approval-mcp** | Creates approval requests, routes to approvers, tracks status | **Build** (custom) |
| **mcp-gateway** | Proxy that enforces tool-level permissions before routing to downstream MCPs | **Build** (use Microsoft AGT or TrueFoundry as starting point) |
| **audit-mcp** | Logs every tool call, decision, and approval event | **Build** (wraps your SIEM/log platform) |
| **policy-mcp** | Serves the permission matrix and persona rules to the gateway | **Build** (simple config server) |

### 3.5 Approval MCP Server Spec

```yaml
name: approval-mcp
tools:
  - name: create_approval_request
    description: "Create an approval request for a gated action"
    input:
      action: string          # e.g., "cmdb-mcp.create_service"
      payload: object         # The parameters for the downstream tool
      justification: string   # AI-DLC context (which stage, why)
      approver_role: string   # "tech-lead", "architect", "sre"
      urgency: enum           # "low", "medium", "high"
    output:
      request_id: string
      request_url: string     # URL to approval UI or Slack deeplink
      status: "pending"

  - name: get_request_status
    description: "Check the status of an existing approval request"
    input:
      request_id: string
    output:
      status: enum            # "pending", "approved", "rejected", "expired"
      approved_by: string?
      approved_at: datetime?
      comments: string?

  - name: execute_approved_action
    description: "Execute a previously approved action"
    input:
      request_id: string
    output:
      result: object          # Response from downstream MCP
```

### 3.6 Approval Routing Backends

The approval-mcp can route to multiple backends depending on org preference:

- **Slack**: Posts an interactive message to a channel; approver clicks Approve/Reject
- **ServiceNow**: Creates a change request or approval record
- **Custom Web UI**: A lightweight approval dashboard
- **Jira**: Creates an approval ticket in a designated project
- **Email**: Sends approval link with one-click approve/reject

---

## 4. Goal 3: IDP Onboarding Skills

### 4.1 IDP Integration Architecture

The agent uses IDP (Internal Developer Platform) skills to scaffold and onboard services. These skills are controlled through the same guardrail system.

```
Agent (with IDP steering rules)
  │
  ├──▶ idp-mcp (scaffold, provision, register)
  │      │
  │      ├──▶ Backstage / Port / Harness IDP
  │      │     • Software Templates (golden paths)
  │      │     • Service Catalog registration
  │      │     • TechDocs generation
  │      │
  │      ├──▶ IaC Engine (Terraform / Crossplane / Pulumi)
  │      │     • Infrastructure provisioning
  │      │     • Environment creation
  │      │
  │      └──▶ CI/CD Pipeline (GitHub Actions / Jenkins / CodePipeline)
  │            • Pipeline scaffolding
  │            • Build/deploy configuration
  │
  └──▶ catalog-mcp (read-only service catalog queries)
         • "What services exist in domain X?"
         • "Who owns service Y?"
         • "What's the API spec for Z?"
```

### 4.2 IDP MCP Server Spec

```yaml
name: idp-mcp
tools:
  # READ operations (ALLOW for all personas)
  - name: list_golden_paths
    description: "List available service templates / golden paths"
    output: array of { name, description, tech_stack, owner }

  - name: get_golden_path_details
    description: "Get full details of a golden path template"
    input: { template_id: string }
    output: { parameters, steps, infrastructure, pipeline_config }

  - name: list_environments
    description: "List available environments for a service"
    input: { service_name: string }
    output: array of { env_name, status, url, last_deployed }

  # WRITE operations (APPROVE for all personas)
  - name: scaffold_service
    description: "Create a new service from a golden path template"
    input:
      template_id: string
      service_name: string
      owner_team: string
      parameters: object       # Template-specific params
    output:
      repo_url: string
      catalog_entry_url: string
      pipeline_url: string
      status: string

  - name: provision_environment
    description: "Provision infrastructure for a service environment"
    input:
      service_name: string
      environment: string      # dev, staging, prod
      config_overrides: object
    output:
      infra_status: string
      endpoints: object
      cost_estimate: object

  - name: register_in_catalog
    description: "Register or update service in the software catalog"
    input:
      service_name: string
      metadata: object         # owner, lifecycle, links, dependencies
    output:
      catalog_url: string
```

### 4.3 IDP Steering Rules

**`.aidlc/org/idp/service-onboarding.md`**

```markdown
# IDP Service Onboarding Rules

## Golden Path Enforcement
When creating a new service during AI-DLC Construction phase:
1. ALWAYS check available golden paths via `idp-mcp.list_golden_paths` first
2. If a matching golden path exists, USE IT — do not create from scratch
3. If no matching path exists, flag this and request architect approval
4. All new services MUST be registered in the software catalog

## Onboarding Sequence
For any new service, follow this exact sequence:
1. Verify the design is approved (check AI-DLC Inception artifacts)
2. Select golden path template
3. Create approval request for scaffolding (approval-mcp)
4. Once approved, scaffold via idp-mcp.scaffold_service
5. Register in CMDB via cmdb-mcp (approval required)
6. Create Jira epic for the new service (approval required)
7. Provision dev environment (approval required)
8. Generate TechDocs stub in Confluence

## Environment Rules
- dev: Auto-provisioned on service creation (tech-lead approval)
- staging: Provisioned on first PR merge to develop (SRE approval)
- prod: Provisioned only via change management (CAB approval)
```

### 4.4 Golden Path Templates

Golden paths encode organizational standards into repeatable templates:

```
golden-paths/
├── microservice-java-spring/
│   ├── template.yaml          # Backstage template definition
│   ├── skeleton/              # Cookiecutter/scaffolding files
│   ├── infra/                 # Terraform modules
│   ├── pipeline/              # CI/CD pipeline definition
│   └── docs/                  # TechDocs template
├── microservice-node-express/
├── api-gateway-pattern/
├── event-driven-lambda/
├── frontend-react-app/
└── data-pipeline-spark/
```

---

## 5. Goal 4: PDLC Tool Integration

### 5.1 MCP Servers for PDLC Tools

| Tool | MCP Server | Status | Key Operations |
|---|---|---|---|
| **Jira** | `jira-mcp` (Atlassian official or community) | Available | Search, create, transition, comment, link issues |
| **Confluence** | `confluence-mcp` (Atlassian official or community) | Available | Search, read, create, update pages |
| **GitHub** | `github-mcp` (official) | Available | Repos, PRs, issues, actions, code search |
| **IDP CLI** | `idp-mcp` (custom) | **Build** | Wraps your IDP CLI (Backstage CLI, custom CLI) |
| **CMDB** | `cmdb-mcp` (custom) | **Build** | ServiceNow CMDB or custom service registry |
| **CI/CD** | `cicd-mcp` (custom) | **Build** | Trigger builds, check pipeline status, get logs |
| **Tech Radar** | `techradar-mcp` (custom) | **Build** | Query approved/hold/trial technologies |
| **Cost Explorer** | `cost-mcp` (custom) | **Build** | Query estimated/actual costs for services |

### 5.2 Integration Flow: Design → Code → Deploy

The PDLC integration follows the AI-DLC phases:

**Inception Phase:**
```
Agent reads requirements from Jira (jira-mcp.get_issue)
  → Checks existing services in CMDB (cmdb-mcp.search)
  → Checks tech radar for approved tech (techradar-mcp.query)
  → Produces design doc → pushes to Confluence (confluence-mcp.create_page) [APPROVE]
  → Creates ADR → pushes to GitHub (github-mcp.create_file) [ALLOW]
  → Updates Jira with design link (jira-mcp.add_comment) [ALLOW]
```

**Construction Phase:**
```
Agent selects golden path (idp-mcp.list_golden_paths)
  → Scaffolds service (idp-mcp.scaffold_service) [APPROVE]
  → Creates GitHub repo (github-mcp.create_repo) [APPROVE]
  → Generates code following coding standards
  → Creates PR (github-mcp.create_pr) [ALLOW]
  → Registers in CMDB (cmdb-mcp.create_service) [APPROVE]
  → Creates Jira sub-tasks (jira-mcp.create_issue) [APPROVE]
  → Provisions dev env (idp-mcp.provision_environment) [APPROVE]
```

**Operations Phase:**
```
Agent checks CI/CD status (cicd-mcp.get_pipeline_status)
  → Reviews test results
  → Provisions staging (idp-mcp.provision_environment) [APPROVE/SRE]
  → Creates change request for prod (approval-mcp) [APPROVE/CAB]
  → Updates CMDB with deployment info (cmdb-mcp.update_service) [APPROVE]
  → Updates Confluence runbook (confluence-mcp.update_page) [ALLOW]
```

### 5.3 IDP CLI Integration

If your org has a custom IDP CLI (e.g., wrapping Backstage, Terraform, and kubectl), the `idp-mcp` server wraps it:

```yaml
# idp-mcp wraps CLI commands behind MCP tools
cli_mappings:
  scaffold_service: "idp-cli create service --template {template_id} --name {service_name}"
  provision_environment: "idp-cli env provision --service {service_name} --env {environment}"
  register_in_catalog: "idp-cli catalog register --service {service_name} --metadata {metadata}"
  list_golden_paths: "idp-cli templates list --format json"
  get_golden_path_details: "idp-cli templates describe {template_id} --format json"
```

---

## 6. Audit & Control

### 6.1 Audit Trail Architecture

Every agent action is logged with full context:

```json
{
  "timestamp": "2026-05-19T14:30:00Z",
  "session_id": "aidlc-sess-abc123",
  "persona": "developer",
  "user": "raghu@org.com",
  "ide": "kiro",
  "aidlc_phase": "construction",
  "aidlc_stage": "code-generation",
  "action": "cmdb-mcp.create_service",
  "permission_level": "APPROVE",
  "approval_request_id": "apr-xyz789",
  "approval_status": "approved",
  "approved_by": "tech-lead@org.com",
  "approved_at": "2026-05-19T14:25:00Z",
  "payload": { "service_name": "order-processor", "owner": "commerce-team" },
  "result": { "status": "created", "catalog_url": "https://backstage.org.com/catalog/order-processor" },
  "duration_ms": 2340
}
```

### 6.2 Audit MCP Server

```yaml
name: audit-mcp
tools:
  - name: log_action
    description: "Log an agent action for audit trail"
    input:
      session_id: string
      persona: string
      action: string
      phase: string
      stage: string
      payload: object
      result: object
      approval_id: string?

  - name: query_audit_log
    description: "Search audit logs"
    input:
      filters: object   # by user, persona, action, date range, approval status
      limit: number
    output: array of audit entries

  - name: generate_compliance_report
    description: "Generate compliance report for a time period"
    input:
      start_date: date
      end_date: date
      format: enum       # "summary", "detailed", "csv"
```

### 6.3 Control Dashboard

Build a lightweight dashboard (or Backstage plugin) that shows:
- Active AI-DLC sessions and their current phase/stage
- Pending approval requests with SLA tracking
- Audit log with filtering and search
- Persona usage distribution
- Tool call frequency and patterns
- Policy violation attempts (denied actions)
- Cost attribution per session/project

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Weeks 1–4)

| Task | Deliverable |
|---|---|
| Fork `awslabs/aidlc-workflows` into org repo | Base AI-DLC rules in your control |
| Create the `.aidlc/` directory structure | Layered rule architecture |
| Write `sync.sh` and test across 2+ IDEs | IDE-agnostic distribution |
| Define 2 personas (architect, developer) | Initial persona rules |
| Write architecture + coding standards | First org standards files |
| Set up audit logging (CloudWatch/ELK) | Audit infrastructure |

### Phase 2: Guardrails (Weeks 5–8)

| Task | Deliverable |
|---|---|
| Build `approval-mcp` server | Approval workflow engine |
| Build `mcp-gateway` (start from Microsoft AGT) | Runtime policy enforcement |
| Define tool permission matrix | Guardrail policies |
| Integrate with Slack for approval routing | Approver notification |
| Build `audit-mcp` server | Structured audit logging |
| Pilot with 1 team using Kiro + Claude Code | Validated guardrail flow |

### Phase 3: IDP Integration (Weeks 9–12)

| Task | Deliverable |
|---|---|
| Build `idp-mcp` wrapping your IDP CLI / Backstage | IDP agent integration |
| Create 3 golden path templates | Reusable service templates |
| Write IDP onboarding steering rules | Agent-guided onboarding |
| Build `catalog-mcp` for service catalog queries | Service discovery for agents |
| Build `techradar-mcp` for technology governance | Tech stack governance |
| Pilot end-to-end: design → scaffold → deploy (dev) | Full lifecycle demo |

### Phase 4: PDLC Integration & Scale (Weeks 13–16)

| Task | Deliverable |
|---|---|
| Integrate `jira-mcp`, `confluence-mcp`, `github-mcp` | PDLC tool connectivity |
| Build `cmdb-mcp` for service registry | CMDB integration |
| Add remaining personas (tech-lead, SRE, QA) | Full persona coverage |
| Build control dashboard (Backstage plugin or standalone) | Visibility & governance UI |
| Roll out to 3+ teams | Broader adoption |
| Document runbooks and training materials | Org enablement |

### Phase 5: Maturity (Ongoing)

| Task | Deliverable |
|---|---|
| Add AI-DLC extensions (security, testing, compliance) | Extended quality gates |
| Implement cost attribution per AI-DLC session | FinOps integration |
| Build metrics: time-to-onboard, approval SLA, policy compliance | Success measurement |
| Contribute improvements back to `awslabs/aidlc-workflows` | Community engagement |
| Automate persona detection from Git/LDAP/SSO | Seamless persona switching |

---

## 8. MCP Server Summary: Build vs Buy

| MCP Server | Action | Effort | Notes |
|---|---|---|---|
| `jira-mcp` | **Use existing** | Low | Atlassian community MCP or build thin wrapper |
| `confluence-mcp` | **Use existing** | Low | Same as Jira |
| `github-mcp` | **Use existing** | Low | Official GitHub MCP available |
| `approval-mcp` | **Build** | Medium | Core to guardrail system; integrate with Slack/ServiceNow |
| `mcp-gateway` | **Build** (from AGT) | Medium-High | Start from Microsoft AGT or TrueFoundry Gateway |
| `audit-mcp` | **Build** | Low-Medium | Wraps your logging/SIEM platform |
| `policy-mcp` | **Build** | Low | Serves permission matrix; could be file-based |
| `idp-mcp` | **Build** | Medium | Wraps your IDP CLI / Backstage API |
| `catalog-mcp` | **Build** | Low | Read-only wrapper for Backstage catalog API |
| `techradar-mcp` | **Build** | Low | Serves technology radar data |
| `cmdb-mcp` | **Build** | Medium | Wraps ServiceNow or custom CMDB API |
| `cost-mcp` | **Build** | Low | Wraps AWS Cost Explorer or FinOps tool |
| `cicd-mcp` | **Build** | Low-Medium | Wraps GitHub Actions / Jenkins / CodePipeline API |

**Total custom MCP servers: ~10** (most are thin API wrappers)
**Total effort estimate: 1 senior engineer full-time for 16 weeks**, or a team of 2–3 at half capacity.

---

## 9. Key Design Decisions

1. **Rules-first, not tools-first**: The steering rules are the primary control mechanism. MCP Gateway is defense-in-depth, not the only layer.

2. **Portable markdown over proprietary configs**: Everything is `.md` files in git. No vendor lock-in. Works with any agent that reads project rules.

3. **Approval-by-request, not approval-by-blocking**: The agent doesn't hang waiting for approval. It creates a request, gives the user a link, and moves on to other work. The user or approver completes the flow asynchronously.

4. **Persona = role + permissions + standards**: A persona isn't just access control — it's the entire behavioral profile including which standards to enforce and which AI-DLC stages to emphasize.

5. **Golden paths over freestyle**: New services MUST use golden path templates. Freestyle creation requires explicit architect approval. This ensures organizational standards are embedded from the start.

6. **Audit everything, block selectively**: The default is to log and allow reads. Writes to external systems go through the approval flow. This balances velocity with control.

---

## 10. Success Metrics

| Metric | Baseline | Target (6 months) |
|---|---|---|
| Service onboarding time | 2+ weeks | < 2 days |
| Standards compliance in AI-generated code | Unknown | > 90% |
| Approval request turnaround | N/A | < 4 hours |
| Audit coverage of agent actions | 0% | 100% |
| Golden path adoption for new services | 0% | > 80% |
| IDE/agent coverage | 1 | 3+ IDEs supported |
| Developer satisfaction (survey) | Baseline | +20% improvement |
