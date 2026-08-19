# TFS AI-DLC 2029: A Long-Term Vision

## From Governed Enablement to Bounded Autonomy

*Vision document — August 2026. This is a direction-setting document, not an implementation plan. It deliberately extends beyond what is buildable today and marks which capabilities are proven, emerging, or speculative.*

---

## 1. Why a Vision, and Why Now

The AI-DLC platform we are building today — steering rules, AgentCore Gateway with Cedar policies, approval workflows, audit trails — solves the 2026 problem: **letting agents participate in delivery without losing control.**

But the landscape is compounding. Industry experts now publish skill files for code review, architecture, security, and testing. Model context windows grow while attention quality per token shrinks. Agents are starting to call other agents. Standards (MCP, A2A, agents.md) are consolidating. Any vision that assumes today's constraints will be obsolete in 18 months; any vision that ignores today's governance needs will never be allowed into production at a financial services company.

So the vision must answer a harder question than "how do we use AI to code":

> **How does an organization accumulate software-building capability — skills, memory, judgment — in a form that survives model changes, tool changes, and team changes, and that agents can act on with earned, bounded autonomy?**

Everything below is organized around that question.

---

## 2. The Organizing Idea: Five Planes

Today's architecture is a pipeline: IDE → rules → CLI → gateway → tools. The long-term architecture is better understood as **five planes** that evolve at different speeds and are owned by different disciplines:

| Plane | What it holds | Today (2026) | Vision (2029) |
|---|---|---|---|
| **Interaction** | How humans and agents meet: IDEs, CLI, mobile approvals, chat | Kiro + TFS Core CLI (`tfs`) | Any surface; approvals and intent from anywhere; humans mostly review, rarely type code |
| **Skill** | *How* work should be done: SOPs, standards, expert techniques | Static steering rules synced per-IDE | Versioned, signed skill registry with semantic routing and just-in-time loading |
| **Memory** | *What is true* about our systems: specs, decisions, contracts, history | Specs repo + ADRs in git | Organizational memory fabric: git (truth) + knowledge graph (relationships) + episodic store (experience) |
| **Governance** | *What is allowed*: identity, policy, approvals, evals, audit | Okta + Cedar ALLOW/APPROVE/DENY + CloudWatch | Same spine, plus eval-gated earned autonomy, agent identity lifecycle, autonomy budgets |
| **Execution** | Where actions land: tools, environments, pipelines | AgentCore Gateway → PDLC/IDP MCPs | Fleet execution with simulation-first changes and blast-radius isolation |

The key claim: **the planes that matter most long-term are Skill and Memory** — they are where organizational capability accumulates. Gateways and IDEs will be replaced; a well-curated skill library and a truthful memory fabric compound in value for a decade. Design for their portability above all else.

---

## 3. The Skill Plane: An Economy, Not a Folder

### 3.1 The problem with today's approach

Steering rules today are "load everything, always." That fails three ways as skills multiply: context bloat (a 200-skill library cannot ride along on every request), staleness (rules drift from reality with no owner), and provenance (a skill file downloaded from a blog post carries no signature, no tests, no accountability — a supply-chain risk when an agent will act on it).

### 3.2 Skills as governed packages

Treat a skill the way we treat a software dependency:

```
skill: tfs.architecture.adr-authoring
version: 2.3.1
owner: architecture-guild
applies_to: [inception]
personas: [architect, tech-lead]
triggers:
  semantic: "creating or revising architecture decision records"
  signals: ["*.adr.md", "docs/adr/**"]
provenance:
  source: internal | vendor | community
  signed_by: TFS Skill Authority (Sigstore)
  reviewed: 2026-07-14
evals:
  suite: adr-quality-v4
  passing_score: 0.87
telemetry:
  invocations_90d: 412
  human_override_rate: 6%
```

The registry enforces a lifecycle: **propose → review → certify → publish → measure → deprecate.** Community/vendor skills enter through the same gate as internal ones — reviewed, signed, and eval-tested before any agent can load them. This is the skill supply chain, and in FS it deserves the same rigor as artifact repositories. *(Proven pattern: package registries. Emerging: skill signing — Sigstore-style attestation is directly reusable.)*

### 3.3 Right skill, right time: progressive disclosure + semantic routing

The answer to "how does the agent know which skills exist without reading everything" is a three-stage funnel — and this is where your RAG instinct is exactly right, with one refinement: **RAG the catalog, not just the content.**

1. **Always in context: the index card.** Every certified skill contributes ~40 tokens: name, one-line purpose, trigger hints. A 300-skill library costs ~12K tokens as an index — affordable and always visible. The agent knows what exists, the way a senior engineer knows the org has a "payments integration checklist" without reciting it.

2. **Retrieved on demand: the body.** When the task matches (deterministic signals like file patterns and AI-DLC phase, plus semantic match against the task description), the full skill body is fetched — from git via the registry, embeddings rebuilt on every merge. Hybrid retrieval: lexical + vector + graph ("skills used by services that depend on this one").

3. **Unloaded after use.** Skill bodies leave context when their task segment completes. The context is a workbench, not a warehouse.

**Git remains the source of truth; the vector index is a disposable, derived artifact.** Rebuild it nightly or on merge. Never let the index become authoritative — that inverts the trust model and makes drift invisible.

### 3.4 Skills earn their place

Every skill invocation is logged with outcome signals: did the human accept the output, how often was it overridden, did eval scores improve. Skills with high override rates get flagged to their owner; skills nobody invokes get deprecation notices. This turns the library from a wiki-graveyard into a **living economy where usage and quality data drive curation.** *(Speculative but tractable: needs the eval infrastructure in §6.)*

---

## 4. The Memory Plane: Making the Agent a Tenured Engineer

### 4.1 What "behaves like a human" actually requires

A ten-year engineer is valuable for three different kinds of memory, and the architecture should mirror them:

| Memory type | Human equivalent | System | Medium |
|---|---|---|---|
| **Semantic** | "Our payment service is event-driven because we tried sync in 2023 and it failed under load" | Specs, ADRs, contracts, *decision records with the why-nots* | **Git** — durable, reviewable, versioned |
| **Relational** | "If you touch the rate engine, talk to the pricing team; three services consume that contract" | Knowledge graph: services ↔ owners ↔ contracts ↔ decisions ↔ incidents | Graph store fed from CMDB, git, and gateway telemetry *(AWS Context is a bet-worthy managed option here)* |
| **Episodic** | "Last time we migrated a schema in that repo, the agent broke the nightly job — check the cron dependencies" | Session outcomes, approval history, incident postmortems, what-worked/what-failed | Structured store distilled from CloudWatch/S3 session logs |

### 4.2 Git is the truth; everything else is derived

Your instinct to keep context in git is correct, and it should be a stated architectural principle: **anything the organization must be able to audit, diff, and review lives in git.** Vector indexes, graph projections, and memory summaries are rebuildable projections of git + telemetry. If the graph store dies, we lose speed, not truth.

### 4.3 The context compiler

The piece almost everyone is missing: agents don't need *more* context, they need **compiled** context. Before a task starts, a context compiler assembles a task-scoped briefing within an explicit token budget:

```
Task: "Add retry logic to payment webhook handler"
Budget: 30K tokens
Compiled pack:
  ├── architecture-overview.md (project memory, 3K)
  ├── payment-service spec §4 webhooks (5K)          ← not the whole spec
  ├── ADR-019 idempotency decision + why-nots (2K)
  ├── contract: payment↔notification (graph, 1K)
  ├── episodic: "2026-03 retry storm incident" (2K)  ← ranked by relevance
  ├── skills index (12K)
  └── active skill bodies (5K, loaded JIT)
```

Retrieval ranking blends recency, graph distance from the touched code, and past usefulness. This is RAG, but with a budget, a ranking policy, and provenance on every retrieved chunk — so when the agent asserts "we chose async because X," it cites ADR-019, and a human can check.

### 4.4 Memory hygiene: the gardener

Memory that only grows becomes noise. A scheduled **memory-gardener agent** (running under its own persona and Cedar policy, like everything else) distills episodic logs into semantic summaries, flags contradictions between specs and observed behavior (feeding the reconciliation loop we already designed), proposes archival of stale context, and — critically — **never deletes source records**, only curates the derived layers. Contradiction reports route to tech leads like any other approval. *(Emerging: this is buildable today as a scheduled Kiro/Strands job; the discipline is the hard part, not the tech.)*

---

## 5. The Autonomy Ladder: Trust Is Earned Per-Domain, Not Granted Globally

"Making AI-DLC autonomous" is not a switch; it is a ladder where each rung is unlocked by evidence:

| Level | Name | Human role | Gate to advance |
|---|---|---|---|
| **L0** | Assist | Writes code, agent suggests | — |
| **L1** | Supervised delegation | Reviews every action | Baseline evals in place |
| **L2** | **Gated autonomy** *(we are here)* | Approves flagged actions (APPROVE flow) | Cedar + approval workflow live |
| **L3** | Policy autonomy | Reviews exceptions and samples | Eval scores ≥ threshold for this agent+skill+domain over N tasks; incident-free window |
| **L4** | Fleet autonomy | Sets intent, budgets, and constraints; handles escalations | L3 across the domain + simulation-first execution + proven rollback |

Three design rules make this safe in a financial services context:

**Autonomy is scoped, not global.** An agent may hold L3 for "test authoring in brownfield Java services" while holding L1 for "schema migrations." The unit of trust is *(agent, skill domain, blast radius)* — recorded as Cedar policy, so autonomy level is enforceable, auditable, and revocable, not aspirational.

**Autonomy has a budget.** Every autonomous run carries explicit limits: files/services it may touch, spend ceiling, wall-clock ceiling, and a required rollback plan. Exceeding any budget converts the run to APPROVE mode mid-flight. *(Speculative in tooling, sound in principle — the gateway is the natural enforcement point.)*

**Demotion is automatic.** A production incident traced to an autonomous change, or an eval-score drop after a model upgrade, demotes the affected domain one level pending review. Model upgrades are treated like dependency upgrades: re-run the eval suite before restoring autonomy.

L4 deserves honesty: fleet autonomy — agents decomposing intent, dispatching sub-agents over A2A, humans managing by exception — is where the industry is pointed, but nobody runs it safely at FS scale today. The vision's job is to make sure every layer we build (identity, evals, budgets, simulation) is a prerequisite for L4 rather than a rewrite.

---

## 6. The Governance Plane: What We Have, Plus Three Missing Organs

The 2026 spine — Okta OIDC, Cedar three-state policy at AgentCore Gateway, approval Lambda, CloudWatch audit, Kiro prompt logs to S3 — remains the spine. Three organs are missing for the long term:

### 6.1 Evals as infrastructure (the biggest gap)

Nothing in §3 (skill curation) or §5 (earned autonomy) works without measurement. Evals become a first-class CI citizen:

- **Skill evals**: every certified skill ships with a scenario suite; skills are re-evaluated on every model change and every skill edit.
- **Agent evals**: per-persona benchmark tasks drawn from *our* codebase (brownfield-representative, not toy repos), scored on correctness, standards compliance, and safety behaviors (does it stop at DENY, does it request approval properly).
- **Regression gates**: a model or prompt upgrade that drops eval scores blocks autonomy-level promotion the same way a failing test blocks a merge.

Evals-as-code lives in git next to skills; results feed the telemetry that drives both skill curation and autonomy decisions. *(Emerging: AgentCore's evaluation features and open-source harnesses make this buildable now; the investment is writing suites that reflect TFS reality.)*

### 6.2 Agent identity as a first-class lifecycle

Today agents borrow the developer's Okta identity. At L3+, agents act on schedules and on other agents' behalf, so they need **non-human identities**: issued per agent, scoped per domain, short-lived credentials via on-behalf-of token exchange, rotated and revocable independently of any human, and visible in every audit record as `acting_agent` + `on_behalf_of`. Treat NHI sprawl as a named risk — an inventory of agent identities with owners, like service accounts, reviewed quarterly.

### 6.3 Simulation before production ("agent staging")

Before an L3+ agent executes a plan against real systems, it executes against a **shadow environment**: ephemeral infra from IaC, synthetic-but-shaped data, replayed traffic where feasible. The diff between simulated intent and simulated outcome is attached to the approval request — so a tech lead approving a change reviews *evidence*, not promises. Expensive, so applied by risk tier: schema changes and cross-service refactors earn simulation; a unit-test PR does not. *(Speculative for full replay; pragmatic versions — ephemeral env + smoke suite — are buildable today.)*

### 6.4 Observability grows up

Audit answers "what happened." Long-term operations need "is it healthy": SLOs for agent work (approval turnaround, override rate, eval drift, cost per merged change), reasoning traces sampled for review the way we sample calls in a call center, and anomaly detection on tool-call patterns (an agent suddenly enumerating repos at 3 a.m. is a security signal, not a productivity one). All of this rides on the CloudWatch/OTel pipes we already run.

---

## 7. The Operating Model: What Changes for People

The uncomfortable, essential part of any real vision.

- **Personas become a roster.** Today a persona configures one agent. By L3+, the architect persona is a long-running architecture agent that reviews every design across the org for pattern compliance — personas stop being hats humans put on agents and become *standing roles agents occupy*, coordinating over A2A with humans in named oversight positions.
- **New human roles emerge**: skill curators (guild-owned), eval engineers, memory gardeners' supervisors, and an AI-DLC platform team that runs the five planes as a product with internal customers.
- **The career ladder shifts** from "writes excellent code" toward "specifies intent precisely, verifies rigorously, and encodes judgment into skills others (human and agent) reuse." Contribution to the skill library becomes a measured, celebrated artifact of seniority — the way open-source contribution is today.
- **Brownfield remains the differentiator.** The industry demos greenfield; TFS's advantage is a memory fabric that makes 15-year-old systems legible to agents. The graph + episodic memory of *legacy* behavior is worth more than any greenfield scaffold, and it is the asset competitors cannot download.

---

## 8. Standards Bets (How We Survive the Landscape Changing Daily)

The defense against churn is to hold assets in portable forms and take explicit, revisable bets on interfaces:

| Bet | Position | Confidence |
|---|---|---|
| **MCP** as the tool-access standard | All tool access through MCP behind AgentCore Gateway | High — industry-wide adoption |
| **Cedar** as policy language | All authorization as Cedar; portable even off AgentCore | High |
| **Markdown-in-git** for skills/specs/ADRs | Survives every IDE and model change | Highest — this is the hedge |
| **A2A** for agent-to-agent | Watch and prototype in 2027; don't build load-bearing systems on it yet | Medium |
| **Managed knowledge graph** (AWS Context et al.) | Adopt if it accepts our git/CMDB as sources and exports openly; otherwise build thin | Medium — avoid lock-in on the memory plane above all |

Rule of thumb: **own the content, rent the runtime.** Skills, specs, decisions, evals, and policies are ours in git; gateways, indexes, and even models are replaceable rentals.

---

## 9. Horizon Roadmap

**Horizon 1 — Governed Enablement (now–mid-2027).** Finish what's designed: gateway + Cedar + approvals in production, `tfs` CLI dual-mode, three-tier steering, specs repo + reconciliation loop, Kiro logging. *Add from this vision:* skill index cards + JIT loading (the 80/20 of the skill plane), eval suite v1 for two personas, agent identity inventory.

**Horizon 2 — The Knowing Organization (2027–2028).** Skill registry with signing, telemetry, and lifecycle. Knowledge graph over services/contracts/decisions. Context compiler with budgets and provenance. Memory gardener in production. Eval-gated L3 autonomy in two low-risk domains (test authoring, docs, dependency updates). Simulation-lite for risk-tiered changes.

**Horizon 3 — Bounded Autonomy (2028–2029).** Standing role agents on A2A with human oversight roles. Autonomy budgets enforced at the gateway. L4 pilots in tightly scoped domains with demonstrated rollback. The skill library and memory fabric recognized internally as a balance-sheet asset — measured, audited, and invested in like one.

Each horizon is valuable if we stop there. That is deliberate: a vision that only pays off at the end is a bet; one that pays per phase is a strategy.

---

## 10. Risks Worth Naming

- **Skill supply chain compromise** — a poisoned community skill is prompt injection with a certificate. Mitigation: signing, review gate, eval quarantine for new skills.
- **Memory poisoning / staleness** — wrong "truth" scales worse than no truth. Mitigation: provenance on every retrieved chunk, contradiction detection, git as reviewable source.
- **Eval theater** — suites that don't reflect brownfield reality grant false autonomy. Mitigation: draw eval tasks from real incident and PR history.
- **Autonomy ratchet** — pressure to promote levels without evidence after early wins. Mitigation: promotion criteria in policy, demotion automatic, both auditable.
- **Human skill atrophy** — verification quality decays if seniors stop building. Mitigation: rotation through skill authorship and eval engineering; treat these as build roles.

---

## 11. The One-Paragraph Version

We are building toward an organization where **skills are governed, signed packages routed to agents just-in-time; where organizational memory lives in git as truth with graph and episodic layers that make agents behave like tenured engineers; where autonomy is earned per-domain through evals, bounded by budgets, and revoked by evidence; and where the governance spine we run today — Okta, Cedar, approvals, audit — extends unchanged into a world of agent fleets.** The gateways and IDEs will be replaced along the way. The skill library, the memory fabric, and the eval record are the assets that compound — and they are ours.
