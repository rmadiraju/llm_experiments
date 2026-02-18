# Support AI Service – Self-Service Help Flow Design

## Overview

This document defines the complete guided self-service help flow for the internal Developer Platform Support Bot built using:

- FastAPI
- LangGraph
- Chroma (Vector Store)
- AWS Bedrock (LLM)
- Optional MCP/API Tools

The bot enables internal users to resolve issues related to:
- Onboarding
- Build failures
- Deployment failures
- Testing issues
- NFR (Performance/Security/Reliability)
- Production releases
- Production incidents
- Access requests
- General inquiries

The design ensures:
- Structured slot-filling
- Deterministic + LLM-assisted routing
- API/tool-driven diagnostics
- Grounded responses via vector retrieval
- Safe escalation when required

---

# 1. High-Level Help Flow

## Conversation Lifecycle

1. Ask user what they need help with.
2. Classify intent:
   - `general_inquiry`
   - `access_request`
   - `issue`
3. If intent = issue:
   - Collect application context (appCode, type)
   - Fetch metadata from platform APIs
   - Identify issue area
   - Collect issue-specific details
   - Call relevant APIs/tools
   - Retrieve documentation from Chroma
   - Generate response
   - Escalate if needed

---

# 2. Conversation State Model

All conversations operate on a structured session state.

## Required Fields

```python
intent: str
issue_area: str
app_code: str
app_code_type: str   # BA | System | Component

environment: str
time_window: str
symptoms: str

build_id: str
deploy_id: str
test_run_id: str
change_id: str
correlation_ids: List[str]

tool_results: List[dict]
evidence: dict
missing_slots: List[str]

last_question: str
final_answer: str
confidence: float
escalate: bool
escalation_packet: dict
```

---

# 3. LangGraph Workflow Nodes

## Core Nodes

### 1. start_or_resume
Load existing state from history.

### 2. classify_intent
Determine:
- intent
- likely issue_area

### 3. ensure_app_context
If app_code missing:
- Ask user for BA/System/Component appCode

If ambiguous:
- Ask for clarification

### 4. fetch_metadata
Call platform APIs:
- Validate appCode
- Fetch hierarchy
- Fetch ownership
- Fetch metadata

### 5. collect_issue_details
Ask 2–3 area-specific questions.

### 6. plan_tools
Map issue_area → required tools.

### 7. execute_tools
Execute tools sequentially or in parallel.
Normalize outputs into structured format.

### 8. analyze_results
Convert tool outputs into evidence summary.

### 9. retrieve_knowledge
Query Chroma with metadata filters:
- app_code
- issue_area
- environment
- component
- tenant

### 10. compose_response
Structured output:

- What I Checked
- What I Found
- Recommended Actions
- References
- Next Steps

### 11. escalate_if_needed
Escalate if:
- Low confidence
- Platform inconsistency
- Permission block
- Incident detected

Generate escalation packet:
- appCode
- env
- logs summary
- correlation IDs
- suspected root cause
- user summary

---

# 4. Issue Area → Tools / Questions Mapping

| Issue Area | Questions to Ask (max 2–3/turn) | Tools to Call |
|------------|----------------------------------|---------------|
| Onboarding | Which step failed? Which env? When onboarded? | `get_hierarchy`, `get_app_metadata`, `check_access` |
| Build Failure | Build ID? Branch/commit? Error snippet? | `get_pipeline_run`, `get_build_logs` |
| Deployment Failure | Deploy ID? Env? Service/component name? | `get_deploy_status`, `service_health` |
| Testing | Test run ID? Flaky or consistent? Env/dataset? | `get_test_run`, `get_env_status` |
| NFR | Which NFR? Expected SLO? Time window? | `get_metrics`, `get_security_scan` |
| Prod Release | Change/release ID? Services impacted? Window/approvals? | `get_change_status`, `get_deploy_status` |
| Prod Incident | Start time? Impact? Severity? | `get_incident_data`, `get_metrics`, `get_logs` |
| Access Request | Role needed? App? Environment? | `check_access`, `get_app_metadata` |
| General Inquiry | Clarify what they’re trying to do (1 question) | Vector search only (`retrieve_knowledge`) |

Notes:
- If `issue_area` is unknown, ask **one** disambiguation question listing the available areas.
- If `app_code` is invalid, ask the user to confirm and provide examples (BA/System/Component).

---

# 5. Tool Registry Design

Each tool has:

```python
name: str
description: str
input_schema: Pydantic model
execute(): returns {
    status,
    summary,
    data,
    links,
    errors
}
```

## Example Tools

- `get_hierarchy(app_code)`
- `get_app_metadata(app_code)`
- `check_access(user_id, app_code)`
- `get_pipeline_run(build_id)`
- `get_build_logs(build_id, tail=200)`
- `get_deploy_status(deploy_id)`
- `get_metrics(app_code, env, window)`
- `get_incident_data(app_code, env, window)`
- `service_health(service, env)`
- `get_test_run(test_run_id)`
- `get_env_status(env, app_code)`

All outputs must be normalized.

---

# 6. Chroma Retrieval Strategy

## Document Types

- runbooks
- FAQs
- known issues
- hierarchy snapshots
- incident templates

## Metadata Fields

- tenant
- app_code
- app_code_type
- issue_area
- env
- service_name
- pipeline_name
- team
- severity
- doc_type

## Retrieval Strategy

1. Filter by `app_code + issue_area (+ env if available)`
2. Fallback to `issue_area` only + key terms extracted from `symptoms`
3. Include top 5–10 chunks
4. Provide references in final answer

---

# 7. Guardrails

- Block out-of-scope requests
- Prevent secret exposure
- Block destructive production actions
- Validate access before performing diagnostics

---

# 8. UX Principles

- Ask max 2–3 questions per turn
- Explain why you’re asking (to pull the right logs / validate hierarchy)
- Always show:
  - What I Checked
  - What I Found
  - What To Do Next
- Escalate cleanly with structured packet

---

# 9. API Response Format

## Question Response

```json
{
  "type": "question",
  "question": "What is the build ID?",
  "required_fields": ["build_id"],
  "state": {}
}
```

## Answer Response

```json
{
  "type": "answer",
  "answer": "...",
  "references": [],
  "state": {},
  "escalation_packet": {}
}
```

---

# 10. Implementation Mapping to Existing Code

- `app/services/workflow.py` → Add new nodes + graph wiring
- `app/services/chat.py` → Chat entrypoint: call graph, return question/answer
- `app/services/guardrail.py` → Scope rules + action safety
- `app/services/onboarding_client.py` → Expand API calls for hierarchy/metadata
- `app/services/tools/registry.py` → New tool registry + implementations/placeholders
- `app/services/search.py` → Metadata-filtered retrieval in Chroma
- `data/intents.json` → Add intents + issue areas + example utterances
- `data/prompts.json` → Add new prompt templates
- `data/system_prompt.txt` → Set system behavior constraints

---

# End of Design
