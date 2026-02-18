# Copilot Implementation Prompt

You are implementing a Support AI Service (FastAPI + LangGraph + Chroma + Bedrock) for an internal developer platform self-service bot.

## GOAL
Implement a guided help workflow with:
1) Ask what user needs help on  
2) Identify intent: `general_inquiry | access_request | issue`  
3) If issue: run an issue flow that collects app context + issue details, calls platform APIs as tools, retrieves relevant docs from Chroma, and returns actions + references.  
4) If confidence low or platform bug suspected, produce an escalation packet.

## EXISTING STRUCTURE (DO NOT BREAK)
- `app/services/chat.py`
- `app/services/workflow.py`
- `app/services/extract_context.py`
- `app/services/guardrail.py`
- `app/services/hydrate.py`
- `app/services/onboarding_client.py`
- `app/services/search.py`
- `app/services/llm_service.py`
- `app/vectorstore/chroma.py`
- `data/intents.json`
- `data/prompts.json`
- `data/system_prompt.txt`

---

## REQUIREMENTS

### A) Create Conversation State Model
Add a Pydantic model with:
- intent
- issue_area
- app_code
- app_code_type
- environment
- time_window
- symptoms
- identifiers (`build_id`, `deploy_id`, `test_run_id`, `change_id`, `correlation_ids`)
- `tool_results[]`
- `evidence{}`
- `missing_slots[]`
- `last_question`
- `final_answer`
- `confidence`
- `escalate`
- `escalation_packet{}`

### B) Add LangGraph Nodes
Add nodes (and wire them) in `app/services/workflow.py`:
- `classify_intent`
- `ensure_app_context`
- `fetch_metadata`
- `collect_issue_details`
- `plan_tools`
- `execute_tools`
- `analyze_results`
- `retrieve_knowledge`
- `compose_response`
- `escalate_if_needed`

### C) Tool Registry
Create: `app/services/tools/registry.py`

Each tool must define:
- `name`
- `description`
- `input_schema` (Pydantic)
- `execute()`

Implement at least placeholders using `onboarding_client.py` or stubs:
- `get_hierarchy`
- `get_app_metadata`
- `check_access`
- `get_pipeline_run`
- `get_build_logs`
- `get_deploy_status`
- `get_metrics`
- `get_incident_data`

Return a normalized structure:

```json
{
  "status": "ok|error|not_implemented",
  "summary": "short summary",
  "data": {},
  "links": [],
  "errors": []
}
```

### D) Prompts
Update `data/prompts.json` with templates for:
- `intent_classification`
- `issue_area_classification`
- `slot_question_generator` (ask minimal questions)
- `response_composer` (structured output)
- `escalation_packet_generator`

Response format must include:
- What I Checked
- What I Found
- Recommended Actions
- References

### E) Behavior Rules
- Ask max 2–3 questions per turn
- Never expose secrets
- Validate `app_code` and ask for confirmation when invalid
- If unknown `issue_area` → ask a single disambiguation question listing the areas
- If permissions block tool calls → escalate with packet

### F) FastAPI Contract
Ensure `POST /chat` passes `{message, history, top_k, use_tools, user_context}` into `chat.py`.

Return either:

**Question**
```json
{
  "type": "question",
  "question": "...",
  "required_fields": [],
  "state": {}
}
```

**Answer**
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

## DELIVERABLES
1) Provide the code changes across the files above  
2) Add minimal unit tests for:
   - `classify_intent`
   - `plan_tools` mapping  
3) Keep code clean, typed, and easy to extend with new issue areas  

END OF PROMPT
