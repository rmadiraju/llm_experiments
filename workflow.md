Use LangGraph for orchestration and LangChain for model/tool abstractions.

For your use case, I would not build one free-form agent that does everything. I would build a stateful workflow graph with a few controlled agentic steps:
	1.	understand the request,
	2.	gather context,
	3.	route to the right path,
	4.	troubleshoot with guided questioning and tool calls,
	5.	either give self-service resolution or hand off to support with a complete case summary.

That matches how LangGraph is intended to be used: workflows for predictable control flow, agents for dynamic tool use inside specific nodes. LangGraph explicitly separates “workflow” vs “agent” patterns, supports persistence, memory, interrupts, and human-in-the-loop, which are all useful for support/self-service systems.  ￼

Recommended architecture

1) Outer graph = deterministic support workflow

Use a StateGraph as the top-level controller. Keep the main path predictable.

Suggested top-level nodes:
	•	ingest_request
	•	load_user_context
	•	classify_intent
	•	resolve_reference
	•	doc_lookup or issue_triage
	•	clarify_if_needed
	•	run_diagnostics
	•	reason_about_root_cause
	•	propose_resolution
	•	escalate_or_close

This is better than a single unconstrained agent because support flows have clear checkpoints, guardrails, and audit needs. LangGraph is designed for that kind of durable, stateful orchestration.  ￼

2) Inner nodes = focused agents/tools

Inside some nodes, use LangChain agents or direct tool-calling models:
	•	classify_intent: structured output classifier
	•	doc_lookup: retrieval agent or simple retriever chain
	•	run_diagnostics: tool-enabled troubleshooting agent
	•	reason_about_root_cause: synthesis node
	•	propose_resolution: answer generation node

LangChain’s tools, agents, structured output, and model abstractions are a good fit here. Use structured outputs heavily so each node returns typed JSON instead of prose.  ￼

The flow I would implement

A. Request understanding

When the user says something like “I have an issue,” first classify into something like:
```json
{
  "intent": "general_info | issue | request_status | how_to | access_problem | deployment_problem | unknown",
  "issue_area": "repo | pipeline | aws_resource | auth | onboarding | release | unknown",
  "confidence": 0.0,
  "needs_clarification": true,
  "missing_fields": ["app_id", "environment", "pipeline_id"],
  "user_goal": "what they are trying to do"
}
```

This is where LangChain structured output helps a lot. It avoids brittle text parsing and gives you routing data directly.  ￼

B. Preload context before asking questions

Before you ask anything back, call context tools:
	•	user profile / team / tenant
	•	recent tickets
	•	recent workflow runs / deployments
	•	recent API errors
	•	recent actions the user attempted
	•	current selected app/project/env from your product UI
	•	last known failed item

Then generate a proactive message like:

“I see you recently tried deploying payments-api to stage and the last run failed in the infra-validate step. Are you asking about that?”

This is exactly where state + memory + tool calls help. LangGraph supports per-thread persistence and long/short-term memory patterns, and LangChain’s context engineering guidance fits this design well.  ￼

C. Route into one of two main paths

Path 1: General information / “how do I…”
Use:
	•	documentation retriever
	•	FAQ KB
	•	policy/SOP retriever
	•	optional doc-grounded answer node

This can be a small RAG node or retrieval agent. LangGraph’s agentic RAG pattern is a close match if you want the model to decide whether retrieval is needed.  ￼

Path 2: Issue / troubleshooting
Use a guided triage loop:
	1.	identify probable issue area
	2.	determine required facts
	3.	ask only the missing questions
	4.	call the relevant diagnostics APIs
	5.	reason over results
	6.	return either:
	•	self-service fix,
	•	safe workaround,
	•	or escalation package

This should be a controlled subgraph, not a free-form open-ended loop. LangGraph subgraphs are a good fit for domain-specific troubleshooting flows.  ￼

Best multi-agent pattern for this use case

Use a supervisor + specialist agents pattern, but keep the supervisor simple.

Suggested specialists:
	•	Intent Router Agent
	•	Documentation Agent
	•	Troubleshooting Agent
	•	Context Collector Agent
	•	Case Summary / Escalation Agent

LangChain’s subagent/supervisor pattern is meant for this: a central coordinator routes work to focused specialists. I would keep memory at the parent graph level and make most subagents stateless.  ￼

Concrete state model

A good AgentState might look like this:
```python
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel

class AgentState(BaseModel):
    thread_id: str
    user_id: str
    tenant_id: Optional[str] = None

    user_message: str
    conversation_history: List[Dict[str, Any]] = []

    intent: Optional[str] = None
    issue_area: Optional[str] = None
    confidence: Optional[float] = None

    referenced_item: Optional[Dict[str, Any]] = None
    recent_context: Dict[str, Any] = {}
    collected_facts: Dict[str, Any] = {}
    missing_fields: List[str] = []

    documentation_hits: List[Dict[str, Any]] = []
    diagnostic_results: Dict[str, Any] = {}
    probable_causes: List[Dict[str, Any]] = []

    next_question: Optional[str] = None
    user_can_self_resolve: Optional[bool] = None
    resolution_steps: List[str] = []
    escalation_needed: bool = False
    escalation_summary: Optional[str] = None

    final_answer: Optional[str] = None
```

The important part is that the graph state should store both the customer conversation state and the operational investigation state. That aligns with LangGraph’s stateful workflow model.  ￼

How to decide what to ask the user

Do not let the agent ask arbitrary questions first.

Create an issue_area -> required facts -> tools -> playbook mapping.

Example:
```python
TROUBLESHOOTING_MAP = {
    "pipeline": {
        "required_fields": ["app_id", "environment", "pipeline_run_id"],
        "tools": ["get_recent_pipeline_runs", "get_pipeline_logs", "get_deployment_status"],
        "playbook": "pipeline_triage_v1"
    },
    "auth": {
        "required_fields": ["user_email", "app_id", "environment"],
        "tools": ["get_user_access", "get_sso_status", "get_recent_auth_failures"],
        "playbook": "auth_triage_v1"
    },
    "onboarding": {
        "required_fields": ["component_type", "app_id", "request_id"],
        "tools": ["get_onboarding_request", "get_repo_status", "get_provisioning_status"],
        "playbook": "onboarding_triage_v1"
    }
}
```

Then the graph logic is:
	•	infer issue_area
	•	look up required fields
	•	compare required fields with already-known context
	•	ask only what is missing
	•	call only the mapped tools

This dramatically improves reliability over “LLM chooses anything.” It also keeps your tool surface safer. LangChain recommends careful tool design, and router patterns are explicitly supported for dispatching requests to specialized flows.  ￼

Example graph shape

```
START
  -> ingest_request
  -> load_user_context
  -> classify_intent
  -> maybe_resolve_reference

if intent == general_info/how_to:
  -> doc_lookup
  -> generate_doc_answer
  -> END

if intent == issue:
  -> identify_issue_area
  -> determine_missing_info
  -> ask_clarifying_question?  (interrupt if needed)
  -> run_diagnostics
  -> analyze_root_cause
  -> decide_self_service_vs_escalate
      -> self_service_answer
      -> or escalation_summary
  -> END
```
Use interrupts when the workflow needs the user to answer a question. LangGraph supports pausing execution and resuming later with persisted state, which is exactly what you want for multi-turn support conversations.  ￼

Where LangGraph specifically shines here

LangGraph features that matter most for your case:
	•	durable execution for multi-turn troubleshooting,
	•	checkpointers so conversations resume cleanly,
	•	interrupts to ask the user for missing info,
	•	human-in-the-loop for escalation approval or sensitive actions,
	•	memory across sessions so the assistant remembers recent failures or user preferences.  ￼

Suggested tool categories

Define tools very explicitly.

Context tools
	•	get_user_profile(user_id)
	•	get_recent_actions(user_id, last_n_days)
	•	get_recent_failures(user_id, tenant_id)
	•	get_recent_support_cases(user_id)

Documentation tools
	•	search_docs(query, product_area)
	•	search_runbooks(issue_area, keywords)
	•	get_policy_snippets(topic)

Operational diagnostics tools
	•	get_pipeline_runs(app_id, env)
	•	get_pipeline_logs(run_id)
	•	get_deployment_status(app_id, env)
	•	get_repo_scaffold_status(app_id)
	•	get_aws_resource_status(app_id, env, resource_type)
	•	get_user_access(user_email, app_id)
	•	get_recent_api_errors(app_id, env)

Support tools
	•	create_support_case(summary, evidence)
	•	suggest_next_steps(issue_area, probable_cause)

LangChain tools are designed exactly for this sort of typed external action surface.  ￼

Troubleshooting node design

Inside the troubleshooting subgraph, I would break the work into 4 smaller nodes:

1. Hypothesis node

Given the user message + recent context, generate:
	•	likely issue areas
	•	likely referenced item
	•	which facts are already known
	•	which facts are still missing

2. Clarification node

If confidence is low or critical facts are missing, ask a pointed question such as:
	•	“Is this about the failed stage deployment from 10:42 AM?”
	•	“Which environment: dev, stage, or prod?”
	•	“Are you blocked on access, repo creation, or pipeline execution?”

Use an interrupt here.  ￼

3. Diagnostics node

Run only the relevant APIs. Keep this mostly rule-driven.

4. Root-cause node

Synthesize results into:
	•	probable cause
	•	evidence
	•	confidence
	•	self-service possible?
	•	exact next steps
	•	escalation needed?

Use structured output again here.  ￼

Self-service vs escalation logic

A simple policy works well:

Self-service
	•	known issue pattern
	•	clear user fix available
	•	safe action
	•	confidence high enough

Escalate
	•	missing permissions
	•	backend/internal defect
	•	repeated failed attempts
	•	ambiguous system state
	•	high-risk environment
	•	confidence too low

When escalating, the bot should not just say “contact support.” It should produce a ready-made case:
	•	user
	•	tenant/app/env
	•	probable issue
	•	recent attempts
	•	diagnostics results
	•	suggested owner/team
	•	copied logs/IDs/links

That creates real operational value.

Proactive “are you talking about this item?” feature

This is a great idea. Implement it as a reference resolution node after context loading.

The node should:
	1.	query recent user activity,
	2.	rank candidate items,
	3.	if one candidate is strong enough, ask a confirmation question.

Output shape:
```json
{
  "candidate_items": [
    {
      "type": "pipeline_run",
      "id": "run_123",
      "label": "payments-api / stage / failed at infra-validate",
      "score": 0.91
    }
  ],
  "should_confirm": true,
  "confirmation_prompt": "Are you asking about the failed payments-api deployment to stage from 10:42 AM?"
}
```

This is a classic combination of context engineering + structured routing + memory.  ￼

Minimal implementation strategy

Build in 3 phases.

Phase 1: Controlled single-agent workflow
	•	one LangGraph state graph
	•	one classifier node
	•	one docs node
	•	one troubleshooting node
	•	5–8 core tools
	•	no multi-agent yet
	•	use interrupt() for clarifications
	•	persistent checkpointer

This will get you production value faster. LangGraph’s persistence and interrupts make this viable without overcomplicating the system.  ￼

Phase 2: Add specialist subagents

Split docs and troubleshooting into separate specialists once tool count grows.

Phase 3: Add memory and proactive resolution
	•	recent user attempts
	•	prior recurring issues
	•	remembered preferred app/team/context
	•	case deflection and escalation summarization

What I would avoid
	•	One giant ReAct agent with every tool exposed.
	•	Letting the model decide all clarifying questions with no schema.
	•	Letting the model decide escalation policy by itself.
	•	Relying only on vector search for troubleshooting.
	•	Putting long-term memory directly into prompts without controlled retrieval.

LangGraph/LangChain support agentic patterns, but the docs also make clear that workflows are preferable where the code path should be controlled and predictable.  ￼

My recommended stack
	•	LangGraph: orchestration, state, interrupts, checkpointer
	•	LangChain: models, tools, structured output, retrievers
	•	Postgres checkpointer in production for persisted graph state
	•	Vector store + doc metadata filters for documentation
	•	Domain API wrapper layer for your internal app/platform APIs
	•	Observability for every node/tool call
	•	Playbook registry for issue-specific troubleshooting logic

LangChain’s docs explicitly recommend persistent checkpointers for production human-in-the-loop flows rather than in-memory persistence.  ￼

Practical blueprint

If I were building this for your internal developer platform, I would start with these graph nodes:
	1.	load_profile_and_recent_activity
	2.	classify_request
	3.	resolve_recent_reference
	4.	route_request
	5.	search_docs
	6.	identify_issue_area
	7.	collect_missing_facts
	8.	ask_user_interrupt
	9.	run_domain_diagnostics
	10.	synthesize_root_cause
	11.	respond_with_fix_or_workaround
	12.	build_support_handoff

That is the sweet spot: structured enough to be reliable, agentic enough to feel smart.
