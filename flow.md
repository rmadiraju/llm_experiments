1) Target UX: one “Help Session” that slot-fills + routes

Treat every chat as a session state machine that progressively collects missing “slots”, calls the right APIs/tools, and returns either:
	•	Answer (docs + data-backed guidance)
	•	Action list (steps + links + commands)
	•	Escalation packet (pre-filled info for human support if needed)

Core slots (minimum)

Intent
	•	general_inquiry | access_request | issue

Identity / scope
	•	tenant (if relevant)
	•	user_id (from auth)
	•	app_code (BA/System/Component)
	•	app_code_type (derived or asked)

Issue specifics
	•	issue_area: onboarding | build | deployment | testing | nfr | prod_release | prod_incident
	•	symptoms: free text
	•	environment: dev | qa | stage | prod
	•	time_window: last 1h/24h/7d (for log queries)
	•	correlation_ids: build id / deploy id / pipeline run id, etc.

⸻

2) High-level LangGraph workflow (recommended graph)

You already have nodes like extract_context, guardrail, hydrate, search, llm_answer. Add these explicit conversation nodes:

A. Router + slot-filling
	1.	start_or_resume
	•	Load session state (from request history, or server-side store if you add it later)
	2.	classify_intent
	•	Determine intent and (if issue) likely issue_area
	3.	ensure_app_context
	•	If app_code missing → ask for it (with examples)
	•	If ambiguous type → ask “Is this BA / System / Component?”
	4.	fetch_metadata
	•	Call your platform API(s) to pull BA/System/Component metadata and validate appCode
	5.	collect_issue_details
	•	Ask 2–4 area-specific questions (see section 4)
	6.	tool_plan
	•	Decide which APIs/tools to call next (deterministic rules first, LLM as fallback)
	7.	execute_tools
	•	Run calls (sequential or parallel)
	8.	analyze_results
	•	Normalize tool outputs into a single evidence bundle
	9.	retrieve_knowledge
	•	Query Chroma with metadata filters (appCode, issue_area, component tags, env)
	10.	compose_response

	•	Produce final response: summary → findings → recommended actions → references

	11.	escalate_if_needed

	•	If confidence low / permission blocked / platform bug suspected → generate escalation packet

B. Guardrails (where they belong)

Keep your guardrail node early and add an “action safety” guardrail before returning:
	•	block requests outside support scope
	•	prevent leaking secrets
	•	restrict “dangerous actions” (e.g., deleting prod resources)

⸻

3) Data + retrieval strategy in Chroma (make answers grounded)

You’ll get best results if you store three types of documents, each with metadata:
	1.	Platform knowledge
	•	runbooks, FAQs, golden paths, known issues
	2.	App onboarding artifacts
	•	BA/System/Component hierarchy snapshots, ownership, pipelines, environments
	3.	Operational evidence (optional)
	•	recent failure signatures, error patterns, “resolution templates”

Metadata to store + filter on
	•	tenant, app_code, app_code_type, issue_area, env
	•	service_name, pipeline_name, repo, team, severity
	•	doc_type: runbook|faq|known_issue|api_snapshot|incident_template

Your retrieval should be:
	•	first: app_code + issue_area filtered search
	•	fallback: issue_area only + keyword from symptoms
	•	include: top 5–10 chunks + citations/links

⸻

4) Issue flows: the “issue_area” playbooks (questions + APIs)

Implement each issue area as a small subgraph (or a “handler” function) that defines:
	•	the questions to ask
	•	which APIs/tools to call
	•	a deterministic decision tree for common outcomes

4.1 Onboarding issues

Ask
	•	What step failed? (registration / hierarchy / permissions / repo linking / env setup)
	•	Which environment?
	•	When did you onboard?
Call
	•	GET /bas/{appCode}/hierarchy
	•	ownership/entitlements API
	•	onboarding status API (if exists)
Respond
	•	Validate hierarchy completeness
	•	Identify missing nodes or mismatch
	•	Provide steps + link to onboarding runbook
	•	Escalate if platform data inconsistent

4.2 Build failures

Ask
	•	Pipeline name + run/build id
	•	Branch/commit
	•	Error snippet
Call
	•	pipeline run status API
	•	build logs API (last N lines)
	•	dependency registry API (if relevant)
Respond
	•	classify failure: auth, dependency, compilation, test, infra
	•	action steps by class + “next best check”

4.3 Deployment failures

Ask
	•	Target env + deploy id
	•	Service/component name
	•	Rollout strategy (blue/green/canary)
Call
	•	deployment status API
	•	service health API
	•	change history API
Respond
	•	highlight failing step: image pull, config, permission, health checks
	•	safe rollback guidance + escalation criteria

4.4 Testing failures

Ask
	•	test suite + run id
	•	flake vs consistent
	•	environment + dataset
Call
	•	test run API
	•	environment readiness API
Respond
	•	isolate flake signals, rerun rules, known flaky list

4.5 NFR (performance/security/reliability)

Ask
	•	which NFR: latency/throughput/errors/vuln/compliance
	•	time window, env, expected SLO
Call
	•	metrics API
	•	security scan API
	•	policy evaluation API
Respond
	•	compare against SLO, show trend, recommend mitigations

4.6 Prod releases

Ask
	•	release id/change id
	•	affected services
	•	window + approvals
Call
	•	change management API
	•	deployment history
Respond
	•	readiness checklist + “go/no-go” guidance

4.7 Prod incidents

Ask
	•	symptoms + start time
	•	severity guess
	•	customer impact
Call
	•	incident API (if any)
	•	metrics + logs + tracing summaries
Respond
	•	triage steps + safe mitigations + escalation packet immediately

⸻

5) Tools design: “API tools” as a registry with schemas

Make tools first-class so LangGraph can plan and run them.

Tool wrapper pattern (recommended)

Each tool has:
	•	name
	•	description
	•	input_schema (Pydantic)
	•	execute() returns a normalized structure:
	•	status, data, summary, links, errors

Examples of tool categories:
	•	metadata.get_hierarchy(app_code)
	•	metadata.get_component(app_code)
	•	pipelines.get_run(run_id)
	•	pipelines.get_logs(run_id, tail=200)
	•	deploy.get_status(deploy_id)
	•	access.check(user_id, app_code)
	•	catalog.search_services(query, app_code)

This makes your tool_plan node simple:
	•	rule-based mapping: (issue_area -> required tools)
	•	plus optional LLM tool selection if the issue is unusual

⸻

6) Conversation rules (make it feel “self-service”, not chatbot-y)
	•	Ask at most 2–3 questions per turn
	•	Always show why you’re asking (“to pull the right logs / validate hierarchy”)
	•	After tool calls, present:
	1.	What I checked
	2.	What I found
	3.	What to do next (ranked)
	4.	If still stuck → escalation packet

⸻

7) Implementation mapping to your existing files

Based on your current structure:
	•	app/services/workflow.py
	•	add nodes + graph wiring for: classify_intent, ensure_app_context, collect_issue_details, tool_plan, execute_tools, analyze_results, compose_response, escalate_if_needed
	•	app/services/chat.py
	•	chat entrypoint: call graph, return either question or final answer
	•	app/services/guardrail.py
	•	add “support scope” + “action safety” policies
	•	app/services/onboarding_client.py
	•	expand for hierarchy + metadata endpoints
	•	app/services/tools/mcp.py
	•	optional: unify MCP tool calls behind same tool interface
	•	app/services/search.py
	•	retrieval with metadata filters
	•	data/intents.json
	•	include intents + issue areas + example utterances
	•	data/prompts.json + data/system_prompt.txt
	•	add prompts for classification, slot filling, response composition
