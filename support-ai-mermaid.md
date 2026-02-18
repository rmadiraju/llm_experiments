# Support AI Service – Mermaid Diagrams

This file contains Mermaid diagrams for the self-service workflow.

---

## 1) End-to-End LangGraph Workflow (Flowchart)

```mermaid
flowchart TD
  U[User / UI: Ask Help] -->|POST /chat (message + history)| API[FastAPI /chat]
  API --> LG[LangGraph Workflow]

  subgraph LGW[LangGraph Nodes]
    SR[start_or_resume]
    CI[classify_intent]
    EA[ensure_app_context]
    FM[fetch_metadata]
    CID[collect_issue_details]
    PT[plan_tools]
    ET[execute_tools]
    AR[analyze_results]
    RK[retrieve_knowledge (Chroma)]
    CR[compose_response]
    EN[escalate_if_needed]
  end

  LG --> SR --> CI --> EA
  EA -->|missing app_code/type| Q1[Ask user for appCode/type]
  Q1 --> API

  EA --> FM --> CID
  CID -->|missing details| Q2[Ask 2-3 issue questions]
  Q2 --> API

  CID --> PT --> ET --> AR --> RK --> CR --> API

  CR -->|low confidence / inconsistency / perms blocked| EN --> API

  %% External systems
  FM --> META[Platform Metadata APIs]
  ET --> TOOLS[API Tool Registry / Optional MCP]
  RK --> VS[Chroma Vector Store]
  CI --> LLM1[Bedrock LLM]
  CR --> LLM2[Bedrock LLM]
  EN --> LLM3[Bedrock LLM]
```

---

## 2) Issue Flow Subgraph (Decision + Slot Filling)

```mermaid
flowchart LR
  A[Intent = issue] --> B{Have app_code?}
  B -- No --> B1[Ask: BA/System/Component appCode] --> B
  B -- Yes --> C[Fetch BA/System/Component metadata]

  C --> D{Have issue_area?}
  D -- No --> D1[Ask single disambiguation: onboarding/build/deploy/testing/NFR/prod release/prod incident] --> D
  D -- Yes --> E[Collect 2-3 issue_area-specific details]

  E --> F[Plan tools for issue_area]
  F --> G[Execute tools + normalize]
  G --> H[Retrieve docs from Chroma (filtered)]
  H --> I[Compose response: Checked/Found/Actions/Refs]
  I --> J{Escalate?}
  J -- Yes --> K[Generate escalation packet]
  J -- No --> L[Return final answer]
  K --> L
```

---

## 3) Issue Area → Tools (Quick View)

```mermaid
flowchart TB
  subgraph Onboarding
    OQ[Questions: step failed? env? when onboarded?]
    OT[Tools: get_hierarchy, get_app_metadata, check_access]
  end

  subgraph Build
    BQ[Questions: build_id? branch/commit? error snippet?]
    BT[Tools: get_pipeline_run, get_build_logs]
  end

  subgraph Deploy
    DQ[Questions: deploy_id? env? service/component?]
    DT[Tools: get_deploy_status, service_health]
  end

  subgraph Testing
    TQ[Questions: test_run_id? flaky vs consistent? env/dataset?]
    TT[Tools: get_test_run, get_env_status]
  end

  subgraph NFR
    NQ[Questions: which NFR? expected SLO? time window?]
    NT[Tools: get_metrics, get_security_scan]
  end

  subgraph ProdRelease
    PRQ[Questions: change_id? services? window/approvals?]
    PRT[Tools: get_change_status, get_deploy_status]
  end

  subgraph ProdIncident
    PIQ[Questions: start time? impact? severity?]
    PIT[Tools: get_incident_data, get_metrics, get_logs]
  end
```

---

## 4) Sequence Diagram (Typical Issue)

```mermaid
sequenceDiagram
  participant UI as UI (Chat Tab)
  participant API as FastAPI /chat
  participant LG as LangGraph Workflow
  participant META as Platform APIs
  participant VS as Chroma
  participant LLM as Bedrock LLM
  participant TOOLS as Tool Registry/MCP

  UI->>API: POST /chat (message + history)
  API->>LG: invoke(state, message)

  LG->>LLM: classify_intent + issue_area

  alt app_code missing
    LG-->>API: Return question: request app_code/type
    API-->>UI: question
  else app_code present
    LG->>META: Fetch hierarchy/metadata (validate app_code)
    META-->>LG: metadata + ownership + envs
  end

  alt issue details missing
    LG-->>API: Return question: request missing issue details (max 2-3)
    API-->>UI: question
  else details present
    LG->>LG: plan_tools(issue_area)
    LG->>TOOLS: Execute tools (logs/status/metrics/access)
    TOOLS-->>LG: normalized tool results

    LG->>VS: vector search (filters: app_code, issue_area, env)
    VS-->>LG: top_k docs

    LG->>LLM: compose_response (evidence + docs)
    LLM-->>LG: structured answer

    alt confidence low / permissions blocked / inconsistency
      LG->>LLM: generate escalation_packet
      LLM-->>LG: escalation_packet
    end

    LG-->>API: final answer (+ escalation packet optional)
    API-->>UI: response
  end
```

---

# End of Mermaid Diagrams
