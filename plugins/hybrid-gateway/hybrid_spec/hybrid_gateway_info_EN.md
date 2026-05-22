# Hybrid Gateway — Hermes Documentation

> **Hermes vs OpenClaw extension differences**
>
> | Topic | Hermes |
> | --- | --- |
> | Hook | `pre_model_resolve` (`plugins/hybrid-gateway/__init__.py`) |
> | Classifier call | `auxiliary.classifier` → `call_llm(task="classifier")` |
> | Edge / cloud execution | Main `AIAgent` (`custom_providers` or inline `provider`/`model`) |
> | Pre-check | `edgeMaxContextTokens` threshold → force cloud (no `newSessionTier`) |
> | Per-level routing override | `routing.policyArray` (5 entries, overrides `policy`) |
> | Edge output truncation | 3× `length` continuations → `switch_model` to cloud (same turn) |
> | API / connection failures | Existing `fallback_providers` chain (not tier-aware) |
> | Logs | `~/.hermes/logs/agent.log` (`hybrid-gateway:` prefix) |

---

## 1. Overall Flow

```
User Input
  │
  ▼
╔═══════════════════════════════════════════════════════════╗
║  Pre-check: Context threshold / Bypass                    ║
║  • approximate_context_tokens ≥ edgeMaxContextTokens      ║
║    → force cloud, skip classify & route                     ║
║  • bypassPatterns matched → return None, keep Agent default ║
╠═══════════════════════════════════════════════════════════╣
║  Step 1: CLASSIFY                                         ║
║  auxiliary.classifier (or heuristic mode)                 ║
║  → { complexity, skills }                                 ║
╠═══════════════════════════════════════════════════════════╣
║  Step 1.5: POST RULES (hard post-processing)              ║
║  skills ≥ 4 → complexity at least complex                   ║
╠═══════════════════════════════════════════════════════════╣
║  Step 2: ROUTE                                            ║
║  Skill Route Override → policyArray / Three-Tier Policy     ║
║  → pick classifier / edge / cloud                          ║
╠═══════════════════════════════════════════════════════════╣
║  Step 3: EXECUTE                                          ║
║  pre_model_resolve returns provider/model for this turn   ║
║  • edge length truncation 3× → escalate to cloud (same turn)║
║  • API failure → fallback_providers (host mechanism)        ║
╚════════╦══════════════════╦══════════════╦════════════════╝
         ▼                  ▼              ▼
    Small local model    Medium/large local   Cloud API
    (classifier)            (edge)              (cloud)
```

---

## 2. Three-Tier Model Architecture

### 2.1 Model Role Overview

| Tier | Example models | Parameters | Location | Purpose | Port (example) |
| --- | --- | --- | --- | --- | --- |
| **classifier** | qwen2.5-3b, etc. | 3B | Local inference | Classifier + lightweight execution | `127.0.0.1:11434` |
| **edge** | gemma4-26B, qwen3.5-35B, nemotron-120B, qwen3.5-122B | 26B–122B | Local inference | Medium–high complexity tasks | `127.0.0.1:13141` |
| **cloud** | gemini-2.5-flash, claude-sonnet, etc. | — | Cloud API | Highest complexity / multimodal | API endpoint |

**Edge model to policy level mapping:**

| Model | Recommended policy |
| --- | --- |
| gemma4-26B | cost-optimize-L2 |
| qwen3.5-35B | cost-optimize-L2 |
| nemotron-120B | cost-optimize-L3 |
| qwen3.5-122B | cost-optimize-L3 |

**Core design:** The classifier model doubles as classifier and lightweight executor. Low-complexity tasks can stay on the classifier tier per policy, reducing large-model calls. Edge size follows hardware, paired with the matching policy level.

**Hermes config mapping:**

| Tier | Typical setting |
| --- | --- |
| classifier | `hybrid_gateway.models.classifier` → `ref: auxiliary.classifier` |
| edge | `ref: custom_providers.<name>` or inline `provider`/`base_url` |
| cloud | inline `provider` + `model` (e.g. `openrouter`) |

---

## 3. CLASSIFIER

### 3.1 Complexity levels

The classifier sends user text in a prompt to **`auxiliary.classifier`** (or `mode: heuristic` for rules only) and expects JSON:

```json
{"complexity":"moderate","skills":["coding"]}
```

| Level | Value | Description | Examples |
| --- | --- | --- | --- |
| `trivial` | 0 | Greetings, yes/no, arithmetic | "Hello", "1+1=?" |
| `simple` | 1 | Basic Q&A, short translation, simple reads | "Capital of France?" |
| `moderate` | 2 | Multi-step work, snippets, writing, most daily tasks | "Write a debounce function" |
| `complex` | 3 | Architecture, multi-doc synthesis, expert reports | "Design microservices" |
| `expert` | 4 | Novel algorithms, frontier research, PDF/image/ML needing cloud | "Distributed consensus" |

Full rules live in `CLASSIFIER_SYSTEM_PROMPT` in `classifier_prompt.py` (multi-file, CSV, PDF escalation, etc.).

**Full routing table per policy:**

| Complexity | cost-optimize-L1 | cost-optimize-L2<br>(Default) | cost-optimize-L3 | edge-first | cloud-first |
| --- | --- | --- | --- | --- | --- |
| trivial (0) | **edge** | **classifier** | **classifier** | **edge** | **cloud** |
| simple (1) | **edge** | **classifier** | **classifier** | **edge** | **cloud** |
| moderate (2) | **cloud** | **edge** | **edge** | **edge** | **cloud** |
| complex (3) | **cloud** | **cloud** | **edge** | **edge** | **cloud** |
| expert (4) | **cloud** | **cloud** | **cloud** | **edge** | **cloud** |

| Policy level | Edge models | Routing logic |
| --- | --- | --- |
| **L1** (~3B) | qwen2.5-3B (classifier = edge) | 0–1 → edge, 2–4 → cloud |
| **L2** (~26–35B) | gemma4-26B, qwen3.5-35B | 0–1 → classifier, 2 → edge, 3–4 → cloud |
| **L3** (~120B) | nemotron-120B, qwen3.5-122B | 0–1 → classifier, 2–3 → edge, 4 → cloud |

> **L3 needs a separate small ~3B classifier** (e.g. qwen2.5-3B). L1 can use one instance when classifier = edge.

**Default policy: `cost-optimize-L2`** (override via `hybrid_gateway.routing.policy`)

**Hermes-only: `policyArray`**

A 5-entry array **overrides** `policy`; indices 0–4 map trivial → expert:

```yaml
routing:
  policyArray: [classifier, classifier, edge, cloud, cloud]
```

---

### 3.2 Skills (skill detection)

Skills are **inferred by the classifier**, not configured manually (prompt in `classifier_prompt.py`).

| Skill | Description | Example triggers |
| --- | --- | --- |
| `coding` | Program logic / debugging | "Write a Python class" |
| `math` | Math / proofs | "Prove infinite primes" |
| `creative` | Stories, marketing copy | "Write a poem" |
| `analysis` | Analysis, comparison, reports | "Compare React and Vue" |
| `translation` | Translation | "Translate to English" |
| `search` | Web / external info | "Latest React 19 docs" |
| `tool-use` | APIs, batch files, shell drivers | "Call weather API" |
| `image-gen` | Image generation | "Draw a logo" |
| `conversation` | Simple chat | "Hello" |
| `summarization` | Summarize provided docs | "Summarize this article" |
| `reasoning` | Deep logic / planning | "Solve this puzzle" |

Multiple skills are allowed, e.g. `["translation", "search"]`.

**Use:** Skill Route Override in the router.

---

### 3.3 Reason (generated at route time)

The classifier does **not** return `reason`. The **router** (`router.py`) builds it.

Examples:

- `"policy=cost-optimize-L2, complexity=trivial -> classifier"`
- `"skill-route: image-gen"`
- `"context_tokens=150000>=131072"` (context threshold → cloud)

Used for debugging, `agent.log`, and TUI hybrid status.

---

### 3.4 Heuristic fallback

On invalid JSON or LLM failure, `heuristic.py` applies rules without throwing.

**Step 1: length**

| Input length | Complexity |
| --- | --- |
| < 20 chars | `trivial` |
| 20–99 | `simple` |
| 100–499 | `moderate` |
| ≥ 500 | `complex` |

**Step 2: keywords** (Hermes subset)

| Pattern | Effect |
| --- | --- |
| `code, function, class, def, async, return` | +coding, at least moderate |
| `debug, error, bug, fix, crash, exception` | +coding, at least complex |
| `translate, 翻譯, 翻译` | +translation, at least simple |
| `search, 搜尋, 搜索` | +search, at least simple |

Default skill: `"conversation"`.

Set `classifier.mode: heuristic` to **always** skip the LLM.

---

### 3.5 Cache

- Exact user input string as key
- In-memory map, cleared on restart
- Default TTL 300s, lazy eviction
- Skips model call on hit
- Disable: `hybrid_gateway.classifier.cacheEnabled: false`

---

### 3.6 Context threshold → cloud (Hermes)

When `approximate_context_tokens` **≥** `routing.edgeMaxContextTokens`, routing is skipped and `models.cloud` is used.

```yaml
hybrid_gateway:
  routing:
    edgeMaxContextTokens: 131072
```

Rationale: long sessions should use cloud when edge context is smaller.

> The OpenClaw extension has **`newSessionTier`** for `/new` and `/reset`; **not implemented** in the Hermes plugin today.

---

### 3.7 Bypass patterns

Matching `bypassPatterns` makes the hook return `None` — **no model override** for that turn.

```yaml
hybrid_gateway:
  routing:
    bypassPatterns: ["^/admin", "internal-debug"]
```

- Each entry is a **case-insensitive RegExp**
- First match wins
- Empty / unset → no bypass

Implemented in `pre_model_resolve` before classification.

---

### 3.8 Disable thinking

`hybrid_gateway.classifier.disableThinking` reduces classification latency (`classifier_thinking.py`):

| Strategy | Models | Implementation |
| --- | --- | --- |
| `gemma4-raw` | Gemma 4 | Raw `/v1/completions` prompt |
| `qwen-nothink` | Qwen (default) | `/nothink` suffix + `enable_thinking: false` |
| `auto` | — | `gemma4` in name → `gemma4-raw`, else `qwen-nothink` |

When disabled, standard `call_llm` chat path is not used for classification.

---

### 3.9 Post rules

Before routing, `apply_post_rules` in `router.py`:

| Condition | Effect |
| --- | --- |
| **≥ 4** skills | Complexity at least **`complex`** |

Adjust thresholds in `apply_post_rules`.

---

## 4. ROUTER

```
{ complexity, skills }
  → Post rules
  → Skill Route Override (if match)
  → policyArray or policy
  → provider/model override for pre_model_resolve
```

### 4.1 Skill route override

**Highest priority.** Example default: **`image-gen` → cloud**.

```yaml
hybrid_gateway:
  routing:
    skillRoutes:
      - skillPattern: image-gen
        forceTier: cloud
        reason: Image generation requires cloud multimodal model
```

`skillPattern` is a **regex** on each skill string. Combine patterns, e.g. `"image-gen|tool-use"`.

| Field | Description |
| --- | --- |
| `forceTier` | Force tier; uses resolved `models.<tier>` |
| `preferModel` | If set, uses `forceTier` branch (default tier `cloud`) |

Rules scan **top to bottom**; first match wins.

### 4.2 Three-tier routing policy

Used when no skill route matches and no valid `policyArray`.

#### Five-policy comparison

| Complexity | `edge-first` | `cloud-first` | `cost-optimize-L1` | `cost-optimize-L2`<br>(Default) | `cost-optimize-L3` |
| --- | --- | --- | --- | --- | --- |
| trivial (0) | **edge** | **cloud** | **edge** | **classifier** | **classifier** |
| simple (1) | **edge** | **cloud** | **edge** | **classifier** | **classifier** |
| moderate (2) | **edge** | **cloud** | **cloud** | **edge** | **edge** |
| complex (3) | **edge** | **cloud** | **cloud** | **cloud** | **edge** |
| expert (4) | **edge** | **cloud** | **cloud** | **cloud** | **cloud** |

### 4.3 Example

**"Write binary search in Python" → edge**

```
Classify: complexity=moderate, skills=[coding]
Skill route: no match
Policy: cost-optimize-L2, moderate(2) → edge

override:
  tier:   edge
  reason: policy=cost-optimize-L2, complexity=moderate -> edge
```

---

## 5. Execution and fallback

### 5.1 Edge length escalation (built into Hermes)

If this turn routed to **edge** and the model stops with `finish_reason=length`, the agent retries continuation on the same model (up to 3×). If still truncated, `_escalate_hybrid_to_cloud()` **`switch_model`s to `models.cloud` in the same turn** (hybrid edge path only).

This is separate from `fallback_providers` and targets “edge works but output does not fit.”

### 5.2 API / connection fallback

HTTP failures, timeouts, and rate limits use the host **`fallback_providers`** list (`provider/model` strings, not tier-aware).

Configure backups in `config.yaml` alongside hybrid tiers.

> OpenClaw’s plugin-level `fallbackEnabled` three-tier chain is **not** in the Hermes plugin; use `fallback_providers` + edge length escalation instead.

---

## 6. Routing decision storage

Hermes keeps per-turn state on `AIAgent` (`run_agent.py`):

- `_hybrid_tier`, `_hybrid_models`, `_hybrid_reason`, `_hybrid_escalated`

Log examples (`~/.hermes/logs/agent.log`):

```
hybrid-gateway: route tier=edge model=custom/gemma-4-26B reason=...
hybrid-gateway: length→cloud — escalated to cloud openrouter/...
```

The TUI shows classify phase and final tier via `status_callback`.

---

## 7. Config structure (`~/.hermes/config.yaml`)

See `hermes-config.example.yaml` in this directory.

```yaml
auxiliary:
  classifier:
    provider: custom
    model: qwen2.5-3b-instruct
    base_url: http://127.0.0.1:11434/v1
    timeout: 15
    max_tokens: 256

custom_providers:
  - name: gemma-4-26B
    base_url: http://127.0.0.1:13141/v1
    model: gemma-4-26B-A4B-it-UD-Q4_K_M.gguf

hybrid_gateway:
  enabled: true
  classifier:
    mode: auxiliary          # auxiliary | heuristic
    auxiliary_task: classifier
    cacheEnabled: true
    cacheTtlSeconds: 300
    disableThinking: false
    thinkingStrategy: auto
  routing:
    policy: cost-optimize-L2
    policyArray: [classifier, classifier, edge, cloud, cloud]
    edgeMaxContextTokens: 131072
    bypassPatterns: []
    skillRoutes:
      - skillPattern: image-gen
        forceTier: cloud
        reason: Image generation requires cloud multimodal model
  models:
    classifier: { ref: auxiliary.classifier }
    edge: { ref: custom_providers.gemma-4-26B }
    cloud:
      provider: openrouter
      model: anthropic/claude-sonnet-4
```

### Model `ref` syntax

| `ref` | Resolves to |
| --- | --- |
| `auxiliary.classifier` | `auxiliary.classifier` block |
| `custom_providers.gemma-4-26B` | Matching `name` in `custom_providers[]` |

Inline `{ provider, model, base_url, api_key }` is also supported.

---

## 8. Related docs

- `hermes-config.example.yaml` — copy-paste snippets
- [Hybrid Gateway feature guide](/docs/user-guide/features/hybrid-gateway) — quick start on the docs site
- `hybrid_arch.mmd` — architecture diagram (Mermaid)
- `hybrid_gateway_info_CN.md` — Chinese version of this document

---
