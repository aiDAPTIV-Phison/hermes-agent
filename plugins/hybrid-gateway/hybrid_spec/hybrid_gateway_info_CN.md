# Hybrid Gateway — Hermes 說明文件

> **Hermes 與 OpenClaw 延伸版差異摘要**
>
> | 項目 | Hermes |
> | --- | --- |
> | Hook | `pre_model_resolve`（`plugins/hybrid-gateway/__init__.py`） |
> | 分類器呼叫 | `auxiliary.classifier` → `call_llm(task="classifier")` |
> | Edge / Cloud 執行 | 主 `AIAgent`（`custom_providers` 或 inline `provider`/`model`） |
> | 啟動前檢查 | `edgeMaxContextTokens` 達閾值 → 強制 cloud（無 `newSessionTier`） |
> | 自訂每級路由 | `routing.policyArray`（5 項，覆寫 `policy`） |
> | Edge 輸出截斷 | `finish_reason=length` 續寫 3 次仍截斷 → `switch_model` 升雲（同回合） |
> | API / 連線失敗 | 既有 `fallback_providers` 鏈（與 tier 無直接耦合） |
> | 日誌 | `~/.hermes/logs/agent.log`（`hybrid-gateway:` 前綴） |

---

## 一、整體流程

```
User Input
  │
  ▼
╔═══════════════════════════════════════════════════════════╗
║  Pre-check: Context 閾值 / Bypass                          ║
║  • approximate_context_tokens ≥ edgeMaxContextTokens       ║
║    → 強制 cloud，跳過分類與路由                             ║
║  • bypassPatterns 命中 → 回傳 None，沿用 Agent 預設模型       ║
╠═══════════════════════════════════════════════════════════╣
║  Step 1: CLASSIFY（分類器）                                ║
║  auxiliary.classifier（或 heuristic 模式）                  ║
║  → { complexity, skills }                                   ║
╠═══════════════════════════════════════════════════════════╣
║  Step 1.5: POST RULES（後處理硬規則）                       ║
║  skills ≥ 4 → complexity 至少 complex                       ║
╠═══════════════════════════════════════════════════════════╣
║  Step 2: ROUTE（路由引擎）                                 ║
║  Skill Route Override → policyArray / Three-Tier Policy     ║
║  → 決定 classifier / edge / cloud                            ║
╠═══════════════════════════════════════════════════════════╣
║  Step 3: EXECUTE（執行）                                   ║
║  pre_model_resolve 回傳 provider/model → AIAgent 本回合執行   ║
║  • edge length 截斷 3 次 → 同回合升雲（switch_model）        ║
║  • API 失敗 → fallback_providers（主程式既有機制）           ║
╚════════╦══════════════════╦══════════════╦════════════════╝
         ▼                  ▼              ▼
    小型本地模型          中大型本地模型      雲端 API
    (classifier)            (edge)            (cloud)
```

---

## 二、三層模型架構

### 2.1 模型角色一覽

| Tier | 可用模型（範例） | 參數量 | 位置 | 用途 | 端口（範例） |
| --- | --- | --- | --- | --- | --- |
| **classifier** | qwen2.5-3b 等 | 3B | 本地推理服務 | 分類器 + 簡單任務執行 | `127.0.0.1:11434` |
| **edge** | gemma4-26B、qwen3.5-35B、nemotron-120B、qwen3.5-122B | 26B–122B | 本地推理服務 | 中高複雜度任務執行 | `127.0.0.1:13141` |
| **cloud** | gemini-2.5-flash、claude-sonnet 等 | — | 雲端 API | 最高複雜度 / 多模態任務 | API endpoint |

**Edge 模型與 Policy Level 對應：**

| 模型 | 建議 Policy |
| --- | --- |
| gemma4-26B | cost-optimize-L2 |
| qwen3.5-35B | cost-optimize-L2 |
| nemotron-120B | cost-optimize-L3 |
| qwen3.5-122B | cost-optimize-L3 |

**核心設計：** classifier 模型同時承擔分類與輕量執行。依 Policy 不同，低複雜度任務可直接由 classifier tier 回應，減少大模型呼叫。Edge 可依硬體選不同規模，並搭配對應 Policy Level。

**Hermes 配置對應：**

| Tier | 典型設定 |
| --- | --- |
| classifier | `hybrid_gateway.models.classifier` → `ref: auxiliary.classifier` |
| edge | `ref: custom_providers.<name>` 或 inline `provider`/`base_url` |
| cloud | inline `provider` + `model`（如 `openrouter`） |

---

## 三、CLASSIFIER（分類器）

### 3.1 Complexity 分級

分類器把 user input 塞進 prompt，送給 **`auxiliary.classifier`**（或 `mode: heuristic` 時純規則），要求回傳 JSON：

```json
{"complexity":"moderate","skills":["coding"]}
```

| 等級 | 數值 | 說明 | 範例 |
| --- | --- | --- | --- |
| `trivial` | 0 | 打招呼、yes/no、單字回答、時間查詢、純四則運算 | "你好"、"1+1=?" |
| `simple` | 1 | 基本問答、簡單數學、短翻譯、查詢型問題、簡單檔案讀取 | "法國首都？"、"翻譯 hello" |
| `moderate` | 2 | 多步驟指令、程式片段、摘要、寫作、檔案建立、設定編輯、資料分析、存檔/記憶儲存、大多數日常工作任務 | "寫個 debounce function" |
| `complex` | 3 | 系統架構設計、多文件深度綜合、需要專家判斷的競爭研究報告 | "設計微服務架構" |
| `expert` | 4 | 新演算法設計、博士級證明、前沿研究、PDF/影像/ML 等需雲端資源 | "實作分散式共識算法" |

完整分級規則見 `classifier_prompt.py` 內 `CLASSIFIER_SYSTEM_PROMPT`（含多檔案、CSV、PDF 等升級規則）。

**各 Policy 的路由對照表（完整）：**

| Complexity | cost-optimize-L1 | cost-optimize-L2<br>(Default) | cost-optimize-L3 | edge-first | cloud-first |
| --- | --- | --- | --- | --- | --- |
| trivial (0) | **edge** | **classifier** | **classifier** | **edge** | **cloud** |
| simple (1) | **edge** | **classifier** | **classifier** | **edge** | **cloud** |
| moderate (2) | **cloud** | **edge** | **edge** | **edge** | **cloud** |
| complex (3) | **cloud** | **cloud** | **edge** | **edge** | **cloud** |
| expert (4) | **cloud** | **cloud** | **cloud** | **edge** | **cloud** |

| Policy Level | 適用 Edge 模型 | 路由邏輯 |
| --- | --- | --- |
| **L1**（小型 ~3B） | qwen2.5-3B（classifier = edge 同一模型） | 0–1 → edge，2–4 → cloud |
| **L2**（中型 ~26-35B） | gemma4-26B、qwen3.5-35B | 0–1 → classifier，2 → edge，3–4 → cloud |
| **L3**（大型 ~120B） | nemotron-120B、qwen3.5-122B | 0–1 → classifier，2–3 → edge，4 → cloud |

> **注意：L3 需同時運行小型 ~3B classifier**（如 qwen2.5-3B）。L1 因 classifier = edge 為同一模型，只需一個實例。

**預設 Policy：`cost-optimize-L2`**（可在 `hybrid_gateway.routing.policy` 覆寫）

**Hermes 專用：`policyArray`**

若設定 5 項陣列，會**優先於** `policy`，索引 0–4 對應 trivial → expert：

```yaml
routing:
  policyArray: [classifier, classifier, edge, cloud, cloud]
```

---

### 3.2 Skills（技能偵測）

Skills **不是手動設定的**，由分類器從 user input 推斷（prompt 定義於 `classifier_prompt.py`）。

| Skill | 說明 | 範例觸發 |
| --- | --- | --- |
| `coding` | 撰寫或除錯**程式邏輯** | "寫個 Python class"、"這段 code 有 bug" |
| `math` | 數學計算或證明 | "證明質數有無限多" |
| `creative` | 創意寫作、故事、行銷文案 | "寫一首詩" |
| `analysis` | 資料分析、比較、評估、報告 | "比較 React 和 Vue" |
| `translation` | 語言翻譯 | "翻譯成英文" |
| `search` | 需要網搜或外部資訊 | "搜尋最新的 React 19 文件" |
| `tool-use` | 呼叫 API、批次檔案、終端腳本等 | "用 API 查天氣" |
| `image-gen` | 產生圖像、插圖 | "幫我畫一張 logo" |
| `conversation` | 簡單聊天、打招呼 | "你好" |
| `summarization` | 摘要既有文件 | "總結這篇文章" |
| `reasoning` | 深度邏輯推理、規劃 | "這個邏輯題怎麼解" |

模型可回傳多個 skills，例如 `["translation", "search"]`。

**用途：** Router 的 Skill Route Override 階段依 skill 強制路由。

---

### 3.3 Reason（路由階段產生）

**分類器本身不回傳 reason。** `reason` 由 **Router**（`router.py`）組裝，說明為何選該 tier。

範例：

- `"policy=cost-optimize-L2, complexity=trivial, skills=[conversation] -> classifier"`
- `"skill-route: image-gen"`
- `"context_tokens=150000>=131072"`（context 閾值強制 cloud）

用途：debug；寫入 `agent.log`，並在 TUI 顯示 hybrid 路由狀態。

---

### 3.4 Heuristic Fallback（關鍵字兜底）

分類器回傳**非合法 JSON** 或 LLM 失敗時，改用 `heuristic.py` 規則（不拋錯）。

**第一步：字數判斷**

| user input 長度 | 判定 complexity |
| --- | --- |
| < 20 字 | `trivial` (0) |
| 20 ~ 99 字 | `simple` (1) |
| 100 ~ 499 字 | `moderate` (2) |
| >= 500 字 | `complex` (3) |

**第二步：關鍵字加碼**（Hermes 實作子集）

| 關鍵字模式 | 效果 |
| --- | --- |
| `code, function, class, def, async, return` | skill +coding，complexity 至少 moderate |
| `debug, error, bug, fix, crash, exception` | skill +coding，complexity 至少 complex |
| `translate, 翻譯, 翻译` | skill +translation，至少 simple |
| `search, 搜尋, 搜索` | skill +search，至少 simple |

無命中時預設 skill 為 `"conversation"`。

可設定 `classifier.mode: heuristic` **完全跳過** LLM，永遠走 heuristic。

---

### 3.5 Cache 機制

- 以**完全相同**的 user input 字串為 cache key
- 記憶體 `Map`，程序重啟後清空
- TTL 預設 300 秒，過期於下次讀取時惰性清除
- 命中時跳過模型呼叫
- `hybrid_gateway.classifier.cacheEnabled: false` 可關閉

---

### 3.6 Context 閾值強制 Cloud（Hermes）

當 hook 收到 `approximate_context_tokens`，且數值 **≥** `routing.edgeMaxContextTokens` 時，**跳過分類與路由**，直接使用 `models.cloud`。

```yaml
hybrid_gateway:
  routing:
    edgeMaxContextTokens: 131072
```

```
上下文 token 估算 ≥ 閾值
  → models.cloud
  → reason = "context_tokens=...>=..."
```

設計考量：edge 模型 context 較小時，長對話應直接上雲，避免 edge OOM 或截斷。

> **OpenClaw 延伸版**另有 `newSessionTier`（`/new`、`/reset` 強制 tier）；**Hermes 插件目前未實作**此項。

---

### 3.7 Bypass Patterns（繞過路由）

prompt 符合 `bypassPatterns` 時，hook 回傳 `None`，**本回合不覆寫模型**，沿用 Agent 當前預設。

```yaml
hybrid_gateway:
  routing:
    bypassPatterns: ["^/admin", "internal-debug"]
```

- 每條為 **RegExp**（大小寫不敏感）
- 第一條命中即停止
- 未設定或空陣列則不攔截

實作於 `__init__.py` 的 `pre_model_resolve`，在分類器之前執行。

---

### 3.8 Disable Thinking（抑制思考模式）

`hybrid_gateway.classifier.disableThinking` 可降低分類延遲。依 `thinkingStrategy` 或模型名稱選策略（`classifier_thinking.py`）：

| 策略 | 適用模型 | 實作方式 |
| --- | --- | --- |
| `gemma4-raw` | Gemma 4 系列 | `/v1/completions` raw prompt，避免注入 think token |
| `qwen-nothink` | Qwen 系列（預設） | system prompt 加 `/nothink`；並設 `chat_template_kwargs.enable_thinking: false` |
| `auto`<br>(Default) | — | 模型名含 `gemma4` → `gemma4-raw`，否則 → `qwen-nothink` |

未啟用時使用標準 `call_llm` chat 路徑。

---

### 3.9 Post Rules（後處理硬規則）

分類結果進入 Router 前，`router.apply_post_rules` 執行確定性規則：

| 條件 | 效果 |
| --- | --- |
| skills 數量 **≥ 4** | complexity 至少 **`complex`** |

範例日誌：`post-rule: moderate -> complex (4 skills >= 4)`（若實作日誌）

調整：修改 `router.py` 中 `apply_post_rules` 的門檻邏輯。

---

## 四、ROUTER（路由引擎）

```
分類結果 { complexity, skills }
  │
  ▼
[Post Rules] skills ≥ 4 → complexity 至少 complex
  │
  ▼
[Stage 1] Skill Route Override → 命中？→ 直接決定 tier
  │
  ▼（沒命中）
[Stage 2] policyArray（若有）或 Three-Tier Policy
  │
  ▼
選出 provider/model → 回傳 pre_model_resolve override
```

### 4.1 Skill Route Override

**最高優先權。** 命中 `skillRoutes` 則不看 Policy。

**預設範例：`image-gen` → 強制 cloud。**

```yaml
hybrid_gateway:
  routing:
    skillRoutes:
      - skillPattern: image-gen
        forceTier: cloud
        reason: Image generation requires cloud multimodal model
```

`skillPattern` 為 **正則**，匹配 `skills[]` 中任一項。可組合，如 `"image-gen|tool-use"`。

| 欄位 | 說明 |
| --- | --- |
| `forceTier` | 強制 tier，使用 `models.<tier>` 解析後的 provider/model |
| `preferModel` | 有值時走 `forceTier`（預設 `cloud`）分支 |

規則**由上到下**，第一條命中即停止。

### 4.2 Three-Tier Routing Policy

僅在 **Skill Route 未命中** 且 **未設定有效 `policyArray`** 時使用 `policy`。

#### 五種 Policy 行為對照表

| 偵測到的 complexity | `edge-first` | `cloud-first` | `cost-optimize-L1` | `cost-optimize-L2`<br>(Default) | `cost-optimize-L3` |
| --- | --- | --- | --- | --- | --- |
| trivial (0) | **edge** | **cloud** | **edge** | **classifier** | **classifier** |
| simple (1) | **edge** | **cloud** | **edge** | **classifier** | **classifier** |
| moderate (2) | **edge** | **cloud** | **cloud** | **edge** | **edge** |
| complex (3) | **edge** | **cloud** | **cloud** | **cloud** | **edge** |
| expert (4) | **edge** | **cloud** | **cloud** | **cloud** | **cloud** |

| Policy | 思路 |
| --- | --- |
| `edge-first` | 一律 edge |
| `cloud-first` | 一律 cloud |
| `cost-optimize-L1` | 小型 edge：0–1 → edge，2–4 → cloud |
| `cost-optimize-L2` | 中型 edge：0–1 → classifier，2 → edge，3–4 → cloud |
| `cost-optimize-L3` | 大型 edge：0–1 → classifier，2–3 → edge，4 → cloud |

### 4.3 路由輸出範例

**「幫我寫一個 Python 的 binary search」→ edge**

```
分類: complexity=moderate, skills=[coding]
Skill Route: 無命中
Policy: cost-optimize-L2, moderate(2) → edge

override:
  tier:     edge
  model:    gemma-4-26B（依 custom_providers 為準）
  reason:   policy=cost-optimize-L2, complexity=moderate -> edge
```

---

## 五、執行與 Fallback

### 5.1 Edge Length 升雲（Hermes 內建）

本回合若路由為 **edge**，且模型因 context 上限以 `finish_reason=length` 截斷，Agent 會嘗試**同模型續寫**（最多 3 次）。仍失敗則呼叫 `_escalate_hybrid_to_cloud()`，**同回合** `switch_model` 至 `models.cloud`（僅限 hybrid edge 路徑）。

此機制與 `fallback_providers` 無關，專門處理「edge 能連線但輸出塞不下」的情況。

### 5.2 API / 連線 Fallback

實際 **HTTP 失敗、timeout、rate limit** 等由 Hermes 主程式的 `fallback_providers` 處理。該列表為 `provider/model` 字串，**不感知** hybrid tier。

建議在 `config.yaml` 的 `model.fallback_providers` 中列出 cloud（及必要時 edge）備援，與 hybrid 三 tier 搭配使用。

> OpenClaw 延伸版文件中的 `fallbackEnabled` 三層遞補**未在 Hermes 插件內實作**；Hermes 以 `fallback_providers` + edge length 升雲覆蓋主要失敗模式。

---

## 六、路由決策記錄

Hermes 在 `AIAgent` 上保存本回合 hybrid 狀態（`run_agent.py`）：

- `_hybrid_tier`、`_hybrid_models`、`_hybrid_reason`、`_hybrid_escalated`

日誌範例（`~/.hermes/logs/agent.log`）：

```
hybrid-gateway: route tier=edge model=custom/gemma-4-26B reason=policy=cost-optimize-L2...
hybrid-gateway: length→cloud — escalated to cloud openrouter/...
```

TUI 透過 `status_callback` 顯示 hybrid 階段（classify）與最終 tier。

---

## 七、設定檔結構（`~/.hermes/config.yaml`）

完整範例見同目錄 `hermes-config.example.yaml`。

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
    thinkingStrategy: auto   # auto | gemma4-raw | qwen-nothink
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

### Model `ref` 語法

| `ref` | 解析為 |
| --- | --- |
| `auxiliary.classifier` | `auxiliary.classifier` 區塊 |
| `custom_providers.gemma-4-26B` | `custom_providers[]` 中 `name` 相符者 |

亦可 inline 寫 `{ provider, model, base_url, api_key }`。

---

## 八、相關文件

- `hermes-config.example.yaml` — 可複製片段
- [Hybrid Gateway 功能指南](/docs/user-guide/features/hybrid-gateway) — 官網快速入門
- `hybrid_arch.mmd` — 架構圖（Mermaid）

---
