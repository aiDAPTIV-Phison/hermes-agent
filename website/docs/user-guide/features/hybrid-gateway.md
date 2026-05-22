# Hybrid Gateway

Route each user turn to a **classifier**, **edge**, or **cloud** model based on input complexity, optional skill patterns, and routing policy.

The bundled plugin lives at `plugins/hybrid-gateway/`. Full design notes (ported from OpenClaw) are in `plugins/hybrid-gateway/hybrid_spec/hybrid_gateway_info_CN.md`.

## Hermes vs OpenClaw

| Topic | Hermes |
|-------|--------|
| Hook | `pre_model_resolve` (once per turn, before the tool loop) |
| Classifier call | `auxiliary.classifier` via `call_llm(task="classifier")` |
| Edge / cloud | Main `AIAgent` backend (`custom_providers` or inline config) |
| Length failover | Edge output truncated 3× → `switch_model` to **cloud** (same turn) |
| API errors | Existing `fallback_providers` chain (unchanged) |

## Quick start

1. Enable the plugin in `~/.hermes/config.yaml`:

```yaml
hybrid_gateway:
  enabled: true
  classifier:
    mode: auxiliary
    auxiliary_task: classifier
  routing:
    policy: cost-optimize-L2
  models:
    classifier: { ref: auxiliary.classifier }
    edge: { ref: custom_providers.gemma-4-26B }
    cloud: { provider: openrouter, model: anthropic/claude-sonnet-4 }
```

2. Configure the classifier side task:

```yaml
auxiliary:
  classifier:
    provider: custom
    model: your-small-model
    base_url: http://127.0.0.1:11434/v1
    timeout: 15
```

Optional classifier thinking control (under `hybrid_gateway.classifier`):

```yaml
hybrid_gateway:
  classifier:
    disableThinking: true
    thinkingStrategy: auto   # auto | gemma4-raw | qwen-nothink
```

When `disableThinking` is true, the plugin appends `/nothink` to the system prompt. With `thinkingStrategy: auto`, model names containing `gemma4` use raw `/v1/completions`; other models use chat completions with `chat_template_kwargs.enable_thinking: false` (for Qwen 3/3.5 on llama.cpp-style servers).

3. Define edge in `custom_providers` (list with `name`):

```yaml
custom_providers:
  - name: gemma-4-26B
    base_url: http://127.0.0.1:13141/v1
    model: your-edge-model.gguf
```

4. Restart Hermes (CLI or gateway). Check `~/.hermes/logs/agent.log` for `hybrid-gateway:` routing lines.

## Routing

1. **Classify**: JSON `{ complexity, skills }` (or heuristic fallback).
2. **Route**: `skillRoutes` → `policyArray` (5 items) → `policy` (e.g. `cost-optimize-L2`).
3. **Execute**: Selected tier becomes the main agent model for that turn.

## Model `ref` syntax

| `ref` | Resolves to |
|-------|-------------|
| `auxiliary.classifier` | `auxiliary.classifier` block |
| `custom_providers.gemma-4-26B` | `custom_providers[]` entry with matching `name` |

Inline `{ provider, model, base_url }` is also supported.

## Related

- [Provider routing](/docs/user-guide/features/provider-routing) — OpenRouter upstream preferences
- [Plugins](/docs/user-guide/features/plugins) — plugin discovery
- [Hooks](/docs/user-guide/features/hooks) — `pre_model_resolve` contract
