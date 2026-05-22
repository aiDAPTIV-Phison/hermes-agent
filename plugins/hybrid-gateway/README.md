# Hybrid Gateway (Hermes plugin)

Per-turn routing between **classifier**, **edge**, and **cloud** models.

## Enable

```yaml
# ~/.hermes/config.yaml
hybrid_gateway:
  enabled: true
```

See `hybrid_spec/hermes-config.example.yaml` and `hybrid_spec/hybrid_gateway_info_CN.md` for full documentation.

## Bundled with hermes-agent

Discovered automatically from `plugins/hybrid-gateway/` (no copy to `~/.hermes/plugins/` required).
