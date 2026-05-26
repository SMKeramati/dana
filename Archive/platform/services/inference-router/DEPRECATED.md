# DEPRECATED — inference-router

This service is superseded by `platform/services/inference-gateway`.

**What it did:** Load-balanced requests across multiple `inference-worker` instances (SGLang/KTransformers).

**Why deprecated:** The inference-gateway provides the same routing capability, plus engine-switching via the adapter pattern. It also integrates cleanly with `dana-engine`'s expert-aware batching, which makes external load balancing redundant for concurrent requests.

**Migration:** Replace `inference-router` → `inference-gateway` in your docker-compose / K8s config. Set `ENGINE=dana` or `ENGINE=sglang`.

**Keep this folder?** Yes — as reference. The queue manager and load balancer logic may be useful if you run multiple `dana-engine` instances (future horizontal scaling).
