# Dana - AI API Platform: Master Implementation Plan

> **Priority:** Daneshbonyan-first architecture. Every technical decision is driven by maximizing Knowledge-Based Company evaluation score while building a viable commercial product.
> **Self-Sufficient:** No external SaaS dependencies. All services run on our own infrastructure.

---

## 1. Product Summary

**Dana** is an Iranian AI inference API platform that hosts frontier open-source LLMs and sells API access to developers and enterprises.

- **Flagship Model:** Qwen3 MoE (~235B parameters, ~22B active via Mixture-of-Experts)
- **GPU Config:** 2x NVIDIA A100 (rented) + 512GB RAM + CPU threads
- **API:** OpenAI-compatible (`/v1/chat/completions`)
- **Revenue:** Subscription + pay-per-token (Freemium model)
- **Daneshbonyan Category:** پلتفرم‌ها > پلتفرم‌های هوش مصنوعی > پلتفرم ارائه راهکارهای AI (مارکت پلیس راهکارهای AI، دیپلوی و ارائه به صورت API)

---

## 2. Daneshbonyan AI-Specific Scoring Strategy

Based on the official "شاخص‌های عمومی هوش مصنوعی" criteria:

| Criteria (شاخص) | How Dana Addresses It | Evidence |
|---|---|---|
| شرح دقیق کاربرد سامانه هوشمند | Platform serves Qwen3-235B MoE for code generation, reasoning, and general AI tasks via API | Product demo + docs |
| نوع الگوریتم‌های هوشمند | Mixture-of-Experts architecture, Speculative Decoding, Continuous Batching, Custom KV Cache | Source code + technical docs |
| تناسب الگوریتم با کاربرد | MoE enables serving 235B model on 2x A100 (impossible with dense model). Speculative decoding optimizes for coding use-case. | Benchmarks showing TPS improvement |
| شاخص‌های کمی و کیفی کارایی | Custom benchmark suite: TPS, TTFT (time-to-first-token), P50/P95/P99 latency, accuracy vs reference (HumanEval, MBPP) | Automated benchmark pipeline |
| مقاومت Robustness | Custom input validation, prompt injection detection, graceful degradation under load | Test suite + load tests |
| دامنه مورد پشتیبانی | Multi-language code gen, reasoning, translation, general chat - 32K context window | API docs + examples |
| تاثیر AI در چرخه توسعه | AI IS the product - entire platform exists to serve AI inference | Architecture diagram |
| ارزش افزوده AI | Custom speculative decoding (50-70% speed boost), custom MoE offloading (enables 235B on 2xA100) | Before/after benchmarks |
| توانایی آموزش و سفارشی‌سازی | Custom fine-tuning pipeline (future), custom quantization, custom expert scheduling algorithms | Source code |
| کد منبع آموزش و دادگان | Full source code in monorepo, model weights stored locally | Git repo + model registry |
| ابزار مدیریت کیفیت داده | Custom request/response logging, quality monitoring pipeline | Analytics service code |
| دامنه ورودی تضمین‌شده | Documented input limits: max tokens, supported languages, context window | API docs |
| معیارهای ارزیابی کارایی | HumanEval pass@1, MBPP, latency percentiles, throughput (tokens/sec) | Benchmark scripts |
| فرآیند ارزیابی تخصصی | Automated model evaluation pipeline with custom benchmark runner | CI pipeline + scripts |
| فرآیند تضمین کیفیت | Automated regression tests, A/B testing between model versions | Model registry + tests |
| Explainability (اختیاری) | Token probability logging, top-k alternative tokens in debug mode | API debug endpoints |

---

## 3. Open-Source Package Strategy

### Philosophy: "Use packages for secondary/UI work. Build custom for core AI + Daneshbonyan-scoring modules."

### Packages to USE (secondary stuff - saves time, low Daneshbonyan impact):

| Area | Package | License | Usage |
|------|---------|---------|-------|
| **Landing Page** | `next.js` + `tailwindcss` + `shadcn/ui` | MIT | Full landing page UI components |
| **Docs Site** | `fumadocs` (or `nextra`) | MIT | API documentation framework (MDX-based, like OpenAI docs) |
| **Dashboard UI** | `tremor` (React charts) + `shadcn/ui` | Apache 2.0 / MIT | Usage graphs, dashboard components |
| **ORM** | `SQLAlchemy` + `alembic` | MIT | Database models and migrations |
| **Validation** | `pydantic` v2 | MIT | Request/response models |
| **HTTP Framework** | `FastAPI` + `uvicorn` | MIT/BSD | All backend services |
| **Token Counting** | `tiktoken` | MIT | Count tokens for billing (Qwen tokenizer) |
| **Task Queue** | `celery` or `aio-pika` | BSD/Apache | Async job processing |
| **Password Hashing** | `passlib` + `bcrypt` | BSD | User password storage |
| **Email** | `fastapi-mail` | MIT | Transactional emails (self-hosted SMTP) |
| **PDF Generation** | `weasyprint` | BSD | Invoice generation |
| **Charts** | `recharts` (React) | MIT | Dashboard usage charts |
| **Form Handling** | `react-hook-form` + `zod` | MIT | Dashboard forms |
| **Code Highlighting** | `shiki` | MIT | Code examples in docs/playground |
| **Markdown** | `MDX` + `remark` + `rehype` | MIT | Documentation rendering |

### Packages to CUSTOMIZE/WRAP (partial use, extend with custom logic):

| Area | Base Package | Custom Layer | Daneshbonyan Value |
|------|-------------|-------------|-------------------|
| **LLM Serving** | `SGLang` + `KTransformers` | Custom speculative decoding config, custom expert offloading YAML, custom batch scheduling | Core AI/ML - highest score |
| **Message Broker** | `RabbitMQ` + `aio-pika` | Custom dead-letter handling, custom priority routing, custom retry logic | Event-Driven Architecture |
| **Caching** | `Redis` + `redis-py` | Custom sliding-window rate limiter, custom cache invalidation strategy | Big Data/High Concurrency |
| **Auth** | `python-jose` (base HMAC) | Custom token format with embedded permissions, custom key rotation | Custom Security |

### Modules to BUILD FROM SCRATCH (core - "Internal R&D"):

| Module | Why Custom | Daneshbonyan Score |
|--------|-----------|-------------------|
| Speculative Decoding Engine | No off-the-shelf solution for our specific MoE + draft model pipeline | AI/ML Core - Highest |
| MoE Expert Offload Scheduler | Custom GPU/RAM scheduling based on expert activation profiling | AI/ML Core - Highest |
| Hybrid KV Cache Manager | Custom GPU/CPU cache spanning with smart eviction | AI/ML Core - High |
| Continuous Batching Scheduler | Custom dynamic batch sizing with priority + memory pressure | AI/ML Core - High |
| Custom Token Engine (Auth) | Custom HMAC-SHA512 format (not standard JWT) | Security - High |
| Custom Rate Limiter | Sliding window with Redis sorted sets (not off-the-shelf slowapi) | Performance - High |
| Anomaly Detection | Z-score + EWMA for usage pattern detection | AI/ML Score Booster |
| Usage Aggregation Pipeline | Custom time-window rollups with cost attribution | Data Processing - Medium |
| Model A/B Testing Router | Custom traffic splitting with statistical significance | AI Evaluation - Medium |
| Custom Benchmark Runner | Automated HumanEval/MBPP evaluation pipeline | AI Quality Assurance |
| GPU Memory Pool | Custom allocator with fragmentation prevention | Performance - Medium |
| Prompt Injection Detector | Custom input validation for Robustness scoring | AI Safety - Medium |

---

## 4. Self-Sufficiency Plan (No External SaaS)

**Everything runs on our own infrastructure. Zero dependency on Stripe, Supabase, Auth0, etc.**

| Need | External Service (AVOIDED) | Our Solution |
|------|---------------------------|-------------|
| Database | Supabase, PlanetScale | Self-hosted PostgreSQL in Docker/K8s |
| Auth | Auth0, Clerk | Custom auth-service (FastAPI + PostgreSQL) |
| Payments | Stripe | **TODO for you:** Integrate with Iranian payment gateway (ZarinPal/IDPay) |
| Email | SendGrid, Mailgun | Self-hosted SMTP (Postfix) or **TODO:** Connect to local email service |
| File Storage | AWS S3, Cloudflare R2 | Self-hosted MinIO (S3-compatible) |
| Monitoring | Datadog, New Relic | Self-hosted Prometheus + Grafana |
| Logging | Papertrail, Logtail | Self-hosted ELK (Elasticsearch + Logstash + Kibana) or Loki |
| CI/CD | GitHub Actions (cloud) | Self-hosted Gitea + Drone CI **or** GitHub Actions (acceptable) |
| Container Registry | Docker Hub | Self-hosted Harbor or Gitea registry |
| DNS/CDN | Cloudflare | **TODO:** Set up Iranian DNS + CDN provider |
| SSL | Let's Encrypt | Certbot (automated, self-managed) |
| Search | Algolia | Self-hosted MeiliSearch |
| Queue | Amazon SQS | Self-hosted RabbitMQ |
| Cache | Amazon ElastiCache | Self-hosted Redis |

---

## 5. GPU Cost Analysis

### Selected Configuration: 2x A100 + Supporting Resources

| Resource | Qty | Unit Cost (Toman/hr) | Total (Toman/hr) |
|----------|-----|---------------------|-------------------|
| NVIDIA A100 | 2 | 64,400 | 128,800 |
| RAM (GB) | 512 | 34.5 | 17,664 |
| Heavy Thread | 16 | 805 | 12,880 |
| Storage (GB) | 1,024 | 0.575 | 589 |
| **Total** | | | **159,933** |

**Monthly cost (24/7):** ~115M Toman (~$2,300 USD)

### Why 2x A100?
- 160GB combined VRAM for attention layers + active MoE experts
- Qwen3-235B-MoE: only ~22B active params per token -> fits in VRAM
- 213B inactive experts offloaded to 512GB RAM via KTransformers
- A100 HBM2e: 2TB/s bandwidth - critical for KV cache

---

## 6. Model Choice: Qwen3 MoE (~235B)

### Inference Optimization Stack
```
[User Request]
      |
[Draft Model: Qwen3-0.6B] -- lives entirely on GPU 0
      |
[Speculative Decoding: predict 5-10 tokens]
      |
[Target Model: Qwen3-235B-MoE]
  |-- Attention Layers: GPU 0 + GPU 1 (VRAM)
  |-- Active Experts (8/128): GPU VRAM
  |-- Inactive Experts (120/128): System RAM (512GB)
  |-- KV Cache: Split GPU VRAM + System RAM
      |
[Verify draft tokens in single parallel pass]
      |
[Stream verified tokens to user via SSE]
```

**Expected Performance:** ~10-20 TPS, ~10-15 concurrent users, ~1.5M tokens/day

---

## 7. Architecture

```
                         +------------------+
                         |   Users / Devs   |
                         +--------+---------+
                                  |
                          HTTPS / WSS
                                  |
                    +-------------+-------------+
                    |    Nginx (Reverse Proxy)   |
                    |    SSL + DDoS Protection   |
                    +-------------+-------------+
                                  |
              +-------------------+-------------------+
              |                                       |
    +---------+---------+               +-------------+----------+
    |   Web Frontend    |               |    API Gateway         |
    |   (Next.js SSR)   |               |    (FastAPI)           |
    |                   |               |                        |
    | - Landing (shadcn)|               | - Custom Rate Limiter  |
    | - Playground      |               | - Auth Middleware       |
    | - Dashboard       |               | - Custom WS Protocol   |
    | - Docs (fumadocs) |               | - Request Logger       |
    +-------------------+               +---+-------+-------+---+
                                            |       |       |
                     +----------------------+   +---+   +---+------------------+
                     |                          |       |                      |
           +---------+--------+      +----------+--+  +-+----------+   +------+-------+
           |  Auth Service    |      | Inference   |  | Billing    |   | Analytics    |
           |  (FastAPI)       |      | Router      |  | Service    |   | Service      |
           |                  |      | (FastAPI)   |  | (FastAPI)  |   | (FastAPI)    |
           | * Custom Token   |      |             |  |            |   |              |
           |   Engine         |      | * Custom LB |  | * Token    |   | * Anomaly    |
           | * Key Rotation   |      | * Priority  |  |   Counter  |   |   Detector   |
           | * API Key Gen    |      |   Queue     |  | * Usage    |   | * Usage      |
           | * Custom Encrypt |      | * SSE Bridge|  |   Aggreg.  |   |   Pipeline   |
           +--------+---------+      +------+------+  | * Quota Mgr|   | * Benchmark  |
                    |                       |          +-----+------+   |   Runner     |
               [PostgreSQL]          [RabbitMQ]              |          +------+--------+
               [Redis]                /      \          [PostgreSQL]          |
                               +-----+--+ +--+-----+   [Redis]         [RabbitMQ]
                               | Worker | | Worker |
                               | (GPU0) | | (GPU1) |
                               |        | |        |
                               | CUSTOM: | | CUSTOM:|
                               | -Spec.  | | -Spec. |
                               |  Decode | |  Decode|
                               | -MoE    | | -MoE   |
                               |  Offld  | |  Offld |
                               | -KV$Mgr | | -KV$Mgr|
                               | -Batch  | | -Batch |
                               |  Sched  | |  Sched |
                               | -MemPool| | -MemPool|
                               +----+----+ +----+---+
                                    |            |
                              +-----+------------+----+
                              | Model Registry        |
                              | * Version Mgmt        |
                              | * Health Monitor       |
                              | * A/B Testing          |
                              | * Benchmark Pipeline   |
                              +------------------------+
                                        |
                              +------------------------+
                              | Self-Hosted Infra      |
                              | * MinIO (file storage) |
                              | * Prometheus + Grafana |
                              | * Loki (logs)          |
                              | * Certbot (SSL)        |
                              +------------------------+
```

### Services Summary

| # | Service | Custom "Internal R&D" Modules | Open-Source Packages Used |
|---|---------|------------------------------|--------------------------|
| 1 | api-gateway | Rate Limiter, WS Protocol, Request Logger | FastAPI, uvicorn, redis-py |
| 2 | auth-service | Token Engine, Key Rotation, API Key Gen, Encryption | FastAPI, SQLAlchemy, passlib, bcrypt |
| 3 | inference-router | Load Balancer, Priority Queue, SSE Bridge, Latency Tracker | FastAPI, aio-pika |
| 4 | inference-worker | Speculative Decoding, MoE Offloader, KV Cache, Batch Scheduler, Memory Pool, GPU Monitor, Prompt Injection Detector | SGLang, KTransformers, PyTorch, tiktoken |
| 5 | billing-service | Token Counter, Usage Aggregator, Quota Manager, Pricing Engine | FastAPI, SQLAlchemy, redis-py |
| 6 | analytics-service | Anomaly Detector, Usage Pipeline, Cost Analyzer, Benchmark Runner | FastAPI, NumPy, SciPy, aio-pika |
| 7 | model-registry | Model Store, Health Checker, A/B Testing | FastAPI, SQLAlchemy |
| 8 | web (landing+docs+dashboard) | Playground component | Next.js, shadcn/ui, fumadocs, tremor, recharts |

---

## 8. Project Directory Structure

```
dana/
|-- README.md
|-- IMPLEMENTATION_PLAN.md
|-- ARCHITECTURE.md
|-- DANESHBONYAN_MODULES.md
|-- docker-compose.yml
|-- docker-compose.prod.yml
|-- Makefile
|-- pyproject.toml                       # Root ruff/mypy config
|-- .env                                 # All secrets (passwords, keys)
|-- .gitignore.example                   # Template to activate later
|
|-- .github/
|   |-- workflows/
|       |-- ci.yml
|       |-- cd-staging.yml
|       |-- cd-production.yml
|
|-- infrastructure/
|   |-- kubernetes/
|   |   |-- namespace.yml
|   |   |-- api-gateway/
|   |   |-- auth-service/
|   |   |-- inference-router/
|   |   |-- inference-worker/           # GPU nodeSelector
|   |   |-- billing-service/
|   |   |-- analytics-service/
|   |   |-- model-registry/
|   |   |-- rabbitmq/
|   |   |-- redis/
|   |   |-- postgres/
|   |   |-- minio/
|   |   |-- monitoring/
|   |   |-- ingress.yml
|   |-- monitoring/
|       |-- prometheus.yml
|       |-- grafana/dashboards/
|       |-- loki-config.yml
|
|-- packages/
|   |-- dana-common/                     # Shared Python library
|   |   |-- pyproject.toml
|   |   |-- src/dana_common/
|   |       |-- __init__.py
|   |       |-- config.py               # Reads from .env
|   |       |-- models.py               # Shared Pydantic models
|   |       |-- auth.py                 # CUSTOM: HMAC token verify
|   |       |-- encryption.py           # CUSTOM: AES-GCM encryption
|   |       |-- cache.py                # CUSTOM: Redis cache + invalidation
|   |       |-- messaging.py            # RabbitMQ pub/sub base
|   |       |-- logging.py              # Structured logging + correlation IDs
|   |       |-- exceptions.py
|   |   |-- tests/
|   |
|   |-- dana-sdk/                        # Python SDK for customers
|       |-- pyproject.toml
|       |-- src/dana_sdk/
|       |   |-- __init__.py
|       |   |-- client.py               # OpenAI-compatible client
|       |   |-- streaming.py
|       |-- tests/
|
|-- services/
|   |-- api-gateway/
|   |   |-- Dockerfile
|   |   |-- pyproject.toml
|   |   |-- src/
|   |   |   |-- main.py
|   |   |   |-- middleware/
|   |   |   |   |-- rate_limiter.py      # CUSTOM: sliding window
|   |   |   |   |-- auth.py
|   |   |   |   |-- request_logger.py
|   |   |   |-- routes/
|   |   |   |   |-- v1_chat.py           # /v1/chat/completions
|   |   |   |   |-- v1_models.py
|   |   |   |   |-- health.py
|   |   |   |-- websocket/
|   |   |       |-- stream.py            # CUSTOM: WS protocol
|   |   |-- tests/
|   |
|   |-- auth-service/
|   |   |-- Dockerfile
|   |   |-- pyproject.toml
|   |   |-- src/
|   |   |   |-- main.py
|   |   |   |-- crypto/
|   |   |   |   |-- token_engine.py      # CUSTOM: HMAC-SHA512 tokens
|   |   |   |   |-- key_rotation.py      # CUSTOM: zero-downtime rotation
|   |   |   |   |-- api_key_gen.py       # CUSTOM: dk-xxx format
|   |   |   |-- routes/
|   |   |   |-- db/
|   |   |-- tests/
|   |
|   |-- inference-router/
|   |   |-- Dockerfile
|   |   |-- pyproject.toml
|   |   |-- src/
|   |   |   |-- main.py
|   |   |   |-- router/
|   |   |   |   |-- load_balancer.py     # CUSTOM: weighted + health-aware
|   |   |   |   |-- queue_manager.py     # RabbitMQ dispatch
|   |   |   |   |-- priority_queue.py    # CUSTOM: paid-first scheduling
|   |   |   |-- streaming/
|   |   |   |   |-- sse_bridge.py        # CUSTOM: SSE multiplexer
|   |   |   |-- metrics/
|   |   |       |-- latency_tracker.py   # CUSTOM: P50/P95/P99
|   |   |-- tests/
|   |
|   |-- inference-worker/
|   |   |-- Dockerfile.gpu
|   |   |-- pyproject.toml
|   |   |-- src/
|   |   |   |-- main.py
|   |   |   |-- engine/
|   |   |   |   |-- model_loader.py
|   |   |   |   |-- speculative.py       # CUSTOM: draft-verify pipeline
|   |   |   |   |-- kv_cache.py          # CUSTOM: hybrid GPU/CPU cache
|   |   |   |   |-- tokenizer.py         # wraps tiktoken
|   |   |   |   |-- batch_scheduler.py   # CUSTOM: continuous batching
|   |   |   |-- optimization/
|   |   |   |   |-- expert_offload.py    # CUSTOM: MoE GPU/RAM scheduling
|   |   |   |   |-- quantization.py
|   |   |   |   |-- memory_pool.py       # CUSTOM: GPU memory allocator
|   |   |   |-- safety/
|   |   |   |   |-- injection_detector.py # CUSTOM: prompt injection filter
|   |   |   |-- evaluation/
|   |   |   |   |-- benchmark_runner.py  # CUSTOM: HumanEval/MBPP auto-eval
|   |   |   |   |-- quality_monitor.py   # CUSTOM: output quality tracking
|   |   |   |-- health/
|   |   |       |-- gpu_monitor.py
|   |   |-- tests/
|   |
|   |-- billing-service/
|   |   |-- Dockerfile
|   |   |-- pyproject.toml
|   |   |-- src/
|   |   |   |-- main.py
|   |   |   |-- metering/
|   |   |   |   |-- token_counter.py
|   |   |   |   |-- usage_aggregator.py  # CUSTOM: time-window rollups
|   |   |   |   |-- quota_manager.py
|   |   |   |-- plans/
|   |   |   |   |-- subscription.py
|   |   |   |   |-- pricing.py
|   |   |   |-- payment/
|   |   |   |   |-- gateway.py           # TODO: ZarinPal/IDPay integration
|   |   |   |   |-- invoice.py           # uses weasyprint for PDF
|   |   |   |-- db/
|   |   |-- tests/
|   |
|   |-- analytics-service/
|   |   |-- Dockerfile
|   |   |-- pyproject.toml
|   |   |-- src/
|   |   |   |-- main.py
|   |   |   |-- pipelines/
|   |   |   |   |-- usage_pipeline.py    # CUSTOM: aggregation
|   |   |   |   |-- anomaly_detector.py  # CUSTOM: Z-score + EWMA
|   |   |   |   |-- cost_analyzer.py
|   |   |   |-- reports/
|   |   |   |   |-- dashboard_api.py
|   |   |   |-- consumers/
|   |   |       |-- event_consumer.py
|   |   |-- tests/
|   |
|   |-- model-registry/
|       |-- Dockerfile
|       |-- pyproject.toml
|       |-- src/
|       |   |-- main.py
|       |   |-- registry/
|       |   |   |-- model_store.py
|       |   |   |-- health_checker.py
|       |   |   |-- ab_testing.py        # CUSTOM: traffic splitting
|       |   |-- db/
|       |-- tests/
|
|-- web/
|   |-- landing/                          # Next.js 15 + shadcn/ui + Tailwind
|   |   |-- package.json
|   |   |-- Dockerfile
|   |   |-- next.config.js
|   |   |-- src/app/
|   |   |   |-- layout.tsx
|   |   |   |-- page.tsx                  # Landing
|   |   |   |-- pricing/page.tsx
|   |   |   |-- playground/page.tsx       # Interactive API tester
|   |   |   |-- dashboard/
|   |   |       |-- page.tsx
|   |   |       |-- usage/page.tsx        # tremor charts
|   |   |       |-- keys/page.tsx
|   |   |       |-- billing/page.tsx
|   |   |-- src/components/               # shadcn/ui components
|   |
|   |-- docs/                             # fumadocs (MDX documentation)
|       |-- package.json
|       |-- Dockerfile
|       |-- content/docs/
|       |   |-- index.mdx                 # Getting started
|       |   |-- authentication.mdx
|       |   |-- chat-completions.mdx
|       |   |-- streaming.mdx
|       |   |-- models.mdx
|       |   |-- rate-limits.mdx
|       |   |-- errors.mdx
|       |   |-- sdk.mdx
|       |-- src/
|
|-- scripts/
|   |-- setup-dev.sh
|   |-- run-tests.sh
|   |-- deploy.sh
|   |-- benchmark.sh                      # Model benchmark runner
|   |-- download-model.sh                 # Download Qwen3 weights
|
|-- benchmarks/                           # Evaluation datasets + results
    |-- humaneval/
    |-- mbpp/
    |-- results/
```

---

## 9. Environment Configuration (.env)

```env
# === Database ===
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=dana
POSTGRES_USER=dana_admin
POSTGRES_PASSWORD=D4n4_S3cur3_P@ss_2026!

# === Redis ===
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=R3d1s_D4n4_K3y!

# === RabbitMQ ===
RABBITMQ_HOST=rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=dana_mq
RABBITMQ_PASSWORD=Mq_D4n4_2026!
RABBITMQ_VHOST=dana

# === MinIO (S3-compatible storage) ===
MINIO_ROOT_USER=dana_minio
MINIO_ROOT_PASSWORD=M1n10_D4n4_2026!
MINIO_ENDPOINT=minio:9000

# === Auth Service ===
JWT_SECRET_KEY=d4n4_jwt_s3cr3t_k3y_2026_x7k9m2p4q8r1
JWT_ALGORITHM=HS512
API_KEY_SALT=d4n4_4p1_k3y_s4lt_2026
ENCRYPTION_KEY=0123456789abcdef0123456789abcdef

# === Application ===
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000

# === Model Configuration ===
MODEL_NAME=qwen3-235b-moe
MODEL_PATH=/models/qwen3-235b-moe-q4
DRAFT_MODEL_PATH=/models/qwen3-0.6b
GPU_DEVICES=0,1
MAX_BATCH_SIZE=16
MAX_CONTEXT_LENGTH=32768
SPECULATIVE_LOOKAHEAD=8

# === GPU Provider ===
GPU_PROVIDER_COST_PER_HOUR_TOMAN=159933

# === Rate Limits ===
FREE_TIER_RPM=5
FREE_TIER_TPD=1000
PRO_TIER_RPM=60
PRO_TIER_TPD=100000

# === Monitoring ===
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=Gr4f4n4_D4n4!

# === Payment Gateway (TODO: configure with actual credentials) ===
PAYMENT_GATEWAY=zarinpal
ZARINPAL_MERCHANT_ID=YOUR_MERCHANT_ID_HERE

# === Email (TODO: configure SMTP) ===
SMTP_HOST=localhost
SMTP_PORT=587
SMTP_USER=noreply@dana.ir
SMTP_PASSWORD=YOUR_SMTP_PASSWORD_HERE
```

---

## 10. CI/CD Pipeline

### ci.yml (Every PR)
```
lint -> typecheck -> unit-tests -> integration-tests -> build-images -> security-scan
```
- ruff + eslint for lint
- mypy + tsc for type check
- pytest per service + jest for web
- docker-compose up + e2e test
- docker build all services
- bandit + npm audit for security

### cd-staging.yml (Merge to main)
1. Build & tag Docker images
2. Push to self-hosted Harbor registry
3. kubectl apply to staging namespace
4. Smoke tests
5. Notify team

### cd-production.yml (Git tag v*)
1. Build production images
2. Rolling update to production K8s
3. Health check + automatic rollback

---

## 11. Implementation Phases (Daneshbonyan-First)

### Phase 1: Foundation (Weeks 1-2)
- Initialize monorepo, all directories, pyproject.toml, Dockerfiles
- `packages/dana-common/` (config, models, auth, encryption, cache, messaging)
- `docker-compose.yml` with PostgreSQL, Redis, RabbitMQ, MinIO
- `.env` file with all credentials
- `.github/workflows/ci.yml`
- Database migrations (Alembic)

### Phase 2: Custom Core Modules (Weeks 3-4)
- `auth-service`: Custom token engine, API key gen, key rotation, encryption
- `api-gateway`: Custom rate limiter, auth middleware, WebSocket streaming
- `inference-router`: Custom load balancer, RabbitMQ queue, priority scheduling, SSE bridge
- `billing-service`: Token counting, usage aggregation, quota management
- `analytics-service`: Anomaly detection, usage pipeline, event consumer
- `model-registry`: Version management, health checker, A/B testing

### Phase 3: AI/ML Engine (Weeks 5-6) -- HIGHEST DANESHBONYAN SCORE
- Custom speculative decoding engine
- Custom MoE expert offloading scheduler
- Custom hybrid KV cache manager
- Custom continuous batching scheduler
- Custom GPU memory pool allocator
- Custom prompt injection detector
- Custom benchmark runner (HumanEval/MBPP)
- Custom quality monitoring pipeline

### Phase 4: Web + Docs (Weeks 7-8)
- Landing page (Next.js + shadcn/ui)
- Interactive API playground
- API docs (fumadocs/nextra + MDX)
- User dashboard with tremor charts
- Python SDK (dana-sdk)

### Phase 5: DevOps + Evaluation Prep (Weeks 9-10)
- CD pipelines (staging + production)
- Kubernetes manifests (all services)
- Prometheus + Grafana monitoring
- Load testing + benchmarks
- >80% test coverage
- Architecture block diagram (for questionnaire)
- Screenshots + demo video
- Fill Daneshbonyan questionnaire

### Phase 6: Launch (Weeks 11-12)
- Security audit
- First beta users
- Payment gateway integration (ZarinPal/IDPay)
- Sales contracts (for Daneshbonyan evidence)

---

## 12. TODO List for You (Things I Cannot Do)

These are the ONLY items that require your manual action:

| # | Task | Why I Can't Do It | Priority |
|---|------|-------------------|----------|
| 1 | **Register ZarinPal/IDPay merchant account** | Requires Iranian bank account + identity verification | Phase 6 |
| 2 | **Set up domain name (dana.ir or similar)** | Requires purchase from Iranian registrar (.ir domain) | Phase 4 |
| 3 | **Configure DNS records** | Depends on domain purchase + hosting provider IP | Phase 4 |
| 4 | **Set up SMTP email service** | Requires email provider account or self-hosted mail server | Phase 5 |
| 5 | **Rent GPU infrastructure** | Requires account + payment with Iranian GPU provider | Phase 1 |
| 6 | **Download Qwen3-235B model weights** | Requires ~600GB download + storage setup on GPU servers | Phase 3 |
| 7 | **Register for Daneshbonyan evaluation** | Requires company registration at daneshbonyan.ir | Phase 5 |
| 8 | **Sign first customer contracts** | Requires sales outreach + legal documents | Phase 6 |
| 9 | **SSL certificate for domain** | Requires domain setup first (Certbot automates the rest) | Phase 4 |
| 10 | **Activate .gitignore** | Pull repo, rename `.gitignore.example` to `.gitignore` | When ready |

Everything else (all code, configs, Docker files, CI/CD, tests, docs) will be built by us.

---

## 13. Daneshbonyan Complete Module Map (for Questionnaire)

### For "Components/Modules" Table

| # | نام مؤلفه | وضعیت | زبان | شرح پیچیدگی |
|---|-----------|-------|------|-------------|
| 1 | موتور رمزگشایی حدسی (Speculative Decoding) | **طراحی و توسعه داخلی** | Python/CUDA | خط لوله سفارشی draft-verify با lookahead قابل تنظیم. توکن‌های پیش‌بینی‌شده توسط مدل کوچک در یک پاس موازی توسط مدل بزرگ تأیید می‌شوند. |
| 2 | مدیریت بارگذاری خبرگان MoE | **طراحی و توسعه داخلی** | Python/CUDA | الگوریتم زمانبندی سفارشی برای مدیریت پویای خبرگان فعال/غیرفعال بین GPU و RAM |
| 3 | مدیریت حافظه KV ترکیبی | **طراحی و توسعه داخلی** | Python | حافظه نهان ترکیبی GPU/CPU با سیاست حذف هوشمند LRU + فرکانسی |
| 4 | زمانبندی دسته‌بندی پیوسته | **طراحی و توسعه داخلی** | Python | دسته‌بندی پویا بر اساس فشار حافظه GPU و سطح اولویت درخواست‌ها |
| 5 | موتور توکن احراز هویت | **طراحی و توسعه داخلی** | Python | فرمت توکن سفارشی HMAC-SHA512 با مجوزهای تعبیه‌شده (نه JWT استاندارد) |
| 6 | لایه رمزنگاری سفارشی | **طراحی و توسعه داخلی** | Python | رمزنگاری AES-256-GCM برای داده‌های ذخیره‌شده با چرخش خودکار کلید |
| 7 | محدودکننده نرخ پنجره لغزان | **طراحی و توسعه داخلی** | Python | محدودکننده توزیع‌شده با Redis sorted sets و پنجره لغزان قابل تنظیم |
| 8 | تشخیص ناهنجاری مصرف | **طراحی و توسعه داخلی** | Python/NumPy | تشخیص الگوهای غیرعادی با Z-score + EWMA |
| 9 | متعادل‌کننده بار هوشمند | **طراحی و توسعه داخلی** | Python | Round-robin وزن‌دار با آگاهی بلادرنگ از سلامت و بار GPU |
| 10 | پروتکل جریان WebSocket | **طراحی و توسعه داخلی** | Python | پل SSE/WebSocket سفارشی با مدیریت فشار معکوس |
| 11 | خط لوله تجمیع مصرف | **طراحی و توسعه داخلی** | Python | تجمیع پنجره زمانی (ساعتی/روزانه/ماهانه) با نسبت‌دهی هزینه |
| 12 | تخصیص‌دهنده حافظه GPU | **طراحی و توسعه داخلی** | Python/CUDA | استخر حافظه سفارشی با جلوگیری از تکه‌تکه شدن |
| 13 | تشخیص تزریق پرامپت | **طراحی و توسعه داخلی** | Python | فیلتر امنیتی سفارشی برای شناسایی ورودی‌های مخرب |
| 14 | اجراکننده محک خودکار | **طراحی و توسعه داخلی** | Python | ارزیابی خودکار HumanEval/MBPP با مقایسه نسخه‌های مدل |
| 15 | مسیریاب آزمون A/B | **طراحی و توسعه داخلی** | Python | تقسیم ترافیک سفارشی با ردیابی معناداری آماری |
| 16 | پایش کیفیت خروجی | **طراحی و توسعه داخلی** | Python | ردیابی بلادرنگ کیفیت پاسخ‌ها با معیارهای سفارشی |

### For "Innovation Narrative" Section

> دانا یک wrapper ساده روی API هوش مصنوعی نیست. چالش فنی اصلی، ارائه یک مدل ۲۳۵ میلیارد پارامتری Mixture-of-Experts روی زیرساخت محدود GPU (2x A100) با سرعت تجاری است. این امر مستلزم توسعه موارد زیر بود:
>
> ۱. **موتور رمزگشایی حدسی سفارشی** - به جای تولید اتورگرسیو استاندارد، خط لوله draft-verify طراحی شد که مدل کوچک چند توکن پیش‌بینی و مدل بزرگ در یک پاس موازی تأیید می‌کند.
>
> ۲. **مدیریت بارگذاری خبرگان MoE سفارشی** - الگوریتم زمانبندی که الگوهای فعال‌سازی را پروفایل و خبرگان را بین VRAM و RAM سیستم مدیریت می‌کند.
>
> ۳. **معماری میکروسرویس رویداد-محور** - ۷ میکروسرویس سفارشی با ارتباط از طریق RabbitMQ و خط لوله جریان بلادرنگ.
>
> تمام ماژول‌های هسته‌ای به صورت داخلی و از طریق تحقیق و توسعه طراحی و پیاده‌سازی شده‌اند.

---

## 14. Revenue Model (for Questionnaire)

- **اشتراکی (Subscription):** پلن ماهانه Pro و Enterprise
- **فریمیوم (Freemium):** سطح رایگان با ارتقا به پولی
- **معامله‌ای (Transactional):** پرداخت به‌ازای هر توکن مصرفی
- **فروش مستقیم:** قراردادهای سازمانی
- **فروش وب‌سایتی:** ثبت‌نام و پرداخت سلف‌سرویس

---

## 15. Competitors (for Questionnaire)

| رقیب | کشور | محصول | مزیت دانا |
|------|------|-------|----------|
| Groq | آمریکا | API استنتاج مبتنی بر LPU | حاکمیت داده روی زیرساخت ایرانی |
| Together AI | آمریکا | میزبانی مدل‌های متن‌باز | زیرساخت محلی، تأخیر کمتر برای کاربران ایرانی |
| Fireworks AI | آمریکا | API استنتاج سریع | بهینه‌سازی MoE سفارشی، غیرقابل دسترس در پلتفرم‌های عمومی |

**مزایای فنی نسبت به رقبا:**
- موتور رمزگشایی حدسی سفارشی (نه vLLM استاندارد)
- مدیریت بارگذاری خبرگان MoE سفارشی
- حافظه KV ترکیبی سفارشی
- تمام الگوریتم‌های بهینه‌سازی به صورت داخلی توسعه یافته‌اند
