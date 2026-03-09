# Dana AI Platform - TODO

> Last updated: 2026-03-07
> Status: Pre-production. Core services built, web frontend complete, needs deployment prep.

---

## Legend
- [x] Done
- [ ] Not done
- [~] Partially done

---

## 1. Backend Services

### dana-common (shared library)
- [x] Logging, config, models, health check
- [x] Token engine (HMAC-SHA512 custom tokens)
- [x] Rate limiter, circuit breaker
- [x] Tests passing

### api-gateway (port 8000)
- [x] FastAPI app with routing
- [x] Dockerfile
- [x] Tests

### auth-service (port 8001)
- [x] User registration and login
- [x] Custom HMAC-SHA512 token engine (not standard JWT)
- [x] API key generation (create, list, delete)
- [x] Profile endpoints (GET /auth/me, POST /auth/refresh)
- [x] SQLAlchemy async + asyncpg (PostgreSQL)
- [x] Dockerfile
- [x] Tests
- [ ] Email verification flow (SMTP not configured)
- [ ] Password reset endpoint
- [ ] OAuth providers (Google, GitHub)

### inference-router (port 8002)
- [x] Request routing to inference workers
- [x] RabbitMQ integration
- [x] Dockerfile
- [x] Tests

### inference-worker (GPU service)
- [x] Model loader (vLLM/HF Transformers)
- [x] Speculative decoding engine
- [x] KV cache management
- [x] Batch scheduler
- [x] Expert offloading for MoE models
- [x] Memory pool optimizer
- [x] Quantization support (GPTQ, AWQ)
- [x] Prompt injection detector
- [x] Tests
- [ ] Dockerfile (needs NVIDIA base image + vLLM)
- [ ] Not in docker-compose.yml (requires GPU host)

### billing-service (port 8003)
- [x] Plan catalogue (free, pro, enterprise)
- [x] Usage aggregator (hourly/daily/monthly rollups)
- [x] ZarinPal payment gateway v4 integration
- [x] Dockerfile
- [x] Tests
- [~] Runtime bugs fixed (plan catalogue, usage aggregator, field names)
- [ ] Actual ZarinPal merchant ID (placeholder in .env)
- [ ] Invoice generation
- [ ] Webhook for payment confirmation

### analytics-service (port 8004)
- [x] Event ingestion and processing
- [x] ClickHouse integration
- [x] Dockerfile
- [x] Tests

### model-registry (port 8005)
- [x] Model metadata CRUD
- [x] Dockerfile
- [x] Tests

### finetuning-service (port 8006)
- [x] LoRA/QLoRA trainer with HF Transformers + PEFT
- [x] Data loader (Alpaca, ShareGPT, JSONL formats)
- [x] Data validator (Persian text normalization, dedup, toxicity)
- [x] Model evaluator (BLEU, ROUGE-L, coherence, Persian quality)
- [x] Checkpoint manager (MinIO storage)
- [x] FastAPI health endpoint
- [x] Tests
- [ ] Dockerfile (needs GPU base image)
- [ ] Not in docker-compose.yml
- [ ] API endpoints for job submission/monitoring (only /health and /status exist)

---

## 2. Web Frontend (Next.js 15)

### Landing site
- [x] Full UX redesign with dark mode
- [x] Design system (Button, Card, Badge, Input, Skeleton)
- [x] Landing page (hero, features, stats, pricing cards, CTA)
- [x] Pricing page (comparison table, FAQ)
- [x] Playground page (chat UI, model settings)
- [x] Login / Register pages
- [x] Documentation page (/docs - quickstart, auth, chat, models, streaming, errors)
- [x] Sticky header with mobile hamburger
- [x] Footer
- [x] Dockerfile (multi-stage, Next.js standalone)

### Dashboard
- [x] Sidebar navigation (desktop)
- [x] Mobile bottom navigation bar
- [x] Dashboard home (welcome banner, stat cards, chart, onboarding checklist)
- [x] API Keys page (create, list, delete - wired to backend via hooks)
- [x] Usage page (period selector, area/bar charts - wired to backend)
- [x] Billing page (current plan, usage bars, payment method - wired to backend)

### API Integration
- [x] Next.js API routes proxying to backend services
- [x] httpOnly cookie auth (dana_token)
- [x] React hooks: useAuth, useApiKeys, useUsage, useBilling
- [x] proxyFetch helper for server-side service communication
- [ ] Playground page: wire to actual inference API (currently no backend call)
- [ ] Real-time usage updates (WebSocket or polling)

---

## 3. Infrastructure

### Docker / Docker Compose
- [x] docker-compose.yml with all infra + 6 backend services + web
- [x] PostgreSQL, Redis, RabbitMQ, MinIO
- [x] Prometheus + Grafana + Blackbox Exporter
- [x] ClickHouse + Metabase
- [x] Focalboard (project management)
- [x] Landing service with env vars for AUTH/BILLING service URLs
- [ ] inference-worker not in compose (needs GPU host)
- [ ] finetuning-service not in compose (needs GPU host)
- [ ] Nginx reverse proxy container (no nginx config exists)
- [ ] SSL/TLS termination

### Kubernetes
- [x] Namespace, ingress manifests
- [x] Deployment manifests for all services
- [x] StatefulSets for PostgreSQL, Redis, RabbitMQ, MinIO
- [x] Monitoring stack manifests
- [ ] GPU node pool configuration
- [ ] HPA (autoscaling) for inference workers
- [ ] Secrets management (sealed-secrets or external-secrets)

### Monitoring
- [x] Prometheus config + alerts
- [x] Blackbox exporter config
- [x] Grafana provisioning + dashboards

### Terraform
- [ ] Empty directory - no IaC written yet

---

## 4. CI/CD

- [x] GitHub Actions workflow: lint, typecheck, test, build, security
- [x] Frontend: lint, typecheck, build jobs
- [x] Docker image build verification
- [x] Bandit security scanning
- [ ] Deployment job (staging/production)
- [ ] Container registry push (ghcr.io or Docker Hub)
- [ ] Database migration step
- [ ] E2E tests

---

## 5. Scripts

- [x] `scripts/setup-dev.sh` - Install all Python packages for development
- [x] `scripts/setup-server.sh` - Full server setup (Docker, Node, Python, firewall, compose)
- [x] `scripts/finetune/train_lora.py` - CLI for LoRA fine-tuning
- [x] `scripts/finetune/evaluate.py` - CLI for model evaluation
- [ ] `scripts/finetune.sh` - Complete fine-tuning shell script (setup GPU env, download model, run training)
- [ ] `scripts/download-model.sh` - Download Qwen3-235B weights
- [x] `scripts/seed-project-board.py` - Focalboard seed data
- [x] `Makefile` - up, down, test, lint, typecheck, build, clean, setup-dev

---

## 6. Configuration & Secrets

- [x] `.env` file with all service configs
- [ ] `.env.example` with placeholder values (current .env has real dev passwords)
- [ ] Separate `.env.production` template
- [ ] SMTP credentials (placeholder)
- [ ] ZarinPal merchant ID (placeholder)

---

## 7. Pre-Launch Checklist

### Server Setup (run `setup-server.sh`)
- [ ] Provision Ubuntu 24.04 server with GPU (NVIDIA A100/H100)
- [ ] Run `sudo ./scripts/setup-server.sh`
- [ ] Configure `.env` with production values
- [ ] Set up domain + DNS (dana.ir)
- [ ] Set up SSL with certbot
- [ ] Configure ZarinPal production merchant ID
- [ ] Configure SMTP for transactional emails

### Model Setup
- [ ] Download Qwen3-235B-A22B weights (or quantized GPTQ/AWQ version)
- [ ] Download draft model (Qwen3-0.6B) for speculative decoding
- [ ] Place at paths from .env: /models/qwen3-235b-moe-q4, /models/qwen3-0.6b
- [ ] Start inference-worker manually on GPU node (not in compose)

### Fine-tuning Setup
- [ ] Prepare Persian instruction dataset (Alpaca/ShareGPT/JSONL format)
- [ ] Run fine-tuning with `scripts/finetune.sh`
- [ ] Evaluate adapter quality
- [ ] Deploy adapter to inference-worker

### Testing
- [ ] Run full test suite: `make test`
- [ ] Manual smoke test of all API endpoints
- [ ] Test payment flow end-to-end (ZarinPal sandbox)
- [ ] Test login/register/dashboard flow
- [ ] Load test inference endpoint

---

## 8. Known Issues

1. Billing service had bugs (fixed): wrong function names, wrong field names in UsageRecord
2. framer-motion removed from frontend (caused blank pages in SSR) - replaced with CSS animations
3. Playground page not wired to actual inference API
4. .env committed to repo with dev passwords (should be .env.example only)
5. inference-worker and finetuning-service have no Dockerfiles
6. Terraform directory is empty
