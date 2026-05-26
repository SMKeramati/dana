#!/usr/bin/env python3
"""Seed Focalboard with Dana project tasks.

Creates a board with Done/In Progress/To Do columns and populates it
with the project's implementation status.

Usage:
    python scripts/seed-project-board.py [--url http://localhost:8080]
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Board definition
# ---------------------------------------------------------------------------

BOARD_TITLE = "Dana AI Platform - Roadmap"

COLUMNS = ["انجام‌شده ✅", "در حال انجام 🔄", "برنامه‌ریزی شده 📋", "بکلاگ 🗂️"]

CARDS: dict[str, list[dict]] = {
    "انجام‌شده ✅": [
        {"title": "Monorepo Foundation", "desc": "Docker Compose, Makefile, CI/CD, pyproject.toml, .env setup"},
        {"title": "dana-common Library", "desc": "Custom HMAC-SHA512 auth, AES-256-GCM encryption, structured logger, sliding-window rate limiter, RabbitMQ messaging"},
        {"title": "Auth Service", "desc": "Custom token engine, API key gen (dk-xxx format), key rotation with grace period, PostgreSQL models"},
        {"title": "API Gateway", "desc": "FastAPI + OpenAI-compatible /v1/chat/completions, SSE streaming, tier-aware rate limiting, auth middleware"},
        {"title": "Inference Router", "desc": "Weighted round-robin load balancer, priority scheduler (5 tiers), SSE bridge, latency tracker (P50/P95/P99)"},
        {"title": "Inference Worker", "desc": "Speculative decoding engine, MoE expert offloading, hybrid KV cache, continuous batching, memory pool"},
        {"title": "Prompt Injection Detector", "desc": "7-pattern regex engine + heuristic scoring (entropy, role markers, instruction density). 5 threat levels."},
        {"title": "Benchmark Runner", "desc": "HumanEval/MBPP auto-evaluation with unbiased pass@k estimator"},
        {"title": "Quality Monitor", "desc": "Bigram repetition, Shannon token entropy, coherence heuristic, degradation alerting"},
        {"title": "Billing Service", "desc": "Token counting (model-specific char ratios), usage aggregation (hourly/daily/monthly), dynamic pricing with tier+volume discounts"},
        {"title": "Analytics Service", "desc": "Z-score + EWMA anomaly detection, usage pipeline, cost analyzer, RabbitMQ consumer"},
        {"title": "Model Registry", "desc": "Version management, health checker, A/B testing with statistical significance (two-proportion Z-test)"},
        {"title": "Python SDK (dana-sdk)", "desc": "OpenAI-compatible client, streaming support, context manager"},
        {"title": "Persian Landing Page", "desc": "Next.js 15, RTL, Vazirmatn font, hero section, features, pricing, playground, dashboard"},
        {"title": "Persian API Docs", "desc": "MDX documentation: getting started, authentication, chat completions"},
        {"title": "Kubernetes Manifests", "desc": "Namespace, ingress (TLS), API gateway deployment + HPA, GPU inference worker deployment"},
        {"title": "Ubuntu 24.04 Setup Script", "desc": "A-Z server setup: Docker, NVIDIA toolkit, Node.js, Python, firewall, all services"},
        {"title": "Grafana Uptime Monitoring", "desc": "Blackbox Exporter probes all /health endpoints, uptime % dashboard, alert rules"},
        {"title": "ClickHouse Analytics DB", "desc": "5 event tables + 2 materialized views for daily/hourly pre-aggregation"},
        {"title": "Metabase Integration", "desc": "Product analytics UI connected to ClickHouse via community driver"},
        {"title": "Focalboard Project Board", "desc": "This board! Task tracking with done/todo/backlog columns"},
    ],
    "در حال انجام 🔄": [
        {"title": "ClickHouse Sink Integration", "desc": "Wiring analytics-service to write events to ClickHouse in production"},
    ],
    "برنامه‌ریزی شده 📋": [
        {"title": "ZarinPal Payment Gateway", "desc": "Replace payment stub with real ZarinPal/IDPay integration. See services/billing-service/src/payment/gateway.py"},
        {"title": "Production GPU Deployment", "desc": "Deploy Qwen3-235B-MoE on 2x A100 80GB with KTransformers + SGLang. Configure worker count based on memory budget."},
        {"title": "Actual Model Loading", "desc": "Replace mock model_loader.py with real Qwen3 MoE weight loading via KTransformers"},
        {"title": "SMTP Email Notifications", "desc": "Configure SMTP for registration confirmation, quota warnings, invoice delivery"},
        {"title": "Custom Domain + TLS", "desc": "Configure api.dana.ir and dana.ir with Let's Encrypt or Cloudflare certificates"},
        {"title": "Metabase Dashboard Setup", "desc": "Create standard product dashboards: DAU, revenue, model performance, user funnel"},
        {"title": "Kubernetes Production Deploy", "desc": "Deploy full stack to K8s cluster with GPU node selector, HPA, pod disruption budgets"},
    ],
    "بکلاگ 🗂️": [
        {"title": "Multi-Region Support", "desc": "Geo-distributed inference workers for lower latency across Iran"},
        {"title": "Fine-tuning API", "desc": "Allow enterprise customers to fine-tune models on their data"},
        {"title": "Function Calling / Tools", "desc": "Implement OpenAI-compatible function calling in the inference pipeline"},
        {"title": "Embedding Models API", "desc": "Add text embedding models alongside chat models"},
        {"title": "Batch Inference API", "desc": "Async batch processing for bulk requests at lower cost"},
        {"title": "User Dashboard Analytics", "desc": "Expose usage analytics to end-users in their dashboard"},
        {"title": "Webhook Notifications", "desc": "Alert users when quota is low, model is degraded, etc."},
        {"title": "Model Marketplace", "desc": "Let third-party model providers register and offer models via Dana's infrastructure"},
        {"title": "Persian LLM Fine-tune", "desc": "Fine-tune Qwen3 on Persian data to improve Farsi quality — major Daneshbonyan differentiator"},
        {"title": "SOC 2 / ISO 27001", "desc": "Security compliance certification for enterprise customers"},
        {"title": "Mobile SDK", "desc": "iOS/Android SDK wrapping dana-sdk for mobile developers"},
    ],
}


# ---------------------------------------------------------------------------
# Focalboard API helpers
# ---------------------------------------------------------------------------


def _request(
    base_url: str,
    path: str,
    method: str = "GET",
    body: dict | None = None,
    token: str = "",
) -> dict | list:
    url = base_url.rstrip("/") + path
    data = json.dumps(body).encode() if body else None
    req = Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urlopen(req, timeout=15) as resp:
        raw = resp.read()
        return json.loads(raw) if raw else {}


def wait_for_focalboard(base_url: str, max_seconds: int = 60) -> bool:
    print(f"Waiting for Focalboard at {base_url} ...")
    for _ in range(max_seconds):
        try:
            _request(base_url, "/api/v2/teams")
            return True
        except Exception:
            time.sleep(1)
    return False


def register_and_login(base_url: str) -> str:
    username = "dana_admin"
    password = "D4n4_B04rd_2026!"
    email = "admin@dana.ir"

    try:
        _request(base_url, "/api/v2/register", method="POST", body={
            "username": username,
            "password": password,
            "email": email,
        })
        print(f"  Registered user: {username}")
    except Exception:
        pass  # user already exists

    result = _request(base_url, "/api/v2/login", method="POST", body={
        "type": "normal",
        "username": username,
        "password": password,
    })
    token = result.get("token", "")  # type: ignore[union-attr]
    print(f"  Logged in, token: {token[:20]}...")
    return token


def get_or_create_team(base_url: str, token: str) -> str:
    teams = _request(base_url, "/api/v2/teams", token=token)
    if teams:
        team_id = teams[0]["id"]  # type: ignore[index]
        print(f"  Using team: {team_id}")
        return team_id
    raise RuntimeError("No team found")


def create_board(base_url: str, token: str, team_id: str) -> tuple[str, dict[str, str]]:
    """Create board with columns; return (board_id, {column_title: column_id})."""
    board_id = str(uuid.uuid4())

    # Build property schema for a "Status" select property
    prop_id = "status_prop"
    options = []
    option_ids: dict[str, str] = {}
    for col in COLUMNS:
        oid = str(uuid.uuid4())
        option_ids[col] = oid
        options.append({"id": oid, "value": col, "color": "propColorDefault"})

    board_body = {
        "id": board_id,
        "teamId": team_id,
        "type": "board",
        "title": BOARD_TITLE,
        "properties": {},
        "cardProperties": [
            {
                "id": prop_id,
                "name": "Status",
                "type": "select",
                "options": options,
            }
        ],
    }
    result = _request(base_url, "/api/v2/boards", method="POST", body=board_body, token=token)
    created_id = result.get("id", board_id)  # type: ignore[union-attr]
    print(f"  Created board: {BOARD_TITLE} ({created_id})")
    return created_id, option_ids, prop_id  # type: ignore[return-value]


def create_card(
    base_url: str,
    token: str,
    board_id: str,
    title: str,
    desc: str,
    prop_id: str,
    option_id: str,
) -> None:
    card_id = str(uuid.uuid4())
    card = {
        "id": card_id,
        "boardId": board_id,
        "type": "card",
        "title": title,
        "fields": {
            "icon": "",
            "description": desc,
            "properties": {prop_id: option_id},
            "contentOrder": [],
        },
    }
    _request(base_url, f"/api/v2/boards/{board_id}/blocks", method="POST", body=[card], token=token)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed Focalboard with Dana project tasks")
    parser.add_argument("--url", default="http://localhost:8080", help="Focalboard base URL")
    args = parser.parse_args()
    base_url: str = args.url

    print("=== Dana Project Board Seeder ===")

    if not wait_for_focalboard(base_url):
        print("ERROR: Focalboard not reachable. Is it running?")
        return

    token = register_and_login(base_url)
    team_id = get_or_create_team(base_url, token)
    board_id, option_ids, prop_id = create_board(base_url, token, team_id)

    total = 0
    for column, cards in CARDS.items():
        option_id = option_ids[column]
        for card in cards:
            create_card(base_url, token, board_id, card["title"], card["desc"], prop_id, option_id)
            total += 1

    print(f"\nDone! Created {total} cards across {len(COLUMNS)} columns.")
    print(f"Open: {base_url}")
    print("\nCredentials:")
    print("  Username: dana_admin")
    print("  Password: D4n4_B04rd_2026!")


if __name__ == "__main__":
    main()
