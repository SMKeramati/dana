-- Dana Analytics - ClickHouse Schema
-- Daneshbonyan: Internal R&D - Custom Event Storage Architecture

CREATE DATABASE IF NOT EXISTS dana_analytics;

-- ============================================================
-- API Request Events (high-volume, raw events)
-- ============================================================
CREATE TABLE IF NOT EXISTS dana_analytics.api_requests
(
    timestamp        DateTime64(3)        DEFAULT now(),
    date             Date                 DEFAULT toDate(timestamp),
    request_id       String,
    user_id          String,
    api_key_prefix   String,
    endpoint         LowCardinality(String),
    method           LowCardinality(String),
    status_code      UInt16,
    response_time_ms UInt32,
    model            LowCardinality(String),
    prompt_tokens    UInt32,
    completion_tokens UInt32,
    total_tokens     UInt32,
    tier             LowCardinality(String),  -- free/pro/enterprise
    country_code     LowCardinality(String),
    error_code       LowCardinality(String),
    streaming        UInt8                DEFAULT 0,
    cached           UInt8                DEFAULT 0
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (date, model, user_id, timestamp)
TTL timestamp + INTERVAL 2 YEAR
SETTINGS index_granularity = 8192;

-- ============================================================
-- User Lifecycle Events
-- ============================================================
CREATE TABLE IF NOT EXISTS dana_analytics.user_events
(
    timestamp    DateTime64(3)        DEFAULT now(),
    date         Date                 DEFAULT toDate(timestamp),
    user_id      String,
    event_type   LowCardinality(String),
    -- event_type: registered | login | api_key_created | tier_upgraded | tier_downgraded | password_changed | api_key_deleted
    properties   String,  -- JSON blob
    ip_address   String,
    user_agent   String
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (date, user_id, event_type, timestamp)
TTL timestamp + INTERVAL 3 YEAR;

-- ============================================================
-- Inference Performance Events
-- ============================================================
CREATE TABLE IF NOT EXISTS dana_analytics.inference_events
(
    timestamp              DateTime64(3)        DEFAULT now(),
    date                   Date                 DEFAULT toDate(timestamp),
    job_id                 String,
    user_id                String,
    model                  LowCardinality(String),
    prompt_tokens          UInt32,
    completion_tokens      UInt32,
    time_to_first_token_ms UInt32,
    total_latency_ms       UInt32,
    tokens_per_second      Float32,
    speculative_accepted   UInt32,
    speculative_total      UInt32,
    batch_size             UInt8,
    gpu_memory_used_gb     Float32,
    worker_id              String,
    cache_hit              UInt8                DEFAULT 0
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (date, model, timestamp)
TTL timestamp + INTERVAL 1 YEAR;

-- ============================================================
-- Billing Events
-- ============================================================
CREATE TABLE IF NOT EXISTS dana_analytics.billing_events
(
    timestamp      DateTime64(3)        DEFAULT now(),
    date           Date                 DEFAULT toDate(timestamp),
    user_id        String,
    org_id         String,
    event_type     LowCardinality(String),
    -- event_type: charge | quota_exceeded | plan_upgraded | invoice_generated
    amount_cents   UInt32,
    total_tokens   UInt32,
    model          LowCardinality(String),
    tier           LowCardinality(String),
    invoice_id     String
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (date, user_id, event_type, timestamp)
TTL timestamp + INTERVAL 7 YEAR;

-- ============================================================
-- Error Events
-- ============================================================
CREATE TABLE IF NOT EXISTS dana_analytics.error_events
(
    timestamp    DateTime64(3)        DEFAULT now(),
    date         Date                 DEFAULT toDate(timestamp),
    service      LowCardinality(String),
    error_type   LowCardinality(String),
    error_code   LowCardinality(String),
    message      String,
    user_id      String,
    request_id   String,
    stack_trace  String
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(timestamp)
ORDER BY (date, service, error_type, timestamp)
TTL timestamp + INTERVAL 6 MONTH;

-- ============================================================
-- Materialized View: Daily API Summary (for Metabase)
-- ============================================================
CREATE TABLE IF NOT EXISTS dana_analytics.daily_api_summary
(
    date              Date,
    model             LowCardinality(String),
    tier              LowCardinality(String),
    request_count     UInt64,
    error_count       UInt64,
    total_tokens      UInt64,
    total_prompt_tokens UInt64,
    total_completion_tokens UInt64,
    avg_response_ms   Float64,
    unique_users      UInt64
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(date)
ORDER BY (date, model, tier);

CREATE MATERIALIZED VIEW IF NOT EXISTS dana_analytics.daily_api_summary_mv
TO dana_analytics.daily_api_summary AS
SELECT
    toDate(timestamp)          AS date,
    model,
    tier,
    count()                    AS request_count,
    countIf(status_code >= 400) AS error_count,
    sum(total_tokens)          AS total_tokens,
    sum(prompt_tokens)         AS total_prompt_tokens,
    sum(completion_tokens)     AS total_completion_tokens,
    avg(response_time_ms)      AS avg_response_ms,
    uniqExact(user_id)         AS unique_users
FROM dana_analytics.api_requests
GROUP BY date, model, tier;

-- ============================================================
-- Materialized View: Hourly Inference Performance
-- ============================================================
CREATE TABLE IF NOT EXISTS dana_analytics.hourly_inference_perf
(
    hour                       DateTime,
    model                      LowCardinality(String),
    request_count              UInt64,
    avg_tps                    Float64,
    avg_latency_ms             Float64,
    p95_latency_ms             Float64,
    speculative_acceptance_rate Float64,
    avg_batch_size             Float64
)
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(hour)
ORDER BY (hour, model);

CREATE MATERIALIZED VIEW IF NOT EXISTS dana_analytics.hourly_inference_perf_mv
TO dana_analytics.hourly_inference_perf AS
SELECT
    toStartOfHour(timestamp)   AS hour,
    model,
    count()                    AS request_count,
    avg(tokens_per_second)     AS avg_tps,
    avg(total_latency_ms)      AS avg_latency_ms,
    quantile(0.95)(total_latency_ms) AS p95_latency_ms,
    avg(if(speculative_total > 0, speculative_accepted / speculative_total, 0)) AS speculative_acceptance_rate,
    avg(batch_size)            AS avg_batch_size
FROM dana_analytics.inference_events
GROUP BY hour, model;
