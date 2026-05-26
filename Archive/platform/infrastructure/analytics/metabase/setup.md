# Metabase Setup for Dana Analytics

## First Launch
1. Visit http://localhost:3002
2. Select language → Persian (فارسی) or English
3. Create admin account (use a strong password)
4. Add database connection → ClickHouse:
   - Driver: ClickHouse (community driver — see below)
   - Host: `clickhouse`
   - Port: `8123`
   - Database: `dana_analytics`
   - Username: `dana_analytics`
   - Password: from `.env` → `CLICKHOUSE_PASSWORD`

## Install ClickHouse Driver
Download and place in `infrastructure/analytics/metabase/plugins/`:

```bash
curl -L -o infrastructure/analytics/metabase/plugins/clickhouse.metabase-driver.jar \
  https://github.com/ClickHouse/metabase-clickhouse-driver/releases/download/1.3.4/clickhouse.metabase-driver.jar
```

Then restart Metabase: `docker compose restart metabase`

## Recommended Dashboards to Create

### 1. Product Overview (کلی)
- Daily Active Users (DAU)
- Total API Requests Today
- Total Tokens Consumed
- Revenue This Month
- Error Rate

### 2. Model Performance
- Tokens/second by model
- P95 Latency by model
- Speculative Decoding Acceptance Rate
- Batch Size Over Time

### 3. User Funnel
- Registrations per day
- Free → Pro conversions
- API Keys Created
- Churn (plan downgrade events)

### 4. Revenue Analytics
- Daily Revenue Trend
- Revenue by Tier
- Revenue by Model
- Top 10 Customers by Spend

### Key Tables in `dana_analytics`:
- `api_requests` — every API call
- `user_events` — user lifecycle
- `inference_events` — GPU performance
- `billing_events` — financial events
- `error_events` — error tracking
- `daily_api_summary` — pre-aggregated daily stats (fast queries)
- `hourly_inference_perf` — pre-aggregated hourly inference metrics
