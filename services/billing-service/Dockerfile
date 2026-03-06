# ---------- build stage ----------
FROM python:3.12-slim AS builder

WORKDIR /app

COPY packages/dana-common /app/packages/dana-common
RUN pip install --no-cache-dir /app/packages/dana-common

COPY services/billing-service /app/services/billing-service
RUN pip install --no-cache-dir /app/services/billing-service

# ---------- runtime stage ----------
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY services/billing-service/src /app/src

EXPOSE 8003

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8003"]
