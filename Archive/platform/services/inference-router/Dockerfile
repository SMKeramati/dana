FROM python:3.12-slim AS builder
WORKDIR /app
COPY packages/dana-common /app/packages/dana-common
RUN pip install --no-cache-dir /app/packages/dana-common

COPY services/inference-router /app/services/inference-router
RUN pip install --no-cache-dir /app/services/inference-router

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY services/inference-router/src /app/src

EXPOSE 8002
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]
