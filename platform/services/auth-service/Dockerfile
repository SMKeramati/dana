FROM python:3.12-slim AS builder
WORKDIR /app
COPY packages/dana-common /app/packages/dana-common
RUN pip install --no-cache-dir /app/packages/dana-common

COPY services/auth-service /app/services/auth-service
RUN pip install --no-cache-dir /app/services/auth-service

FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY services/auth-service/src /app/src

EXPOSE 8001
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
