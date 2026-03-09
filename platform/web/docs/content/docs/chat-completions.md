# Chat Completions

## `POST /v1/chat/completions`

تولید پاسخ بر اساس مکالمه ورودی. سازگار با فرمت OpenAI.

## پارامترهای درخواست

| پارامتر | نوع | الزامی | پیش‌فرض | توضیح |
|---------|------|--------|---------|-------|
| `model` | string | بله | — | نام مدل (مثلاً `qwen3-235b-moe`) |
| `messages` | array | بله | — | آرایه‌ای از پیام‌ها |
| `temperature` | float | خیر | `0.7` | میزان خلاقیت (۰ تا ۲) |
| `max_tokens` | int | خیر | `4096` | حداکثر توکن‌های خروجی |
| `stream` | bool | خیر | `false` | فعال‌سازی استریم |
| `top_p` | float | خیر | `1.0` | نمونه‌برداری nucleus |
| `frequency_penalty` | float | خیر | `0.0` | جریمه تکرار |
| `presence_penalty` | float | خیر | `0.0` | جریمه حضور |

## ساختار پیام

```json
{
  "role": "user",
  "content": "متن پیام"
}
```

نقش‌های مجاز: `system`, `user`, `assistant`

## نمونه درخواست

### Python

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.dana.ir/v1",
    api_key="dk-YOUR_KEY"
)

response = client.chat.completions.create(
    model="qwen3-235b-moe",
    messages=[
        {"role": "system", "content": "تو یک برنامه‌نویس حرفه‌ای هستی"},
        {"role": "user", "content": "یک REST API ساده با FastAPI بنویس"}
    ],
    temperature=0.3,
    max_tokens=2048
)

print(response.choices[0].message.content)
```

### cURL

```bash
curl -X POST https://api.dana.ir/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dk-YOUR_KEY" \
  -d '{
    "model": "qwen3-235b-moe",
    "messages": [
      {"role": "user", "content": "تابع مرتب‌سازی سریع بنویس"}
    ]
  }'
```

### JavaScript

```javascript
const response = await fetch("https://api.dana.ir/v1/chat/completions", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Authorization": "Bearer dk-YOUR_KEY"
  },
  body: JSON.stringify({
    model: "qwen3-235b-moe",
    messages: [{ role: "user", content: "سلام دانا!" }]
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

## نمونه پاسخ

```json
{
  "id": "chatcmpl-abc123def456",
  "object": "chat.completion",
  "created": 1709827200,
  "model": "qwen3-235b-moe",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "سلام! چطور می‌تونم کمکتون کنم؟"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 15,
    "total_tokens": 27
  }
}
```

## استریم

با `"stream": true` پاسخ به صورت Server-Sent Events (SSE) ارسال می‌شود:

```python
stream = client.chat.completions.create(
    model="qwen3-235b-moe",
    messages=[{"role": "user", "content": "داستان کوتاه بنویس"}],
    stream=True
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

هر chunk به صورت SSE ارسال می‌شود:
```
data: {"id":"chatcmpl-...","choices":[{"delta":{"content":"سلام"},"index":0}]}

data: {"id":"chatcmpl-...","choices":[{"delta":{},"finish_reason":"stop","index":0}]}

data: [DONE]
```

## کدهای خطا

| کد | توضیح |
|----|-------|
| `400` | پارامتر نامعتبر |
| `401` | احراز هویت ناموفق |
| `429` | محدودیت نرخ |
| `500` | خطای سرور |
