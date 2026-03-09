# مستندات API دانا

به مستندات پلتفرم هوش مصنوعی **دانا** خوش آمدید.

## شروع سریع

### ۱. دریافت کلید API

ابتدا یک حساب کاربری بسازید و از [داشبورد](/dashboard/keys) یک کلید API دریافت کنید.

### ۲. نصب SDK

```bash
pip install openai
```

### ۳. اولین درخواست

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.dana.ir/v1",
    api_key="dk-YOUR_API_KEY"
)

response = client.chat.completions.create(
    model="qwen3-235b-moe",
    messages=[
        {"role": "user", "content": "سلام! یک تابع مرتب‌سازی بنویس"}
    ]
)

print(response.choices[0].message.content)
```

### ۴. استریم پاسخ

```python
stream = client.chat.completions.create(
    model="qwen3-235b-moe",
    messages=[
        {"role": "user", "content": "تابع فیبوناچی بنویس"}
    ],
    stream=True
)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="")
```

## آدرس پایه API

```
https://api.dana.ir/v1
```

## مدل‌های موجود

| مدل | توضیح | حداکثر توکن |
|-----|--------|-------------|
| `qwen3-235b-moe` | مدل اصلی - ۲۳۵ میلیارد پارامتر MoE | ۳۲,۷۶۸ |

## نقاط پایانی (Endpoints)

- [`POST /v1/chat/completions`](./chat-completions) - تولید پاسخ چت
- [`GET /v1/models`](./models) - لیست مدل‌های موجود
- [`POST /auth/register`](./authentication) - ثبت‌نام
- [`POST /auth/login`](./authentication) - ورود
- [`POST /auth/api-keys`](./authentication) - ساخت کلید API

## محدودیت نرخ

| پلن | درخواست/دقیقه | توکن/روز |
|------|---------------|----------|
| رایگان | ۵ | ۱,۰۰۰ |
| حرفه‌ای | ۶۰ | ۱۰۰,۰۰۰ |
| سازمانی | ۶۰۰ | نامحدود |
