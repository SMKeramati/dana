# احراز هویت

API دانا از دو روش احراز هویت پشتیبانی می‌کند:

## ۱. کلید API (توصیه‌شده)

کلید API را در هدر `Authorization` قرار دهید:

```bash
curl https://api.dana.ir/v1/chat/completions \
  -H "Authorization: Bearer dk-YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-235b-moe", "messages": [{"role": "user", "content": "سلام"}]}'
```

### ساخت کلید API

```bash
# ابتدا ثبت‌نام کنید
curl -X POST https://api.dana.ir/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "your_password"}'

# وارد شوید و توکن بگیرید
curl -X POST https://api.dana.ir/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "your_password"}'
# پاسخ: {"access_token": "...", "token_type": "bearer"}

# کلید API بسازید
curl -X POST https://api.dana.ir/auth/api-keys \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "کلید اصلی", "permissions": ["chat"]}'
# پاسخ: {"key": "dk-f1_abc...", ...}
```

### فرمت کلید API

کلیدهای API دانا با `dk-` شروع می‌شوند:
- `dk-f...` — پلن رایگان
- `dk-p...` — پلن حرفه‌ای
- `dk-e...` — پلن سازمانی

**مهم:** کلید API فقط یکبار در زمان ساخت نمایش داده می‌شود. آن را در جای امنی نگه دارید.

## ۲. توکن دسترسی

توکن دسترسی از endpoint `/auth/login` دریافت می‌شود و برای مدیریت حساب (ساخت کلید، مشاهده مصرف) استفاده می‌شود.

```python
import httpx

# ورود
resp = httpx.post("https://api.dana.ir/auth/login", json={
    "email": "you@example.com",
    "password": "your_password"
})
token = resp.json()["access_token"]

# ساخت کلید API
resp = httpx.post("https://api.dana.ir/auth/api-keys",
    headers={"Authorization": f"Bearer {token}"},
    json={"name": "کلید پروژه", "permissions": ["chat"]}
)
api_key = resp.json()["key"]
print(f"کلید شما: {api_key}")
```

## خطاهای احراز هویت

| کد وضعیت | توضیح |
|----------|-------|
| `401` | کلید API نامعتبر یا منقضی |
| `403` | دسترسی غیرمجاز |
| `429` | محدودیت نرخ |
