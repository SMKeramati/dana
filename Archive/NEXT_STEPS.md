# قدم بعدی — اجاره GPU و آپگرید مدل

## ۱. اجاره GPU

**انتخاب: RTX 3090 (24GB) — ۸۵٬۰۰۰ تومن/ساعت (Qom-7)**

### چرا
- Qwen3-30B-A3B-GPTQ-Int4 (و نسل جدیدش Qwen3.6-35B-A3B-GPTQ-Int4) حدود ۱۸-۲۰ گیگ VRAM می‌خواد + KV cache → روی ۲۴ گیگ راحت جا میشه.
- T4 کولب (۱۶ گیگ) به همین دلیل OOM می‌داد.
- ارزون‌ترین گزینه با VRAM کافی. برای inference روی مدل کوانتایز شده، bottleneck حافظه‌ست نه compute — یعنی ۴۰۹۰ تفاوت محسوسی نداره.

### چرا بقیه نه
| GPU | دلیل رد |
|---|---|
| RTX 4090 24GB | ۱۶% گرون‌تر، سود ناچیز برای inference کوانتایز |
| A6000 48GB | فقط وقتی fine-tune یا batch بزرگ بخوای |
| L40S / A100 / H100 | overkill، پول هدر |

### هزینه
- روزی ۴ ساعت کار → **۳۴۰٬۰۰۰ تومن/روز**
- پلن ماهانه (۳۹٫۹ میلیون) فقط اگه روزی >۷ ساعت کار می‌کنی به‌صرفه‌ست.

### قبل از اجاره چک کن
- [ ] فضای storage کافی (مدل ~۱۸ گیگ)
- [ ] bandwidth دانلود از HuggingFace
- [ ] نسخه CUDA و درایور سازگار با gptqmodel

---

## ۲. آپگرید مدل

**از Qwen3-30B-A3B → Qwen3.6-35B-A3B-GPTQ-Int4**

### HF repo
[`palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`](https://huggingface.co/palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4)

### مقایسه
| ویژگی | Qwen3-30B-A3B (فعلی) | Qwen3.6-35B-A3B (جدید) |
|---|---|---|
| Total params | 30B | 35B |
| Active per token | 3B | 3B |
| Experts | — | 256 total, 8 routed + 1 shared |
| Multimodal | خیر | بله (vision encoder) |
| VRAM (Int4) | ~۱۸ گیگ | ~۱۸-۲۰ گیگ |
| Release | — | ۱۶ آوریل ۲۰۲۶ |
| MTP speculative decoding | خیر | بله |

### چرا این و نه بقیه
- **Qwen3.6-27B (dense)** — سریع‌تر (۷۲ tok/s روی ۳۰۹۰) ولی dense هست. برای Dana MoE engine بی‌فایده.
- **GLM-4.7-Flash 30B MoE** — رقیب جدی، ولی architecture فرق می‌کنه؛ router code رو باید ادپت کنی. فاز دوم.
- **Gemma 3 27B** — dense.
- **DeepSeek R1 distills** — dense distill.

---

## ۳. ترتیب اجرا

1. RTX 3090 اجاره کن (ساعتی شروع، نه ماهانه).
2. مدل فعلی (`Qwen3-30B-A3B-GPTQ-Int4`) رو روی ۳۰۹۰ بزن — اول بفهم engine سالم کار می‌کنه.
3. model ID رو عوض کن به `palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`.
4. اگه engine ساپورت می‌کنه، **MTP speculative decoding** رو فعال کن — throughput جهش می‌کنه.
5. benchmark بگیر و با baseline T4 مقایسه کن.
6. (اختیاری، فاز ۲) GLM-4.7-Flash رو به‌عنوان MoE با architecture متفاوت تست کن.

## ⚠️ ریسک‌ها

- نسخه GPTQ Int4 از Qwen3.6-35B-A3B با ۴× RTX 3060 quantize شده — community-made، نه رسمی Qwen. اگه نتیجه عجیب گرفتی، به مدل رسمی BF16 برگرد (۷۰ گیگ، روی A6000 نیاز).
- bandwidth ابری ایران ممکنه دانلود ۲۰ گیگ مدل رو طول بده — اولین ساعت پولش هدر میره.
