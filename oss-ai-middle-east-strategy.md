# OSS AI Model Serving in Middle East — Strategy Notes

## Original Prompt

> Suppose i want to serve oss ai models to make money in middle east market.
> There are some paths, i want help from you to think and strategize.
> 2 main factors here
>
> Segment
> 0. Customer: people generally (creating an app / claudecode or chatgpt clone / ...) and wire it to that model. Sell subscription. Like selling claude code on selfhosted deepseekv4
> 1. Customer: general devs (serving public api like groq)
> 2. Customer: enterprises (serving private api for enterprises on their GPU / Rent for them)
>
> Model size/speed
> M. Medium (under ~30b), blazing
> L. Large (~150b), good speed ok quality
> XXL. Behemoth (~400+), acceptable speed or even slow, premium quality
>
> Which for whom:
> 0 needs M & L
> Focus on scale and concurrency issues
> Most dev dependant and very software heavy
> Question: why not use deepseek, glm, ... Apis directly? Even if internet limits then metis/gapgpt...
>
> 1 needs XXL as the expensive plan, L as good and M as commodity
> Can start simpler with fewer models. But which is best value?
> Best general purpose for all kinds of users, potential to turn into 1
> Very scalable but concurrency and... Must be handled well
>
> 2 needs XXL only with some expensive gear in my opinion
> They will pay good to have near claude quality in hand for how much use and have privacy but demand quality and speed
>
> What do i have:
> 3+ years using every ai tech almost for vibecoding. Done 100+ successful coding projects. Deceloped some ai agents.
> Know a lot about ai engineering
> Connections in the startup community in iran.
> Probably some gov support in future to have rental gpu at lower prices (e g. 40% off)
> Rentals also do some customizations in their month rent, connecting up to 8 gpus, e.g. 8x 4090 or H100!
> Good info on inference and top osses and speed increasing techniques.
> I have infinite ai tokens to burn in claudecode and build.
>
> Now please first revise and do searches needed to understand the context and help me think, then ask some questions to gather more info on vague stuff

---

## Concise Summary

### Landscape (May 2026)

**Top OSS models:**
- **XXL:** DeepSeek V4 Pro/Max, Kimi K2.6 (1T/32B active), GLM-5.1 (744B/40B, MIT), Qwen3.5-397B, MiMo-V2.5-Pro
- **L:** GLM-4.7-Flash (30B MoE), Qwen3.6-35B-A3B, MiniMax-M2.7, DeepSeek V4 Flash
- **M:** Qwen3.6-27B, Devstral Small 2, Mistral Small 4, Gemma 4 26B-A4B

**Self-host break-even:** ~5M tok/day single-host, **30–50M tok/day frontier MoE**. Below that, hosted APIs win on TCO.

**Inference engine:** SGLang for agentic/RAG/multi-turn (29% > vLLM on H100, 3.1x on DeepSeek V3, 6.4x prefix-heavy). vLLM as safe default.

**Iran competitive context:**
- **AvalAI** already does Segment 1 — 26+ providers, OpenAI-compatible, zero markup.
- **GapGPT** owns consumer Persian-UI chat (~$6/mo).
- Iran = Tier 3 export control: H100/H200/Blackwell formally prohibited. Realistic hardware = 4090s, smuggled A100s, gray-market H100s.

### Segment Read

| Segment | Verdict |
|---|---|
| **0 — Consumer apps** | Don't self-host. Model is commodity; product is the moat. Buy tokens from AvalAI/DeepSeek; invest in the app. |
| **1 — Public API** | AvalAI is entrenched. Beat them only via (a) self-host pricing at >30M tok/day, or (b) features they lack (SLAs, Anthropic-compat Claude Code passthrough, fine-tuning, dedicated endpoints). |
| **2 — Enterprise private** | **Strongest structural moat.** Privacy/sovereignty need is real (banking, telco, oil, gov). AvalAI can't serve this — they route externally. Matches your gov-GPU-discount + customization edge. |

### Model Picks

| Segment | M | L | XXL |
|---|---|---|---|
| 0 (apps) | Qwen3.6-27B | GLM-4.7-Flash | (don't self-host) |
| 1 (pub API) | Qwen3.6-27B | **GLM-4.7-Flash** | **GLM-5.1** |
| 2 (enterprise) | — | Qwen3.6-35B-A3B | **GLM-5.1** or DeepSeek V4 Pro |

**GLM-5.1** is the recurring answer: MIT license, 40B active params (single-host viable on 8xH100 / 8x4090 INT4), frontier-level coding/agentic scores.

### Open Questions

1. Iran only, or pan-MENA (UAE/Saudi/Egypt = Tier 2, totally different game)?
2. What hardware does the gov subsidy actually unlock? 4090s vs. real H100s determines whether XXL is on the table.
3. Capital & timeline — bootstrap from revenue or runway-funded?
4. Payment rails — Rial only, or USD/crypto for non-Iran users?
5. Position vs. AvalAI — compete, complement (sell them capacity?), or sidestep via product?
6. Persian/Arabic quality requirements? (Qwen/DeepSeek strong; GLM weaker on Persian.)
7. Segment 1 concurrency profile — burst chat (Groq-style) or sustained agentic (Claude Code style)?
8. Infra company or product company that owns infra? Different lives.
