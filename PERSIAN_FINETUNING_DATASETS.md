# Persian Fine-Tuning Datasets Research

Research into the best available Persian/Farsi datasets for fine-tuning Qwen3-235B-MoE.

**Date:** 2026-03-07
**Target Model:** Qwen/Qwen3-235B-A22B (QLoRA fine-tuning)
**Supported Formats:** Alpaca, ShareGPT, JSONL, OpenAI-chat (per `finetuning-service/src/training/data_loader.py`)

---

## Tier 1: Recommended (High Quality, Ready to Use)

### 1. FarsInstruct — `ParsiAI/FarsInstruct`
- **Size:** 10M-100M tokens, 197 templates across 21 datasets
- **Format:** Parquet (instruction-response pairs)
- **License:** Apache 2.0
- **Quality:** Peer-reviewed, accepted for oral presentation at LoResLM @ COLING 2025
- **Details:** The most comprehensive open-source Persian instruction dataset. Contains both manually written instructions and translations from Public Pool of Prompts. Introduces the Co-CoLA framework for multi-task LoRA adaptability. Covers a wide range of NLP tasks.
- **Why use it:** Largest native Persian instruction dataset, academically validated, broad task coverage
- **HuggingFace:** https://huggingface.co/datasets/ParsiAI/FarsInstruct

### 2. Maux-Persian-SFT-30k — `xmanii/Maux-Persian-SFT-30k`
- **Size:** 30,000 conversations
- **Format:** ShareGPT-style messages (role/content arrays) — directly compatible with our loader
- **License:** Apache 2.0
- **Quality:** Multi-source, translated by DeepSeek R1, edited via OpenAI API for grammar/tone
- **Sources breakdown:**
  - infini-instruct-top-500k: 10,137 (33.8%)
  - Mauxi-SFT-Persian: 5,000 (16.7%)
  - WebInstructSub_axolotl: 4,636 (15.5%)
  - ultrainteract_trajectories_sharegpt: 3,534 (11.8%)
  - mauxi-talk-pro: 3,000 (10.0%)
  - mauxitalk-persian: 1,700 (5.7%)
- **Why use it:** Chat-formatted, diverse sources, directly compatible with ShareGPT loader
- **HuggingFace:** https://huggingface.co/datasets/xmanii/Maux-Persian-SFT-30k

### 3. MatinaAI Instruction Tuning — `MatinaAI/instruction_tuning_datasets`
- **Size:** Multiple subsets (exact count requires login; includes 6,380 ORPO alignment instances)
- **Format:** Instruction-tuning pairs + preference data (SFT + DPO/RLHF ready)
- **License:** Requires login to access
- **Quality:** Human-evaluated (1-5 scale, only score >= 3 retained), accepted at ACL 2025
- **Details:** Culturally grounded for Iranian context. Built using:
  - Culturally relevant keywords via LLaMA3.1-70B-Instruct
  - QA pairs on Iranian norms, values, beliefs
  - Evol-Instruct augmentation via GPT-4o-mini
  - Translation of ORCA/UltraChat via GPT-4o
- **Why use it:** Only dataset with explicit Iranian cultural alignment + human quality filtering. Perfect for Daneshbonyan alignment (demonstrates Persian cultural competency)
- **HuggingFace:** https://huggingface.co/datasets/MatinaAI/instruction_tuning_datasets

---

## Tier 2: Good Supplementary Datasets

### 4. Bactrian-X Persian — `MBZUAI/Bactrian-X` (lang: `fa`)
- **Size:** ~67,000 instruction-response pairs (52K Alpaca + 15K Dolly translated)
- **Format:** Alpaca-style (instruction, input, output) — directly compatible
- **License:** CC BY-NC 4.0 (non-commercial)
- **Quality:** Machine-translated instructions (Google Translate), GPT-3.5-turbo responses
- **Details:** Part of a 52-language multilingual dataset. Translations are adequate but not native-quality. Used by AHD Co. in their Persian LLM fine-tuning.
- **Why use it:** Large volume, Alpaca format plug-and-play, good for bootstrapping
- **Caveat:** NC license may conflict with commercial use; translation quality varies
- **Load:** `load_dataset("MBZUAI/Bactrian-X", "fa")`
- **HuggingFace:** https://huggingface.co/datasets/MBZUAI/Bactrian-X

### 5. Mauxi-SFT-Persian — `xmanii/Mauxi-SFT-Persian`
- **Size:** 5,000 rows (4.64 MB)
- **Format:** Chat messages
- **License:** Apache 2.0
- **Quality:** Translated from OpenHermes-100k using state-of-the-art LLMs
- **Why use it:** Clean, small, good for validation/testing pipeline before scaling up
- **HuggingFace:** https://huggingface.co/datasets/xmanii/Mauxi-SFT-Persian

### 6. xmanii Persian SFT/COT Collection
- **Size:** Multiple datasets (30k + 5k + 3k + 2k + 1k subsets)
- **Format:** Various chat formats
- **License:** Apache 2.0
- **Details:** A curated collection of high-quality Persian chat datasets including chain-of-thought reasoning data
- **Why use it:** COT data is valuable for improving reasoning capabilities in Persian
- **HuggingFace:** https://huggingface.co/collections/xmanii/persian-sft-cot-fine-tuning-datasets-67a4b94aeadb4a6b2ae3fc40

### 7. Aya Persian Instruction — `Shafagh/aya_persian_instruction_pn-summary`
- **Size:** Unknown (related to Aya multilingual project)
- **Format:** Instruction pairs
- **Details:** Part of the broader Aya multilingual initiative. Persian instruction data paired with PN-Summary.
- **HuggingFace:** https://huggingface.co/datasets/Shafagh/aya_persian_instruction_pn-summary

---

## Tier 3: Domain-Specific / Niche

### 8. BaSalam Product Catalog — `BaSalam/entity-attribute-dataset-GPT-3.5-generated-v1`
- **Use case:** Structured JSON output generation for e-commerce
- **Details:** Persian product catalog generation, good for domain-specific fine-tuning if Dana targets e-commerce use cases
- **HuggingFace:** Referenced in HF cookbook

### 9. ParsiNLU — `persiannlp/parsinlu_translation_en_fa`
- **Use case:** Translation, reading comprehension, QA benchmarks
- **Details:** Academic benchmark suite. More useful for evaluation than training.
- **HuggingFace:** https://huggingface.co/datasets/persiannlp/parsinlu_translation_en_fa

### 10. PersianQA
- **Use case:** Question answering
- **Size:** 989 questions, 21,915 annotated answers
- **Details:** Community QA dataset, useful for QA fine-tuning
- **GitHub:** https://github.com/sajjjadayobi/PersianQA

---

## Recommended Training Strategy

### Phase 1: General Persian Instruction Following
**Primary datasets:**
1. **FarsInstruct** (~100K+ samples) — broad task coverage, academic quality
2. **Maux-Persian-SFT-30k** (30K samples) — conversational, multi-source
3. **Bactrian-X Persian** (67K samples) — volume, Alpaca format

**Combined:** ~200K samples for initial SFT round

### Phase 2: Cultural Alignment & Quality
**Primary dataset:**
4. **MatinaAI** — culturally grounded, human-filtered, DPO/RLHF-ready

**Purpose:** Align model outputs with Iranian cultural context (critical for Daneshbonyan scoring)

### Phase 3: Chain-of-Thought & Reasoning
**Primary dataset:**
5. **xmanii COT collection** — Persian chain-of-thought data

**Purpose:** Improve reasoning quality in Persian responses

### Phase 4: Evaluation
**Benchmarks:**
- ParsiNLU (reading comprehension, QA)
- Custom Persian quality scoring (already implemented in `evaluator.py`)
- BLEU/ROUGE-L vs base model (already implemented)
- Persian typography quality (ZWNJ usage, sentence structure)

---

## Format Compatibility Matrix

| Dataset | Alpaca | ShareGPT | JSONL | OpenAI-chat | Native Compat |
|---------|--------|----------|-------|-------------|---------------|
| FarsInstruct | Parquet (needs conversion) | - | - | - | Needs adapter |
| Maux-SFT-30k | - | Yes (messages) | - | - | Direct |
| MatinaAI | Needs login to verify | - | - | - | TBD |
| Bactrian-X | Yes (instruction/input/output) | - | - | - | Direct |
| Mauxi-SFT | - | Yes (messages) | - | - | Direct |

**Action needed:** Write a small Parquet-to-Alpaca/JSONL converter for FarsInstruct, or add Parquet support to `data_loader.py`.

---

## License Summary

| Dataset | License | Commercial OK? |
|---------|---------|----------------|
| FarsInstruct | Apache 2.0 | Yes |
| Maux-SFT-30k | Apache 2.0 | Yes |
| MatinaAI | Unknown (gated) | TBD |
| Bactrian-X | CC BY-NC 4.0 | **No** |
| Mauxi-SFT | Apache 2.0 | Yes |
| xmanii COT | Apache 2.0 | Yes |

**Note:** If Dana is a commercial product, avoid Bactrian-X in production fine-tunes or use it only for research/development. FarsInstruct + Maux-SFT-30k + xmanii COT are all Apache 2.0 and commercially safe.

---

## Quick Start

```python
from datasets import load_dataset

# Tier 1 datasets
farsinstruct = load_dataset("ParsiAI/FarsInstruct")
maux_30k = load_dataset("xmanii/Maux-Persian-SFT-30k")

# Tier 2 datasets
bactrian_fa = load_dataset("MBZUAI/Bactrian-X", "fa")
mauxi_5k = load_dataset("xmanii/Mauxi-SFT-Persian")

# Convert Bactrian-X to our Alpaca format (already compatible):
# Each row has: instruction, input, output
```

---

## References

- FarsInstruct paper: https://arxiv.org/html/2407.11186
- Matina paper (ACL 2025): https://aclanthology.org/2025.findings-acl.1074/
- Bactrian-X paper: https://github.com/mbzuai-nlp/bactrian-x
- PersianMind: https://arxiv.org/html/2401.06466v1
- PersianLLaMA: https://arxiv.org/pdf/2312.15713
