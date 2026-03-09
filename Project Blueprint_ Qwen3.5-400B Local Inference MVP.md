# **Project Blueprint: Qwen3.5-400B Local Inference MVP**

## **1\. The Core Objective**

The goal is to run a massive Mixture of Experts (MoE) model—specifically Qwen3.5 (\~397B total parameters, \~17B active parameters per token)—locally and cost-effectively at 4-bit quantization (Q4).

The primary strategy relies on **hybrid memory offloading**: storing the \~17B "active" parameters and KV cache in fast GPU VRAM, while offloading the inactive bulk of the \~400B parameters (the "cold" experts) to cheap system RAM (DDR5).

## **2\. The Hardware Bottleneck & Evaluation**

Traditional layer-by-layer offloading fails with MoE models because the routing network dynamically selects different experts for every single token. This causes severe bottlenecks across the PCIe bus.

**Hardware Configurations Evaluated:**

* **Legacy Data Center GPUs (Tesla P40s):** Rejected. While cheap, they run on PCIe Gen 3.0 (\~16 GB/s). Fetching cold experts across Gen 3.0 results in fractional tokens-per-second (TPS).  
* **High-VRAM Clusters (8x RTX 3090 or Apple Mac Studio):** Capable of fitting the entire 200GB model in high-bandwidth memory, yielding 15-30 TPS. However, this abandons the cheap RAM-offloading strategy.  
* **Single-GPU Powerhouses (A100 vs. H100):** \* *A100 (80GB):* Uses PCIe Gen 4 (64 GB/s). Causes noticeable stutter on cache misses.  
  * *H100 (80GB):* Uses PCIe Gen 5 (128 GB/s). The optimal single-card solution for RAM offloading, cutting fetch latency in half.

## **3\. The Speculative Decoding "Trap" in MoE**

Attempting standard speculative decoding (using a tiny draft model to guess tokens) on an MoE model chokes the PCIe bus. If a draft model guesses 5 tokens, verifying them might require 20+ different experts from system RAM simultaneously, crashing TPS.

**The Solution (Emerging Research):**

To utilize speculative decoding on an offloaded MoE model, the software must use:

1. **Expert Budgeting:** Placing a hard limit on how many cold experts can be fetched during verification.  
2. **Speculative Prefetching:** Using the draft model's output to predict which experts the target model will need, and fetching them asynchronously across the PCIe bus *before* they are required by the compute cores.

## **4\. Inspiration: The Groq Architecture**

Exploration into Groq’s ultra-fast LPU architecture revealed two key advantages:

1. **SRAM vs. HBM:** Using localized SRAM eliminates the memory-fetch latency inherent in standard GPU HBM.  
2. **Deterministic Compiling:** Pre-calculating the exact path of every tensor, eliminating dynamic scheduling.  
   *Conclusion:* The most capital-efficient MVP for a software-focused founder is to replicate Groq's *deterministic data movement* purely in software, using off-the-shelf hardware.

## **5\. The Hybrid MVP Hardware Lab Setup**

To develop the software MVP, the hardware must be optimized for maximum PCIe bandwidth and system RAM speed.

* **CPU:** AMD Threadripper Pro 7000 WX-Series (Massive PCIe Gen 5 lane counts).  
* **Motherboard:** Workstation board (e.g., Asus Pro WS TRX50-SAGE).  
* **System RAM:** 256GB to 512GB of 8-channel DDR5 RDIMM ECC Memory.  
* **GPUs:** 2x Nvidia RTX 6000 Ada Generation (48GB each). (Enough VRAM to hold the 17B active parameters, KV cache, draft model, and a prefetch buffer).

## **6\. The Three MVP Development Paths Ahead**

### **Path A: The "Heatmap & Router" Hack (Pure Python/ML \- Medium Difficulty)**

* Build a "smart orchestrator" over an existing backend. Train a tiny MLP to predict cold experts based on prompts and prefetch them.

### **Path B: The "Budgeted Speculator" Engine (C++/CUDA \- High Difficulty)**

* Write a custom CUDA engine. A tiny draft model mathematically predicts required experts for the 400B model. The engine enforces strict expert load budgets per layer to prevent PCIe choking.

### **Path C: The "Pre-Gated" Architecture (Model Fine-Tuning \- Extreme Difficulty)**

* Modify the MoE routing so Layer 1's router predicts experts for Layer 2\. Requires deeply fine-tuning the 400B model.

## **7\. The API Pivot: Concurrency Software Solutions**

Pivoting to an API-as-a-Service model introduces the **Concurrency Trap**. If User A needs Expert 1 and User B needs Expert 2 simultaneously, PCIe bus traffic doubles, causing severe bottlenecks.

* **Strict Node-to-Stream Routing:** Load balancer limits each physical Node to 1-2 concurrent users.  
* **Expert-Aware Batching:** A custom algorithm intercepts incoming API requests and groups users based on predicted expert overlap. If multiple users need the same "cold" expert, the engine fetches it once, holds it in VRAM, and processes those users' tokens simultaneously.

## **8\. Hardware & Cost Breakdown: Training vs. API Inference Nodes**

**Standard API Node Specs (Owned):** 1x Threadripper Pro, 512GB DDR5, 2x RTX 6000 Ada. (\~$18,000 \- $20,000 per node).

| Feature | Path A: Heatmap Hack | Path B: Budgeted Speculator | Path C: Pre-Gated Arch |
| :---- | :---- | :---- | :---- |
| **Training Gear (Rented)** | 1x Server (8x A100) | None required. | Cluster (64x H100) |
| **Training Cost** | \< $2,000 | $0 | $50,000 \- $150,000+ |
| **Est. API Node Cost** | \~$20k per node | \~$20k per node | \~$20k per node |
| **API Concurrency** | Very Low | Low to Medium | Medium |
| **Overall API Viability** | Good | Best | Too Risky |

## **9\. The Ultimate Value-to-Cost Strategy: The "Groq-Lite" API**

*(Assuming software development cost/effort is zero)*

If the goal is to build an ultra-fast, highly profitable API MVP while scaling hardware linearly, **Path B** combined with advanced concurrency software is the undisputed winner. It allows a startup to offer data-center-tier performance using consumer/prosumer workstation hardware.

### **The "Free Software" Moat**

Because software effort is assumed to be free, the startup invests 100% of its technical resources into building a proprietary, low-level **C++/CUDA Inference Engine**. This engine includes:

1. **Continuous Expert-Aware Batching:** The engine groups incoming API user queries not by when they arrived, but by *which experts they are about to need*.  
2. **Hybrid Quantization:** The engine dynamically formats the model. The active parameters sitting in VRAM are kept at standard 4-bit for intelligence. The cold experts in DDR5 RAM are aggressively quantized to 2-bit or 3-bit, drastically reducing their file size and doubling the speed they can travel across the PCIe Gen 5 bus.  
3. **Asynchronous Prefetching:** The PCIe bus never stops moving. It streams experts into a rotating VRAM "Smart Cache" buffer precisely one millisecond before the GPU compute cores need them.

### **The Lean Hardware Scaling Economics**

By utilizing this software stack, the startup completely bypasses the massive CapEx required for traditional AI infrastructure.

* **Zero Training Cost:** $0 is spent on renting Nvidia H100 clusters because the base Qwen3.5-400B and Qwen-1.5B (draft) models are used exactly as-is. All the "magic" happens in the memory routing engine, not the neural network weights.  
* **The "Unit of Compute":** The fundamental unit of the company is a single **$20,000 Workstation Node** (Threadripper Pro, 512GB DDR5, 2x RTX 6000 Ada).  
* **Linear Scaling:** \* **Day 1 (The MVP):** The startup buys exactly two $20k nodes ($40,000 CapEx) and places a load balancer in front of them. This is enough to serve early beta testers at blistering speeds.  
  * **Day 100 (Traction):** As API requests grow and the nodes hit maximum concurrency, the startup simply buys *one more $20k node at a time*.  
  * **The Result:** There are no massive data center leases, no $1.5M multi-node racks, and no hardware idling. Capacity is added linearly, exactly matching monthly recurring revenue (MRR) growth.

This strategy yields an API company with incredibly high gross margins per token, protected by a software moat that larger competitors relying on brute-force GPU hardware cannot easily replicate.