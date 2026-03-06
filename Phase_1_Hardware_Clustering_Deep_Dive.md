# **Phase 1 Deep Dive: The "Twin-Z" Cluster Architecture**

**Objective:** To build a reliable, scalable AI inference cluster capable of serving a **120B Parameter Model** at **35+ Tokens Per Second (TPS)** for under **$600 USD**.

## **🏆 The Final Decision: "Twin-Z" Architecture**

After analyzing over a dozen hardware combinations, the **HP Z440 \+ Tesla M40** configuration was selected as the undisputed winner for cost-to-performance.

### **The "Golden Brick" Unit**

A single node in this cluster consists of:

* **Host:** **HP Z440 Workstation** (Xeon E5-1620 v3 or similar).  
* **Compute:** **2x Nvidia Tesla M40 (24GB)** GPUs.  
* **Network:** **1x Mellanox ConnectX-3 VPI** (40/56Gb/s).

### **The 2-Node Cluster Specs**

* **Total VRAM:** **96 GB** (4x 24GB).  
* **Total System RAM:** **144 GB** (128GB \+ 16GB).  
* **Interconnect:** **56Gb/s Infiniband** (Direct Attach).  
* **Power Draw:** \~1,100 Watts (Max Load).  
* **Total Cost:** **\~$577 USD**.

## **🧠 Why We Chose This (The "Why")**

We didn't just pick cheap parts; we picked parts that exploit specific market inefficiencies.

### **1\. Why the Tesla M40 (24GB)?**

* **The Math:** At **\~$45**, it offers **$1.87 per GB of VRAM**.  
* **The Competitors:**  
  * *Tesla K80 ($25):* Dual-chip architecture causes massive software headaches and crashing. **Rejected.**  
  * *Tesla P40 ($175):* Faster (Pascal architecture), but 4x the price for the same VRAM capacity. Good upgrade, bad starting point. **Rejected (for MVP).**  
  * *RTX 3090 ($700):* Too expensive for a garage startup. **Rejected.**  
  * *Chinese/Moore Threads ($200):* Zero software support for llama.cpp. **Rejected.**  
* **Verdict:** The M40 is the only card that fits a 120B model (needs \~70GB) within a \<$600 budget.

### **2\. Why the HP Z440 Workstation?**

* **The Problem:** Most cheap PCs (Dell Optiplex) have weak power supplies (250W) and small cases.  
* **The Solution:** The Z440 comes with a **700W Power Supply (90% Efficiency)** standard.  
* **PCIe Lanes:** It utilizes the **Xeon E5-1600/2600 v3** CPU, which provides **40 PCIe Lanes**.  
  * *Result:* Both GPUs run at full **x16 Speed**. Consumer PCs (Core i7) often throttle the second slot to x4 or x8.  
* **Cost:** At **\~$100**, it is cheaper than buying a case \+ motherboard \+ PSU separately.

### **3\. Why Mellanox ConnectX-3 (Infiniband)?**

* **The Bottleneck:** Splitting a model across two computers requires moving \~100MB of data *per token*.  
* **Ethernet (1Gb/s):** Takes 800ms per token. (1 TPS). **Unusable.**  
* **10Gb Ethernet ($30):** Takes 80ms per token. (12 TPS). **Okay.**  
* **ConnectX-3 ($25):** Takes **14ms** per token via 56Gb/s Infiniband. **Instant.**  
* **Verdict:** It is faster *and* cheaper than 10Gb Ethernet because enterprise datacenters dumped them years ago.

## **🚫 Rejected Alternatives (Pros & Cons Analysis)**

We explored several "out of the box" ideas. Here is why they failed the Phase 1 criteria.

### **Option A: The "Orange Pi Mesh" (ARM Cluster)**

* **Concept:** Chain 4x Orange Pi 5 Plus boards ($135 each) via 10GbE.  
* **Pros:** Futuristic, low power, "Cluster on a Desk."  
* **Cons:**  
  * **Bandwidth Starvation:** The PCIe 3.0 x4 slot bottlenecks the 40Gb cards.  
  * **Compute Weakness:** The ARM CPU is too slow to feed the network cards efficiently.  
  * **Cost:** Ends up costing **\~$750** for slower performance than the Z440.  
* **Verdict:** Good for a PhD paper, bad for a business.

### **Option B: The "Xeon X99 Mesh" (CPU Only)**

* **Concept:** Use cheap Xeons and massive DDR4 RAM (512GB) to run models on CPU.  
* **Pros:** Massive capacity (can run 400B+ models).  
* **Cons:**  
  * **Speed:** **0.5 \- 2 TPS.** The CPU memory bandwidth (\~60GB/s) is 5x slower than the M40 (\~288GB/s).  
  * **User Experience:** "Reading speed" is too slow for a paid product.  
* **Verdict:** Good for data analysis, bad for chat.

### **Option C: The "3-GPU Node"**

* **Concept:** Cram 3x M40s into one Z440 to save money on buying a second PC.  
* **Pros:** Slightly cheaper (\~$50 saved).  
* **Cons:**  
  * **Power:** Exceeds the 700W PSU limit. Requires a sketchy external power supply.  
  * **Thermals:** The middle card suffocates and throttles.  
  * **Physical:** Requires leaving the case open (Safety hazard).  
* **Verdict:** Unstable. Not production ready.

### **Option D: The "Mining Rig" (4 GPUs on a Frame)**

* **Concept:** Open-air frame, risers, dual PSUs.  
* **Pros:** High density (96GB VRAM in one rig).  
* **Cons:**  
  * **PCIe Bandwidth:** Most mining boards run slots at **x1 speed** (extremely slow for AI).  
  * **Complexity:** Debugging riser failures is a nightmare.  
  * **Resale:** Hard to sell a mining rig; easy to sell a workstation.  
* **Verdict:** Too complex for Phase 1\.

## **⚙️ Technical Configuration for Phase 1**

### **1\. Wiring Diagram**

\[ Node A (Brain) \]                      \[ Node B (Muscle) \]  
|-- PCIe x16 Slot 1: Tesla M40          |-- PCIe x16 Slot 1: Tesla M40  
|-- PCIe x16 Slot 2: Tesla M40          |-- PCIe x16 Slot 2: Tesla M40  
|-- PCIe x8 Slot 3: ConnectX-3  \<====\>  |-- PCIe x8 Slot 3: ConnectX-3  
      (Port 1\)              DAC Cable            (Port 1\)

### **2\. The "Safety" Mod (Critical)**

* **Issue:** The Tesla M40 does not use standard PCIe power.  
* **Fix:** You must use a **Dual 6-Pin Female to EPS 8-Pin Male** adapter.  
* **Warning:** If you plug a standard CPU 4+4 cable or PCIe 6+2 cable directly, it often does not fit or provides the wrong voltage rails. Use the specific adapter.

### **3\. Software Strategy (The Speed Hack)**

We use a **Hybrid Architecture** to maximize the hardware.

* **Node A (RAM Heavy):**  
  * Installs **128GB DDR4 RAM**.  
  * Runs the **Draft Model** (Small 1B model) on the CPU.  
  * Runs the **Context Cache** (User history) in RAM.  
  * Runs the **Speculative Decoding** logic.  
* **Node B (RAM Light):**  
  * Keeps stock 16GB RAM.  
  * Acts as a "Dumb" compute node, simply processing layers 41-80 of the big model.

## **💰 Final Cost Breakdown (Phase 1\)**

| Item | Notes | Cost |
| :---- | :---- | :---- |
| **2x HP Z440** | Host Machines ($100 each) | \*\*$200\*\* |
| **4x Tesla M40** | 24GB GPUs ($45 each) | \*\*$180\*\* |
| **2x ConnectX-3** | 40/56Gb Network Cards ($25 each) | \*\*$50\*\* |
| **1x DAC Cable** | QSFP+ Direct Attach Cable | **$15** |
| **4x Power Adapters** | EPS 8-Pin for Tesla | **$20** |
| **4x Blower Fans** | Cooling Mod | **$32** |
| **4x 32GB RAM** | DDR4 ECC (Used) | **$80** |
| **TOTAL** |  | **\~$577** |

**Conclusion:**

This architecture provides the **highest possible VRAM bandwidth per dollar** available on the used market today. It sacrifices physical space (two towers) for reliability and enterprise-grade connectivity (Infiniband).