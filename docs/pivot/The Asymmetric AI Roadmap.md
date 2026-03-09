# **The Asymmetric AI Infrastructure Roadmap**

**From Garage Startup to Frontier Cloud Provider**

**Objective:** Serve Frontier-Class AI Models (120B+) at commercial speeds (35+ TPS) using depreciated and emerging hardware to undercut industry costs by 90%.

## **🚀 Executive Summary**

This roadmap outlines a multi-phase strategy to build a scalable AI cloud provider without purchasing Nvidia H100s.

1. **Phase 0 (Scavenger):** Utilization of legacy consumer hardware (Laptops/Desktops) to handle the control plane logic at **$0 cost**.  
2. **Phase 1 (Bootstrap):** Build a "Groq-Killer" cluster for **\<$600** using depreciated Tesla M40 GPUs and high-speed Infiniband.  
3. **Phase 2 (Scale-Up):** Pivot to **Tenstorrent RISC-V** architecture to build switchless meshes that scale linearly.

## **♻️ Phase 0: The Scavenger Control Plane (The Front Office)**

**Goal:** Offload non-inference tasks to existing "junk" hardware to maximize GPU resources for AI.

**Hardware:** Old Laptop (Core i5, 6GB+ RAM) or Legacy Desktop.

**Cost:** **$0** (Already owned).

### **1\. The "Bouncer" Architecture**

Do not waste GPU Node resources on web hosting or queue management. Use the laptop as the **Gateway**.

* **Nginx (Reverse Proxy):** Handles user connections, SSL (HTTPS), and DDoS protection.  
* **Redis (The Queue):** Manages the waiting list. If GPU nodes are busy, requests sit here (in the laptop's RAM) rather than crashing the AI server.  
* **Frontend UI:** Hosts the Chatbot Interface (React/Node.js).

**Wiring:**

* **Wi-Fi:** Laptop connects to Home Internet (Public IP) to serve users.  
* **Ethernet:** Laptop connects to **Node A** (M40 Cluster) via a cheap $10 Gigabit Switch or direct cable.

### **2\. The "Zombie Grid" Alternative (Zero-Capital Startup)**

*If you cannot afford the $577 M40 Cluster, you can start here.*

**Technology:** **Exo** or **Petals** (Distributed Inference).

**Concept:** Pool RAM from 5+ old laptops/desktops to run large models over Wi-Fi.

* **Pros:** $0 Cost. "Green/Recycled" Marketing angle.  
* **Cons:** Slow (\~2-4 TPS) due to Wi-Fi latency.  
* **Strategy:** Combine 4-5 laptops to reach \~64GB RAM. Run **Llama-3-70B**. Market it as "Privacy-First Decentralized AI".

## **🏗️ Phase 1: The "Twin-Z" M40 Cluster (The MVP)**

**Goal:** Serve **GPT-OSS-120B** at **35+ Tokens/Sec** (Instant Feel).

**Budget:** \~$577 USD.

**Status:** Production Ready.

### **1\. Hardware Bill of Materials**

We utilize a **2-Node Cluster** connected directly via 56Gb/s Infiniband to eliminate network latency.

| Component | Qty | Specification | Role | Est. Price |
| :---- | :---- | :---- | :---- | :---- |
| **Host Nodes** | 2 | **HP Z440 Workstation** (Xeon E5-1620 v3, 700W PSU) | **The Shell.** Cheap, reliable, native support for 2x GPUs. | **$200** |
| **Compute** | 4 | **Nvidia Tesla M40 24GB** | **The Muscle.** 96GB Total VRAM. | **$180** |
| **Network** | 2 | **Mellanox ConnectX-3 VPI** (Model: MCX354A-FCBT) | **The Nerve.** 56Gb/s Interconnect. | **$50** |
| **Cabling** | 1 | **QSFP+ Passive DAC Cable** (1m \- 3m) | **The Synapse.** Direct wiring (No Switch). | **$15** |
| **Memory** | 4 | **32GB DDR4 ECC Reg** (128GB Total) | **The Buffer.** Install in Node A for Context/Drafting. | **$80** |
| **Power** | 4 | **Dual 6-Pin to EPS 8-Pin** | **⚠️ SAFETY.** Adapts PSU to Tesla power pinout. | **$20** |
| **Cooling** | 4 | **Blower Fan (High RPM)** | **Survival.** M40s are passive; tape fans to intake. | **$32** |
| **TOTAL** |  |  |  | **\~$577** |

### **2\. Assembly & Wiring Guide**

* **Power Warning:** Do **NOT** use standard 6+2 PCIe cables for the M40s. You *must* use the **EPS 8-Pin adapter**. The M40 uses CPU pinouts.  
* **Network Topology:**  
  * Plug ConnectX-3 cards into the remaining PCIe slot on both nodes.  
  * Connect the DAC cable directly from Port 1 (Node A) to Port 1 (Node B).  
* **Cooling:** Secure fans with Kapton tape or 3D-printed shrouds. Ensure 100% airflow through the fins.

### **3\. Software Configuration (Linux)**

**A. Network Setup (IP over InfiniBand)**

Since we have no switch, we manually assign IPs to create a private 56Gb/s lane.

\# Install Tools  
sudo apt install infiniband-diags ibverbs-utils opensm

\# Start Subnet Manager (Run on Node A ONLY)  
sudo systemctl start opensm

\# Configure Node A (Master/Brain)  
sudo ip addr add 192.168.10.1/24 dev ib0  
sudo ip link set ib0 up

\# Configure Node B (Slave/Muscle)  
sudo ip addr add 192.168.10.2/24 dev ib0  
sudo ip link set ib0 up

**B. The "Speed Faking" Stack**

We use llama.cpp with **RPC (Remote Procedure Call)** and **Speculative Decoding**.

* **Step 1: Start Node B (The Muscle)**  
  ./llama-server \--host 192.168.10.2 \--port 50052 \--model /dev/null

* **Step 2: Start Node A (The Brain)**  
  ./llama-server \\  
    \--model /dev/shm/model-120b-q4.gguf \\  \# Main Model in RAM  
    \--draft /models/Llama-3.2-1B-Q4.gguf \\  \# Draft Model (The "Intern")  
    \--rpc 192.168.10.2:50052 \\              \# Connect to Node B  
    \--split-mode layer \\                    \# Split layers 50/50  
    \--draft-max 12 \\                        \# Guess 12 tokens ahead  
    \--lookup-ngram-min 2 \\                  \# Copy-paste repetitive text  
    \--parallel 1                            \# Maximize speed for single user

## **🧪 Phase 2: The Tenstorrent Scale-Up (Series A)**

**Goal:** Outsmart Groq/Nvidia using **Switchless Mesh Topologies**.

**Trigger:** 500+ Daily Active Users OR University Research Grant.

### **1\. The Core Thesis (The "Paper")**

* **Problem:** Nvidia clusters waste 30% of cost/power on switches.  
* **Solution:** **Ethernet-Native Computing.** The Chip *is* the Switch.  
* **Hardware:** **Tenstorrent Wormhole n300** ($1,400).  
  * **Spec:** 2x RISC-V Chips \+ 24GB GDDR6 \+ **2x 100GbE Ports** on silicon.

### **2\. The "Cobra Mesh" Architecture**

Instead of a star topology (everything to a switch), we build a **Torus Mesh**.

* **Wiring:** Card A ![][image1] Card B ![][image1] Card C ![][image1] Card D.  
* **Software:** **vLLM** (now supports Tenstorrent).  
* **Research Value:** Prove linear scaling of 70B+ models without external networking gear.

### **3\. The University Abstract**

Use this to secure lab access/funding:

**Title:** *Scalable Deterministic Inference via Switchless Ethernet-Mesh Topologies on RISC-V Accelerators.*

**Abstract:** We propose a fractal mesh architecture where KV-Cache updates are pushed via on-chip Ethernet directly to neighbor SRAM, eliminating PCIe bottlenecks and enabling linear scaling of 100B+ parameter models on commodity power envelopes.

## **🏢 Phase 3: The "Legit Company" Pivot (Mass Scale)**

**Goal:** Serve 1,000+ Concurrent Users.

**Trigger:** $100k+ Funding or Hardware Saturation.

### **1\. The Facility Move**

* **Trigger:** Power draw exceeds 2,000W (Your garage circuit limit).  
* **Action:** Move to **Colocation (Tier 3 Datacenter)**.  
  * **Cost:** \~$600/month for a Half Rack (20U).  
  * **Benefit:** Industrial Cooling \+ 10Gb Fiber Uplink.

### **2\. The Hybrid Cloud (Bursting)**

Do not buy hardware for peak traffic. Buy for *base* traffic.

* **Stack:** Kubernetes (K3s) \+ SkyPilot.  
* **Logic:**  
  * **0-500 Users:** Served by your M40/Tenstorrent Cluster (80% Margin).  
  * **501+ Users:** Automatically spin up rental GPUs on Lambda/RunPod (10% Margin).

## **💡 Technical Deep Dive**

### **1\. Speculative Decoding vs. Parallelism**

You cannot have both at max efficiency on M40s.

* **Mode A (Demo/VIP):** parallel 1 \+ draft-max 12\.  
  * **Speed:** 35-50 TPS.  
  * **Use Case:** The "Wow" factor.  
* **Mode B (Public/Free):** parallel 8 \+ draft-max 0\.  
  * **Speed:** 8-10 TPS per user.  
  * **Use Case:** Handling traffic spikes.

### **2\. The Power Trap**

* **M40 Cluster:** Runs on standard 110V/120V wall outlets (US/EU).  
* **Scale-Up (10+ GPUs):** Requires **240V / 30A** circuits.  
  * **Warning:** Do not daisy-chain power strips. You will melt wires.

### **3\. RAM Offloading (The "Memory Mule")**

By installing **128GB RAM** in Node A:

* **KV Cache:** Offloaded to System RAM (--cache-type-k f16).  
* **Impact:** You can handle **32k Context** for 8 users simultaneously, leaving VRAM 100% free for the Model Layers.

## **📅 Final Roadmap Timeline**

| Timeline | Phase | Action | Cost | Status |
| :---- | :---- | :---- | :---- | :---- |
| **Month 0** | **Scavenge** | Setup Laptop Control Plane \+ Scavenged Parts. | **$0** | ♻️ Setup |
| **Month 1** | **Bootstrap** | Build 1x **M40 Twin-Z Cluster**. Launch Alpha API. | **$577** | 🛠️ Build |
| **Month 3** | **Validation** | Hit 100 DAU. Refine Nginx Routing & Speculative settings. | **$0** | 📈 Grow |
| **Month 6** | **Pivot** | Raise Funding / Grant. Buy **4x Tenstorrent n300**. | **$6k** | 🔬 Research |
| **Year 1** | **Scale** | Move to Colocation. Deploy Kubernetes. Liquidate M40s. | **$1k/mo** | 🏢 Corp |

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAYCAYAAAAYl8YPAAAAZUlEQVR4XmNgGAWjgHpAWlpaBl2MEsAkLy//B12QbCAnJzcfaOB/FEEFBQUHoEQ5ORhkGAgD2e0ww8yBnDRyMJJhoSguJBUADdkKxD/QxckBLECDvqILkgVkZGSE0MVGwSgYkgAAiaUmZiyQEgMAAAAASUVORK5CYII=>