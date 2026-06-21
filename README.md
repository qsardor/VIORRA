<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6C63FF,100:A855F7&height=200&section=header&text=VIORRA&fontSize=80&fontAlignY=38&animation=fadeIn&fontColor=ffffff&desc=The%20Zero-Hallucination%20Admissions%20Engine&descAlignY=60&descSize=20" width="100%"/>

<br/>

<p align="center">
  <a href="https://huggingface.co/spaces/qsardor/VIORRA">
    <img src="https://img.shields.io/badge/🤗%20Try%20Demo-Live%20Now-FFD21E?style=for-the-badge&labelColor=1a1a2e" />
  </a>
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white&labelColor=1a1a2e" />
  &nbsp;
  <img src="https://img.shields.io/badge/Gemma_4-E2B--IT-FF6F00?style=for-the-badge&logo=google&logoColor=white&labelColor=1a1a2e" />
  &nbsp;
  <img src="https://img.shields.io/badge/LiteRT-C++_Engine-4285F4?style=for-the-badge&logo=google&logoColor=white&labelColor=1a1a2e" />
  &nbsp;
  <img src="https://img.shields.io/badge/License-AGPL--3.0-8B5CF6?style=for-the-badge&labelColor=1a1a2e" />
</p>

<br/>

> **VIORRA** is the only AI built to tell students the truth about their college essays.\
> Not compliments. Not suggestions. **The exact changes needed to get accepted.**

<br/>

```bash
git clone https://github.com/qsardor/VIORRA.git
cd VIORRA
pip install -e .
viorra
```

<br/>

</div>

---

## 🎯 The Problem

Every year, millions of students make the same fatal mistake. They open a browser and ask ChatGPT to review their college essay.

**This is a disaster.**

| The Problem | Why It Kills Applications |
|---|---|
| 💬 Generic chatbots **sugarcoat** feedback | They are trained by RLHF to be "agreeable", not honest |
| 📉 They grade against **SEO blog rubrics** | Not real Ivy League admissions standards |
| 💸 Human consultants cost **$100–$500/doc** | Creating a deeply unequal admissions system |
| 🌀 They **hallucinate scores** with no data source | Arbitrary grades with zero mathematical grounding |

VIORRA was built to solve every one of these problems simultaneously.

---

## ⚡ The Solution

<div align="center">

```
╔══════════════════════════════════════════════════════════════╗
║                       YOUR ESSAY                            ║
╚══════════════════╤══════════════════════════╤═══════════════╝
                   │                          │
         ┌─────────▼──────────┐    ┌──────────▼──────────┐
         │   1. THE CONTEXT   │    │    RAG RETRIEVAL    │
         │  FAISS Vector DB   │    │  Top 2 Real Essays  │
         │  all-MiniLM-L6-v2  │    │  Mathematically     │
         │  (384-dim vectors) │    │  Matched to Yours   │
         └─────────┬──────────┘    └──────────┬──────────┘
                   └──────────────────────────┘
                                  │
                   ┌──────────────▼──────────────┐
                   │      2. THE BRAIN           │
                   │   Gemma 4 E2B-IT via        │
                   │   Google LiteRT C++ Engine  │
                   │   RAG-Anchored Reasoning    │
                   └──────────────┬──────────────┘
                                  │
                   ┌──────────────▼──────────────┐
                   │      VIORRA'S VERDICT       │
                   │   Honest. Anchored. Real.   │
                   └─────────────────────────────┘
```

</div>

VIORRA uses a **Duo Architecture** that chains two AI systems together. The LLM is **never** given free reign — it is always anchored to a real, mathematically verified dataset of Ivy League essays before it generates a single word of feedback.

---

## 🧠 Why VIORRA Beats Every Chatbot

<table>
  <thead>
    <tr>
      <th>Feature</th>
      <th>ChatGPT / Claude</th>
      <th>VIORRA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Data Source</td>
      <td>❌ SEO blogs & Wikipedia</td>
      <td>✅ Real JHU accepted essays</td>
    </tr>
    <tr>
      <td>Feedback Honesty</td>
      <td>❌ RLHF-trained to be "nice"</td>
      <td>✅ Hardcoded brutal <code>SOUL.md</code> persona</td>
    </tr>
    <tr>
      <td>Score Grounding</td>
      <td>❌ Arbitrary hallucinated score</td>
      <td>✅ Mathematically anchored via FAISS</td>
    </tr>
    <tr>
      <td>Cost</td>
      <td>❌ $20/month subscription</td>
      <td>✅ Free (prototype phase)</td>
    </tr>
    <tr>
      <td>Privacy</td>
      <td>❌ Essay sent to external servers</td>
      <td>✅ Runs on your own hardware</td>
    </tr>
    <tr>
      <td>Framework Bloat</td>
      <td>❌ PyTorch + Transformers (20 GB+)</td>
      <td>✅ Pure C++ LiteRT backend (~3.5 GB)</td>
    </tr>
  </tbody>
</table>

---

## 📊 Real Performance Stats

Benchmarked live on a standard consumer laptop (RTX 4060):

<div align="center">

| Metric | Result |
|:---|:---:|
| 🚀 Cold Boot Time | **~6.3 seconds** |
| 🔍 RAG Vector Search | **~67 ms** |
| ⚡ Inference Speed | **~68 Tokens/sec** |
| 🧠 RAM Footprint | **~1.3 GB** |
| 🎮 VRAM Usage | **~3.5 GB** |

Run it yourself: `viorra --benchmark`

</div>

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/qsardor/VIORRA.git
cd VIORRA

# 2. Install dependencies
pip install -e .

# 3. Launch (opens in your browser at localhost:8000)
viorra
```

> [!NOTE]
> On first launch, VIORRA automatically downloads the Gemma 4 LLM and RAG vector database (~2.5 GB). Subsequent boots use the local cache.

**Don't want to install?** → [Try the live demo on HuggingFace Spaces](https://huggingface.co/spaces/qsardor/VIORRA)

---

## 🛠️ CLI Reference

```bash
# Analyze an essay file without launching the web UI
viorra --cli my_essay.txt

# Run a real-time hardware benchmark
viorra --benchmark

# Check GPU/VRAM compatibility
viorra --status

# Force-sync the latest Ivy League RAG database
viorra --update

# Purge all downloaded model caches
viorra --clear-cache

# Wipe all user data and sessions
viorra --factory-reset
```

---

## 🏗️ Tech Stack

<div align="center">

| Layer | Technology | Purpose |
|:---:|:---:|:---|
| 🧠 **Brain** | [Gemma 4 E2B-IT](https://huggingface.co/google/gemma-4-E2B-it) via [LiteRT](https://ai.google.dev/edge/litert) | Chain-of-Thought reasoning engine |
| 📚 **Context** | [FAISS](https://github.com/facebookresearch/faiss) + [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Vector search over Ivy League essay DB |
| ⚙️ **Backend** | FastAPI + Uvicorn | Async REST API server |
| 🖥️ **Frontend** | Vanilla HTML / CSS / JS | Zero-framework SPA |
| 📦 **Packaging** | Local PIP Install | `pip install -e .` |

</div>

---

## 💻 System Requirements

| Component | Minimum | Recommended |
|:---:|:---:|:---:|
| **RAM** | 8 GB | 16 GB+ |
| **VRAM** | 4 GB | 6 GB+ |
| **GPU** | Integrated | NVIDIA / AMD Dedicated |
| **Storage** | 4 GB free | 8 GB free |
| **OS** | Windows 10 / macOS 12 / Ubuntu 20.04 | Latest |

> [!IMPORTANT]
> VIORRA's database is trained exclusively on **US Common App essays**. UCAS (UK) personal statement support is in active development.

---

## 🔭 Roadmap

- [x] Core RAG + LiteRT inference pipeline
- [x] Chat follow-up with conversational memory
- [x] Session history with export/import
- [x] Local environment setup via `pip install -e .`
- [x] CLI headless analysis mode
- [x] Live hardware benchmark (`--benchmark`)
- [ ] Cloud-hosted version (centralized server)
- [ ] UCAS / UK personal statement support
- [ ] Multi-university RAG dataset expansion
- [ ] Mobile-responsive UI

---

## 👥 Team Violets

<div align="center">

| | Name | Role |
|:---:|:---|:---|
| 👑 | **Azizakhan Rustamova** | Founder |
| ⚙️ | **Sardor Qurbonov** | Lead Developer |
| 📈 | **Ruhshona Farhodova** | Business Developer |

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:6C63FF,100:A855F7&height=120&section=footer&animation=fadeIn" width="100%"/>

**Built with precision by Team Violets** · Engineered with [Google Antigravity](https://github.com/google-deepmind) AI Agent

[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-qsardor-FFD21E?style=for-the-badge&labelColor=1a1a2e)](https://huggingface.co/qsardor)
&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-qsardor/VIORRA-181717?style=for-the-badge&logo=github&labelColor=1a1a2e)](https://github.com/qsardor/VIORRA)
&nbsp;
[![Demo](https://img.shields.io/badge/Live_Demo-Try_Now-8B5CF6?style=for-the-badge&labelColor=1a1a2e)](https://huggingface.co/spaces/qsardor/VIORRA)

</div>