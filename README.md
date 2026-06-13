<div align="center">

# VIORRA

<h3 align="center">Intelligent coaching. Beautifully delivered.</h3>

<p align="center">
  The elite, mathematically-anchored college admissions essay coach.<br>
  Built to run 100% offline on your own GPU/CPU. Your essays never leave your computer.
</p>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LiteRT](https://img.shields.io/badge/LiteRT-Google-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/edge/litert)
[![Gemma](https://img.shields.io/badge/Gemma_4-E2B--IT-FF6F00?style=for-the-badge&logo=google&logoColor=white)](https://huggingface.co/google/gemma-4-E2B-it)
[![License](https://img.shields.io/badge/License-AGPL--3.0-blue?style=for-the-badge)](LICENSE)
[![Demo](https://img.shields.io/badge/🤗_Try_Demo-HuggingFace-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/qsardor/VIORRA)

---

*Paste your essay. Get the truth.*

</div>

## The Problem: Why ChatGPT, Gemini & Claude Suck For This

Students trust generic chatbots like ChatGPT, Gemini, or Claude to review their admissions essays. **This is a massive mistake.**

1. **They have "universal" brains.** Chatbots are trained to be jacks-of-all-trades. They lack the hyper-specialized focus required to act as an Ivy League admissions officer.
2. **They rely on SEO bloggers and middle-school worksheets.** Even if you use advanced prompting to force ChatGPT or Claude to "search the web," they just scrape commercial test-prep blogs (like *College Essay Guy*). They build generic rubrics out of SEO marketing content and printable brainstorming worksheets. VIORRA bypasses bloggers entirely and mathematically maps your writing against a proprietary database of the *actual raw essays* that got students into Harvard and Yale. 
3. **They are overkill and nerfed.** Using massive corporate models is overkill. Furthermore, many chatbots lock their most capable reasoning engines behind expensive monthly subscriptions. If you refuse to pay the paywall, you are restricted to basic, non-reasoning tiers that instantly generate shallow, generic advice. VIORRA gives you access to a dedicated, elite reasoning engine for free, running locally on your own hardware.
4. **They feed you into predatory marketing funnels.** If you force a chatbot to "search the web" for essay advice, it won't find actual Ivy League rubrics. Instead, the chatbot scrapes the top Google results, which are heavily-tracked commercial test-prep funnels (complete with email gates and HubSpot trackers). You aren't getting objective academic advice; you are getting graded by a corporate advertisement.
5. **They are literally blocked from reading the best data.** The most valuable datasets on the internet are protected by strict anti-bot firewalls (like Cloudflare). When a standard chatbot tries to "search the web," it gets blocked by these firewalls and is forced to hallucinate the content based on the URL title, or settle for scraping low-tier sites that allow bots. VIORRA's proprietary RAG database was constructed using advanced scraping techniques to bypass these restrictions, meaning VIORRA has access to raw, verified data that generic chatbots physically cannot see.

If general AI models were actually good at this, we wouldn't have wasted months building this software.

## The Solution

VIORRA is your smartest coach and teacher. It doesn't guess, and it refuses to pander. Unlike other AI chatbots that offer weak, sugar-coated, and generic advice, VIORRA uses a mathematically-anchored RAG database of *hundreds of successful Ivy League essays* and their official admissions feedback. It forces deep Chain-of-Thought reasoning to map your writing directly against proven success, providing sharp, objective, and deeply encouraging guidance.

Instead of generating arbitrary feedback from thin air, VIORRA uses a custom **Retrieval-Augmented Generation (RAG) database** compiled from top admissions resources. Before it evaluates your writing, it searches this dedicated database of hundreds of real, verified Ivy League essays. 

Furthermore, unlike the free "flash" models offered by Big Tech, VIORRA's core engine uses a deep **Chain-of-Thought reasoning process** before it ever delivers an answer. It retrieves the exact essays that match your theme, compares your writing directly against them, and delivers brutal, unvarnished critiques mapped directly against successful admits.

## How It Works

VIORRA uses a 2-step **"Duo" architecture** — grounding the LLM in real admissions data:

```
┌─────────────────────────────────────────────────────┐
│                    YOUR ESSAY                       │
└───────────┬─────────────────────┬───────────────────┘
            │                     │
            │          ┌──────────▼───────────────┐
            │          │  1. THE CONTEXT (RAG)    │
            │          │  Real Ivy Essays         │
            │          │  via FAISS + FastEmbed   │
            │          └──────────┬───────────────┘
            │                     │
          ┌─▼─────────────────────▼─┐
          │   2. THE BRAIN (Gemma)  │
          │   RAG injected into     │
          │   prompt via LiteRT C++ │
          └───────────┬─────────────┘
                      │
          ┌───────────▼─────────────┐
          │    VIORRA'S VERDICT     │
          │  Honest. Actionable.    │
          └─────────────────────────┘
```

| Layer | Model | Role |
|-------|-------|------|
| **Context** | [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) + [FAISS](https://github.com/facebookresearch/faiss) | Retrieves the 2 most similar real, human-graded Ivy-tier essays from a vector database |
| **Brain** | [`Gemma 4 E2B-IT`](https://huggingface.co/google/gemma-4-E2B-it) via [LiteRT](https://ai.google.dev/edge/litert) | Receives RAG examples in its prompt and uses internal Chain-of-Thought reasoning to score and review your essay |

## Features

- **📄 Smart Document Parsing** — Drop in `.pdf`, `.docx`, `.txt`, or `.md` files. VIORRA extracts the text instantly.
- **🎯 Deep Qualitative Diagnostics** — Evaluates structural flow, narrative hooks, and authenticity without hallucinating fake grading rubrics.
- **💬 Chat with Viorra** — Ask follow-up questions about your essay in a conversational interface.
- **🗃️ Session History** — Your analyses are saved locally. Export backups, import old sessions, or use **Factory Reset** to completely wipe everything and start fresh.
- **⚡ Native Hardware Support** — Runs using your system's native GPU/CPU power via LiteRT.
- **💤 Auto-Sleep** — VIORRA automatically goes to sleep after 5 minutes of inactivity to keep your laptop cool, and wakes right back up when you need it.
- **✈️ True Offline Mode** — After the first boot, no internet connection is required to evaluate essays.
- **🛡️ Rock-Solid Stability** — Built-in protections prevent crashes, auto-recover your sessions, and ensure the AI's feedback always formats perfectly.
- **🖥️ System Diagnostics** — On boot, VIORRA checks your OS, RAM, CPU, and GPU to warn you before problems happen.

---

## Quick Start

```bash
git clone https://github.com/qsardor/VIORRA.git
cd VIORRA
pip install -e .
viorra
```

That's it. VIORRA opens in your browser at `http://localhost:8000`.

> [!NOTE]
> On first launch, VIORRA automatically downloads the required Gemma LLM and vector database (~2.5 GB). On subsequent boots, it silently checks Hugging Face and natively upgrades your models if updates are available.

> **Don't want to install anything?** [Try the limited demo on HuggingFace Spaces →](https://huggingface.co/spaces/qsardor/VIORRA)

---

## Installation

### Option 1: Install from source (Recommended)

```bash
git clone https://github.com/qsardor/VIORRA.git
cd VIORRA
pip install -e .
viorra
```

### Option 2: Try the online demo

No installation needed. Visit the [HuggingFace Space](https://huggingface.co/spaces/qsardor/VIORRA) for a limited cloud demo.

> [!WARNING]
> The HuggingFace demo is a lightweight preview only. For the full experience - including chat follow-ups, session history, and document upload - install the complete application.

---

## Advanced CLI Commands

VIORRA includes several administrative flags for power users and AI agents to run headless analyses or manage the local cache:

```bash
# Headless Evaluation: Analyze an essay directly in the terminal without starting the web UI
viorra --cli sample_essay.txt

# System Diagnostics: Check your hardware compatibility (GPU/VRAM)
viorra --status

# Database Update: Force-sync the offline database and LLM with the latest Ivy League data
viorra --update

# Cache Management: Purge heavy HuggingFace/LiteRT model caches to free up disk space
viorra --clear-cache

# Factory Reset: Delete all local user data and logs to restore a fresh installation
viorra --factory-reset
```

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB+ |
| **CPU** | 4 Cores | 8+ Cores |
| **GPU** | Integrated | NVIDIA / AMD Dedicated |
| **Storage** | 4 GB free | 8 GB free |
| **OS** | Windows 10 / macOS 12 / Ubuntu 20.04 | Latest |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS (SPA) |
| Inference | Google LiteRT (C++ Engine) |
| Vector DB | FAISS |
| Embeddings | FastEmbed (ONNX) |

## Why Offline? (Still in Development)

Currently, VIORRA is operating as an **Offline-First Prototype** because it is still in active development.

Because we do not currently have the VPS (Virtual Private Server) infrastructure required to host these models in the cloud for thousands of users, we engineered VIORRA to run entirely on your local machine.

**Pricing:** VIORRA is completely **free right now** during this prototype phase. However, please note that it will not be free forever once the final version is officially launched.

> [!WARNING]
> **Current Status & Known Issues**
> As an early prototype, we are fully aware of some ongoing problems and bugs within the application. Team Violets is actively working on patches to improve stability, lower hardware requirements, and squash these bugs in future updates.

## Team Violets

| Name | Role |
|------|------|
| **Azizakhan Rustamova** | Founder & Marketing |
| **Sardor Qurbonov** | Main Developer of Software |
| **Ruhshona Farhodova** | Business Developer |
| **Damirbek Xolnazarov** | Full Stack Developer |

## Links

| Resource | Link |
|----------|------|
| Live Demo | [HuggingFace Spaces](https://huggingface.co/spaces/qsardor/VIORRA) |
| LLM Engine | [Google Gemma 4 E2B-IT](https://huggingface.co/google/gemma-4-E2B-it) |
| Embeddings | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Dataset | [qsardor/viorra-admissions-essays](https://huggingface.co/datasets/qsardor/viorra-admissions-essays) |
| Runtime | [LiteRT (formerly TFLite)](https://ai.google.dev/edge/litert) |

---

<div align="center">

**Built with precision by Team Violets**

Engineered with the assistance of [Google Antigravity](https://github.com/google-deepmind) AI Agent

</div>