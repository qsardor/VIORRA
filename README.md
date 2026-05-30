# VIORRA: The Zero-Hallucination Admissions Evaluator

## 🚀 Why It Was Created & Why It Is Needed
In the age of AI, students and institutions frequently turn to Large Language Models (LLMs) to grade or evaluate admissions essays. However, standard LLMs suffer from a critical flaw: **Hallucination**. 

If you feed an essay to a standard chatbot, its grading will fluctuate based on token temperature. It might give a terrible essay a passing grade simply because it felt "generous," or it might invent fake grammar rules that don't exist. There is no mathematical consistency.

**VIORRA** was built to solve this. It is a strict, brutal, and highly accurate AI admissions counselor that relies on a **"Zero-Hallucination"** architecture. It anchors the creative power of an LLM to undeniable, deterministic mathematical models so it can never hallucinate a score.

## ⚡ The Difference: How It Works
Instead of relying a single massive cloud LLM, VIORRA runs entirely on your local hardware using a 3-step "Trio" architecture:

1. **The Anchor (AES Scorer):** Before the LLM even sees the essay, it is processed by an Encoder-Only Automated Essay Scoring model (`Kevintu/Engessay_grading_ML`). This deterministic model guarantees a mathematically flawless structural score (e.g., 2.66/5.0).
2. **The Context (RAG Database):** The essay is embedded (`all-MiniLM-L6-v2`) and compared against a FAISS Vector Database. This retrieves real, human-graded historical examples.
3. **The Brain (Gemma 4):** Finally, `google/gemma-4-E2B-it` is fed the deterministic AES score and the human-graded RAG examples. Because it is anchored to reality, Gemma acts purely as a reasoning engine, generating strict, actionable feedback and identifying exact spelling/grammatical errors without hallucinating.

## 💻 System Architecture & Requirements
This version of VIORRA has been completely rewritten from Gradio to a high-performance **Local Web App**. 
* **Backend:** FastAPI Server serving local HTTP.
* **Frontend:** Vanilla HTML/CSS/JS Single Page Application (SPA).
* **Inference Engine:** Google's `LiteRT` C++ Engine.
* **Hardware Acceleration:** Native PyTorch GPU Detection wrapping WebGPU/Direct3D 12 (Windows) or Metal/MPS (macOS).

## 🚀 Installation & Usage
You can run VIORRA entirely locally on your own machine.

1. Clone the repository and install requirements.
2. Build the package: `pip install build` -> `python -m build`
3. Install the generated `.whl` package via pip.
4. Type `viorra` in your terminal to boot the local server!

## 🏆 Credits

* **Founder & Vision** – Azizakhan Rustamova
* **Developer & AI Engineer** – Qurbonov Sardor
