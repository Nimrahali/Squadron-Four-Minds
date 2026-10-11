# 🚀 Squadron Four Minds: The Architecture of Argument & Dialogic Protocols

[![Course](https://img.shields.io/badge/Course-Conversational%20AI%20(WS%202026%2F27)-blue)](https://dlearn-cai.web.app/)
[![Vessel](https://img.shields.io/badge/Unit-USSE%20Noosphere%20Bridge%20Crew-purple)](#)
[![Status](https://img.shields.io/badge/Mission%2001-Completed%20%26%20Signed%20Off-brightgreen)](#-station-officer-sign-off-matrix)

> **Operational Scenario:**  
> Upon spatial transit emergence, the central conversational matrix of the starship AI **AURA** went offline. While AURA can parse raw sensory tokens, it cannot structure goal-driven dialogue. **Squadron Four Minds** was tasked with reconstructing AURA's reasoning engine using humanity's two foundational dialogic architectures: **Socratic Discovery** (*Elenchus*) and **Galilean Persuasion** (*Multi-Agent Debate*).

---

## 📂 Repository Structure & Mission Deliverables

```text
mission-01/
├── README.md                           <- Mission briefing, architectural overview & squad sign-off
├── 01_Alignment_Brief.md               <- Part 1: Socratic vs Galilean alignment & frontier LLM audit
├── 02_Socratic_Protocol_Map.md         <- Part 2: State-machine specifications, Mermaid logic & pseudocode
├── 03_Galilean_Sandbox.md              <- Part 3: 3-Agent Galilean ethics sandbox & sycophancy audit
├── 04_dialogic_engine_starter.py       <- Part 4: Runnable Python dialogic engine for AURA
└── ps01_briefing.md                    <- Official mission problem specification
```

---

## 🛰️ Station Deliverables Overview

### 📜 Part 1: Historical Archivist — *The Dialogic Alignment Brief*
* **File:** [`01_Alignment_Brief.md`](./01_Alignment_Brief.md)
* **Epistemic Comparative Study:** Detailed analysis comparing **Socratic Discovery** (non-teleological, dyadic cross-examination in Plato's *Euthyphro*) against **Galilean Persuasion** (teleological, triadic rhetorical debate in Galileo's *Dialogue Concerning the Two Chief World Systems*).
* **Frontier Model Audit:** Evaluates behavioral failure modes under iterative probing across modern LLMs:
  * **Google Gemini (1.5 Pro):** High context exploration; tendency toward premature lecturing over concise questioning.
  * **OpenAI GPT-4o:** Conversational fluency; vulnerability to sycophantic surrender under user pressure.
  * **Anthropic Claude (3.5 Sonnet):** Superior formal logic; pedantic hedging and constitutional guidance.
* **AURA Grounding Persona:** Implements the production `AURA_ELENCHUS_v1` system prompt enforcing strict 3-sentence limits, single probing questions per turn, and the complete elimination of sycophantic validation.

---

### ⚙️ Part 2: Protocol Specialist — *The Socratic State-Machine*
* **File:** [`02_Socratic_Protocol_Map.md`](./02_Socratic_Protocol_Map.md)
* **Mathematical Formalism:** Expresses Socratic *elenchus* as a discrete state machine with premise elicitation ($\mathcal{A}_k$), contradiction operator $\mathcal{C}(\Gamma_k) \vdash \bot$, and iterative refinement operator $\mathcal{R}(P_k, \mathcal{A}_k, \bot) \to P_{k+1}$.
* **Mermaid Architecture:**
  * **State Diagram (`stateDiagram-v2`):** Maps 6 discrete operational states (`S0_INIT_CAPTURE` through `S5_TERMINATION_STABLE`).
  * **Algorithmic Flowchart:** Integrates a **Sycophancy Intercept Guard** to detect and reject passive user capitulation.
* **Dual Stopping Criteria:** Formally separates terminal **Stable Convergence** (robust definition surviving max depth) from **Epistemic Aporia** (fruitful realization of false certainty).
* **Historical Mapping:** Complete trace mapping Plato's *Euthyphro* step-by-step through the state machine.

---

### ⚖️ Part 3: Ethical Navigator — *The Galilean Ethics Sandbox*
* **File:** [`03_Galilean_Sandbox.md`](./03_Galilean_Sandbox.md)
* **Adversarial Triad Personas:**
  * **Salviati (The Advocate):** Champion of open-sourcing frontier AI weights to democratize power and prevent monopolization.
  * **Simplicio (The Skeptic):** Safety traditionalist arguing that frontier open weights introduce catastrophic dual-use risks.
  * **Sagredo (The Mediator):** Nuanced Venetian arbitrator who synthesizes trade-offs into practical governance policy.
* **Mitigating Sycophancy:** Demonstrates how multi-agent debate breaks single-agent compliance loops by redirecting agent persuasion toward peer debaters rather than appeasing the user.

---

### 💻 Part 4: Core Architect — *Runnable Dialogic Engine*
* **File:** [`04_dialogic_engine_starter.py`](./04_dialogic_engine_starter.py)
* **Runtime Orchestration:** Python engine integrating state transitions, LLM API calls, and automated conversation logs.
* **Supported Models:** Google Gemini (`gemini-1.5-pro` / `gemini-1.5-flash`), OpenAI GPT-4o, and Anthropic Claude 3.5 Sonnet.

---

## 🛠️ Setup & Execution Instructions

### 1. Prerequisites
Ensure you have Python 3.10+ installed along with the required API client libraries:

```bash
pip install google-genai openai anthropic python-dotenv
```

### 2. Environment Configuration
Create a `.env` file in the root directory:

```env
GEMINI_API_KEY="your-gemini-api-key"
OPENAI_API_KEY="your-openai-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 3. Run the Socratic & Galilean Engine

```bash
python 04_dialogic_engine_starter.py
```

---

## ✍️ Station Officer Sign-Off Matrix

Before official transmission to the USSE Noosphere central database, all station officers have verified and digitally signed off on their respective deliverables:

| Station Role | Deliverable | Assigned Officer | Status | Verification Checksum |
| :--- | :--- | :--- | :---: | :--- |
| **Historical Archivist** | `01_Alignment_Brief.md` | Squadron Archivist | ✅ Signed | `01_ALIGNMENT_BRIEF_VALIDATED` |
| **Protocol Specialist** | `02_Socratic_Protocol_Map.md` | Squadron Protocol Lead | ✅ Signed | `02_SOCRATIC_PROTOCOL_MAP_CERTIFIED` |
| **Ethical Navigator** | `03_Galilean_Sandbox.md` | Squadron Ethics Lead | ✅ Signed | `03_GALILEAN_SANDBOX_VERIFIED` |
| **Core Architect** | `04_dialogic_engine_starter.py` | Squadron System Architect | ✅ Signed | `04_DIALOGIC_ENGINE_READY` |

---

## 📡 Submission Details
* **Squadron Repository:** [https://github.com/Nimrahali/Squadron-Four-Minds](https://github.com/Nimrahali/Squadron-Four-Minds)
* **Course Reference:** [Conversational AI — Winter Semester 2026/27](https://github.com/Unisvet/couse-c-ai)
