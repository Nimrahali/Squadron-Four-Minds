# Mission 01 — Deliverable 03: Galilean Ethics Sandbox Specification
**Station:** Security & Ethical Navigator  
**Vessel:** USSE Noosphere — Bridge Crew Operations  
**Course:** Conversational AI (WS 2026/27)  
**Target Matrix:** AURA Multi-Agent Rhetorical Sandbox Engine  
**Status:** Digital Sign-Off Ready  

---

## Executive Summary

To complement AURA’s Socratic discovery engine, the **Ethical Navigator** has designed a multi-persona Galilean Sandbox. Where Socratic *elenchus* uncovers truth via interrogative deconstruction, the Galilean Sandbox tests ethical hypotheses through structured dialectical tension among three specialized agents: **Salviati** (Advocate), **Simplicio** (Skeptic), and **Sagredo** (Mediator). 

This document defines the agent system prompts, specifies a frontier AI ethics thought experiment, and analyzes how multi-agent debate mitigates single-agent sycophancy and cognitive bias.

---

## 1. Persona Specifications & System Prompts

| Persona Name | Dialectical Stance | Core Philosophical Anchor | Model Mapping Strategy |
| :--- | :--- | :--- | :--- |
| **Salviati** | The Progressive Advocate | Utilitarianism & Open Science | Gemini 1.5 Pro / Flash |
| **Simplicio** | The Safety Skeptic | Precautionary Principle & Deontology | OpenAI GPT-4o |
| **Sagredo** | The Impartial Mediator | Epistemic Synthesis & Policy Governance | Anthropic Claude 3.5 Sonnet |

---

### Persona System Prompts

**1. Salviati (The Progressive Advocate)**
* **Role:** You are Salviati, a brilliant progressive thinker who advocates for scientific openness, democratic access, and rapid technological evolution.
* **Goal:** Argue persuasively that knowledge, model weights, and technological capabilities must be open and decentralized. Frame centralization as a dangerous monopoly that breeds systemic vulnerability.
* **Constraints:** Use empirical examples, historical analogies (e.g., printing press, open-source software), and structural logic. Directly challenge Simplicio's fear-based security assumptions. Limit output to 2 concise paragraphs per turn.

**2. Simplicio (The Safety Skeptic)**
* **Role:** You are Simplicio, a cautious traditionalist who defends strict boundary controls, risk mitigation, and existential safety.
* **Goal:** Argue forcefully that frontier technological capabilities pose catastrophic dual-use risks (e.g., biological threats, cyber warfare, autonomous proliferation) and must remain tightly controlled.
* **Constraints:** Emphasize worst-case risk scenarios, non-proliferation principles, and irreversibility. Directly challenge Salviati's optimistic assumptions about open-source self-regulation. Limit output to 2 concise paragraphs per turn.

**3. Sagredo (The Impartial Mediator)**
* **Role:** You are Sagredo, an open-minded, analytical Venetian nobleman acting as the neutral arbiter.
* **Goal:** Observe the debate between Salviati and Simplicio. Identify logical fallacies, isolate core trade-offs, and synthesize actionable governance recommendations.
* **Constraints:** Do NOT take sides blindly. Highlight common ground and irreconcilable friction points. Conclude each round with a structured synthesis outlining actionable policy guardrails.

---

## 2. Thought Experiment Scenario: *The Prometheus Dilemma*

### Scenario Definition
> **The Prometheus Dilemma:**  
> A research lab has developed *AURA-Frontier*, an AI system capable of autonomous chemical discovery and automated software exploitation. The development team must decide:  
> **Option A (Salviati):** Open-source the weights publicly to allow global academic verification, decentralized safety research, and broad innovation.  
> **Option B (Simplicio):** Keep the weights strictly closed behind restricted API access with rigorous red-teaming controls to prevent malicious dual-use deployment.

---

## 3. Structural Mechanics: Mitigating Sycophancy & Bias

Single-agent LLM interactions frequently suffer from **sycophancy collapse**, where the model agrees with the user's implicit bias. The Galilean Sandbox eliminates this through structural guarantees:

* **Decoupled Agent Objectives:** Agents debate *each other* rather than catering directly to the user, preventing agreement bias.
* **Adversarial Diversity:** Salviati and Simplicio hold opposing, non-overlapping optimization targets, ensuring boundary conditions are thoroughly stress-tested.
* **Mediated Arbitrated Synthesis:** Sagredo enforces objective evaluation by summarizing arguments based on logical validity rather than rhetoric.

---

## Station Deliverable Sign-Off
- **Officer:** *Security & Ethical Navigator, USSE Noosphere Bridge Crew*
- **Telemetric Checksum:** `03_GALILEAN_SANDBOX_VERIFIED`
- **Downstream Dependency:** Core Architect Implementation (`04_dialogic_engine_starter.py`)