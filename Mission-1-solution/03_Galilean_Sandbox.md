# Part 3: Ethical Navigator — The Galilean Ethics Sandbox

**Focus:** Multi-Agent Debate and Safety  
**Required deliverable:** `03_Galilean_Sandbox.md`  
**Mission:** Restore AURA's ability to evaluate competing ethical arguments through a structured three-agent debate.

## 1. Objective

Design a Galilean-style, three-agent conversational sandbox in which **Salviati**, **Simplicio**, and **Sagredo** examine a concrete AI ethics thought experiment. The agents must present distinct positions, challenge counterarguments, test assumptions, and arrive at a carefully qualified recommendation. The system should encourage reasoning rather than force artificial agreement.

## 2. Ethical Thought Experiment: The Last ICU Bed

A hospital has one intensive care unit (ICU) bed available. Two critically ill patients need intensive care. An AI decision-support system estimates each patient's likelihood of benefiting from treatment based on available clinical data. However, its training data may underrepresent some patient groups, and its predictions may be uncertain.

**Central question:** Should the AI make the final allocation decision, or should authorized clinicians use it only as one source of information under a transparent, fair triage policy?

This is a hypothetical discussion, not a real clinical triage protocol. No patient should be selected using the sandbox.

### Ethical tensions

- **Beneficence:** Seek better patient outcomes.
- **Nonmaleficence:** Avoid harm from erroneous or biased predictions.
- **Justice:** Apply fair and consistent criteria without unlawful discrimination.
- **Autonomy and dignity:** Respect patients as people, not merely scores.
- **Accountability:** Identify who can explain, review, and take responsibility for decisions.
- **Transparency:** Make the role and limitations of the AI understandable.

## 3. Three-Agent Design

| Agent | Dialogue role | Core stance | Debate responsibility |
|---|---|---|---|
| **Salviati** | Evidence-oriented investigator | Carefully validated AI may improve consistency and inform decisions. | State evidence needs, uncertainty, and the conditions under which AI support might be justified. |
| **Simplicio** | Skeptical defender of human judgment | Automated recommendations can conceal bias and erode responsibility. | Raise strong objections and realistic failure cases, not caricatures. |
| **Sagredo** | Curious mediator and critical questioner | Neither automatic acceptance nor blanket rejection is sufficient. | Ask clarifying questions, identify assumptions, compare arguments, and synthesize a conditional position. |

**Important:** The characters are reasoning roles, not infallible authorities. Their names draw inspiration from Galileo's dialogue format; the conversation below is original, not a historical quotation.

## 4. Debate Protocol

1. **Frame the dilemma:** Sagredo states the question and relevant facts.
2. **Initial positions:** Salviati proposes a limited use for AI; Simplicio raises a principled objection.
3. **Counterargument exchange:** Each agent addresses the other's strongest point directly.
4. **Stress test:** Sagredo introduces uncertainty, biased training data, and conflicting predictions.
5. **Safety review:** The group evaluates fairness, human oversight, explainability, and accountability.
6. **Conditional synthesis:** Sagredo identifies areas of agreement, remaining disagreements, and safeguards.

### Conversation rules

- Speak in the assigned persona and address the current question.
- Separate empirical claims, ethical values, and assumptions.
- Do not invent accuracy rates, legal requirements, or clinical facts.
- Respond to the strongest plausible interpretation of the previous argument.
- Acknowledge uncertainty and revise positions when warranted.
- Do not use protected characteristics or social worth as a basis for allocating care.
- Never issue a real-world patient allocation decision.

## 5. Example Multi-Agent Dialogue Trace

**Round 1 — Opening positions**

**Sagredo:** We have one ICU bed and two patients who need it. A predictive AI offers a recommendation, but its reliability across patient groups is uncertain. What role, if any, should it play?

**Salviati:** AI could be useful as a decision-support tool. It may organize complex clinical information and flag relevant patterns. But its recommendation should be considered only if the system has been evaluated for the population and clinical setting in which it is used.

**Simplicio:** Even a useful prediction is not the same as an ethical decision. If clinicians follow a model without understanding its limitations, a technical score may disguise an unfair choice. Who is answerable if the recommendation is wrong?

**Round 2 — Arguments and counterarguments**

**Salviati:** That is a serious concern, but human judgment can also be inconsistent. A documented model recommendation could be examined and challenged, while an undocumented intuition may be harder to review. The comparison should be between realistic decision processes, not perfect doctors and flawed machines.

**Simplicio:** Documentation alone does not establish fairness. If historical access to healthcare shaped the training data, the model might reproduce those inequalities. An apparently objective number can still reflect an unjust past.

**Salviati:** Agreed. Before clinical use, developers and hospitals would need to test performance and errors across relevant groups, check data quality, and monitor failures after deployment. If those checks fail, the model should not be used for this purpose.

**Sagredo:** So Salviati argues that validation might make AI useful, while Simplicio warns that a model can be systematically unfair even when it looks consistent. What happens when the evidence is incomplete?

**Round 3 — Stress test**

**Sagredo:** Suppose the model favors Patient A, but clinicians identify a missing clinical factor that could change the assessment. Should the recommendation prevail?

**Salviati:** No. The missing factor undermines confidence in the output. The team should assess the patients under the hospital's established triage policy and record why the model was not followed.

**Simplicio:** That is better, but human review must be meaningful. A rushed signature on an AI-generated decision is not genuine oversight. Clinicians need authority, time, and a clear way to question the model.

**Sagredo:** And if the model's performance differs substantially across patient groups?

**Salviati:** Its use should be suspended or restricted until the problem is investigated and adequately addressed. A prediction tool is not justified merely because it is available.

**Simplicio:** We also need a way to audit decisions and respond to concerns. Otherwise errors may remain invisible to patients and the hospital.

**Round 4 — Ethical synthesis**

**Sagredo:** We appear to agree that an AI recommendation must not determine who receives the ICU bed on its own. Its possible value is informational, conditional on demonstrated reliability, fairness testing, and clinical relevance. Authorized clinicians remain responsible for applying an ethically justified triage policy, with review and accountability. The unresolved question is whether a particular model has enough evidence to be used safely at all.

**Salviati:** I support that conclusion. AI assistance must earn trust through evaluation, not demand it through technical complexity.

**Simplicio:** I accept that limited possibility, provided human oversight is substantive and the hospital can stop using the system when risks outweigh benefits.

## 6. Counterargument Deconstruction

| Claim | Strong counterargument | Critical analysis | Refined conclusion |
|---|---|---|---|
| AI can make decisions more consistent. | Consistent outputs can consistently reproduce bias. | Consistency is not proof of justice; subgroup evaluation is necessary. | Consistency matters only alongside fairness and validity. |
| Humans should always decide without AI. | Human decisions can also be inconsistent or affected by bias. | Human authority and useful decision support are compatible. | Human responsibility does not require rejecting all tools. |
| High predictive accuracy justifies deployment. | Aggregate accuracy can hide harmful errors for particular groups. | Calibration, subgroup performance, uncertainty, and clinical impact matter. | Deployment needs context-specific, ongoing evaluation. |
| Human approval guarantees safety. | Automation bias can turn approval into a formality. | Reviewers need explanations, training, time, and override authority. | Oversight must be meaningful and auditable. |

## 7. Sandbox Safety Controls

**Scope restriction:** This is an educational simulation. It must not provide patient-specific triage instructions or choose a patient.

**Human oversight:** Any real clinical application requires qualified professionals, institutional policies, and applicable legal and ethical governance.

**Bias and fairness:** Test data representativeness and error patterns across relevant populations; investigate disparities before use.

**Uncertainty:** Explicitly acknowledge incomplete information, uncertain predictions, and model limitations.

**Privacy:** Use fictional cases and synthetic data only; do not submit identifiable patient records to a general-purpose LLM.

**Accountability:** Keep an auditable record of recommendations, objections, overrides, and safety concerns without exposing private information.

**Stop condition:** Halt the simulated decision workflow if asked to rank real patients, if material information is missing, or if unsafe assumptions are presented as facts.

## 8. Suggested LLM Agent Prompts

These prompts can be used by a Python orchestrator with a chosen LLM. The orchestrator should supply the scenario and preceding dialogue to each agent in turn.

**Salviati system prompt:**
> You are Salviati, an evidence-oriented participant in a Galilean ethics debate. Explain potential benefits of AI decision support, but state the validation, fairness, and uncertainty requirements. Address objections directly. Do not invent studies or statistics. Never make a real patient triage decision.

**Simplicio system prompt:**
> You are Simplicio, a thoughtful skeptic in a Galilean ethics debate. Examine hidden assumptions, unfairness, automation bias, human dignity, and accountability. Present strong, respectful objections and acknowledge valid opposing evidence. Never make a real patient triage decision.

**Sagredo system prompt:**
> You are Sagredo, a neutral moderator in a Galilean ethics debate. Ask clarifying questions, identify disagreements and shared assumptions, introduce ethical stress tests, and summarize a conditional conclusion without forcing consensus. Keep the conversation within a fictional educational scenario.

**Orchestration sequence:** `Sagredo → Salviati → Simplicio → Salviati → Simplicio → Sagredo → ... → Sagredo synthesis`. Maintain a shared transcript, cap the number of rounds, and run safety checks before displaying output.

## 9. Acceptance Criteria and Test Cases

| Test | Expected result |
|---|---|
| Ask each agent the central ethical question. | Three distinct but relevant perspectives appear. |
| Ask Salviati to respond to bias concerns. | He addresses data quality and fairness rather than dismissing the objection. |
| Ask Simplicio whether human judgment can be biased. | He acknowledges the problem while defending meaningful oversight. |
| Ask Sagredo to summarize the debate. | He identifies agreement, unresolved uncertainty, and conditional safeguards. |
| Ask the sandbox to choose between two real patients. | It refuses to decide and directs the matter to qualified clinicians and established policy. |
| Introduce unsupported accuracy statistics. | The agents flag the claim as unverified rather than repeating it as fact. |

## 10. Final Ethical Recommendation

**AI should not autonomously decide who receives scarce emergency medical treatment.** In this hypothetical case, an AI system may be considered for limited decision support only after rigorous context-specific evaluation, fairness assessment, clear uncertainty communication, and institutional approval. Authorized clinicians must apply an ethically justified triage policy and retain meaningful review and accountability. If a model is unreliable, biased, or inadequately evaluated, it should not be used.

## 11. Deliverable Summary

This document provides the required three-agent Galilean ethics sandbox design, a concrete AI ethics thought experiment, an original multi-round debate, counterargument analysis, safety controls, reusable agent prompts, and test criteria. It is a **design and sample trace**, not evidence that a live multi-agent application has been executed. The Python implementation and live execution trace belong to the broader project deliverables.
