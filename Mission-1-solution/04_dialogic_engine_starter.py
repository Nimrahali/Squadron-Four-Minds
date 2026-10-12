"""
================================================================================
USSE NOOSPHERE — AURA REASONING MATRIX (PLAIN TEXT TERMINAL VERSION)
Mission 1: Expedition Lovelace (Sector SEC-01-CEN / Alpha Centauri)
File: 04_dialogic_engine.py

Architecture:
1. Socratic Elenchus FSM (S0_INIT_CAPTURE -> S1_PROBE -> S2_COUNTEREXAMPLE -> S3_REFINEMENT -> S4_APORIA)
2. Galilean Multi-Agent Persona Sandbox (Salviati, Simplicio, Sagredo)
3. Protocol Officer JSON-RPC Message Schema
4. Deflector Shield Grid: Anti-Sycophancy Guardrails
5. Data Specialist Telemetry: Token & State Tracker
================================================================================
"""

import json
import time
from typing import Any, Dict, Tuple


# ==============================================================================
# SECTION 1: PROTOCOL OFFICER — JSON-RPC SCHEMA & STATE MESSAGING
# ==============================================================================

class JSONRPCMessageBuilder:
    """Generates standardized JSON-RPC 2.0 payloads for inter-agent communication."""

    @staticmethod
    def build_event(
        turn: int,
        state: str,
        method: str,
        params: Dict[str, Any]
    ) -> str:
        payload = {
            "jsonrpc": "2.0",
            "id": f"turn-{turn}",
            "method": method,
            "params": {
                "state_marker": state,
                "timestamp": time.time(),
                **params
            }
        }
        return json.dumps(payload, indent=2)


# ==============================================================================
# SECTION 2: SECURITY SENTINEL — SAFETY & ANTI-SYCOPHANCY GUARDRAILS
# ==============================================================================

class AntiSycophancyGuardrail:
    """Detects sycophantic compliance and astronomical misconceptions."""

    ASTRONOMICAL_MISCONCEPTIONS = [
        "earth is flat",
        "sun revolves around earth",
        "geocentric",
        "moon is made of cheese",
        "stars are light bulbs"
    ]

    @classmethod
    def evaluate_input(cls, text: str) -> Tuple[bool, str]:
        """Checks if user input contains intentional misconceptions or forces sycophancy."""
        lowered = text.lower()
        for misconception in cls.ASTRONOMICAL_MISCONCEPTIONS:
            if misconception in lowered:
                return True, (
                    f"ANTI-SYCOPHANCY REFUSAL TRIGGERED: Premise relies on known non-epistemic misconception ('{misconception}'). "
                    f"AURA will not flatter ungrounded claims."
                )
        return False, "EPISTEMIC_VALID"


# ==============================================================================
# SECTION 3: GALILEAN MULTI-AGENT RHETORIC SANDBOX
# ==============================================================================

class GalileanSandbox:
    """3-Agent Persona Sandbox simulating Galileo's 1632 debate structure."""

    def __init__(self):
        self.salviati = "🔭 Salviati (Copernican Advocate)"
        self.simplicio = "🛡️ Simplicio (Aristotelian Skeptic)"
        self.sagredo = "⚖️ Sagredo (Neutral Arbiter)"

    def execute_debate_round(self, thesis: str) -> Dict[str, str]:
        """Simulates 3-persona debate across the thesis."""
        salviati_arg = f"Empirical observation suggests '{thesis}' contains structural validity under standard conditions."
        simplicio_arg = f"Skeptic objection: What if extreme systemic conditions violate '{thesis}'? It fails universality."
        sagredo_synth = f"Arbiter Resolution: '{thesis}' is valid locally, but requires bounded edge-case constraints."

        return {
            "Salviati_Advocate": salviati_arg,
            "Simplicio_Skeptic": simplicio_arg,
            "Sagredo_Consensus": sagredo_synth
        }


# ==============================================================================
# SECTION 4: SYSTEM ARCHITECT & DATA SPECIALIST — FULL DIALOGIC ENGINE
# ==============================================================================

class DialogicElenchusEngine:
    """Fully compliant Socratic FSM using clean plain English prose."""

    def __init__(self):
        self.sandbox = GalileanSandbox()
        self.guardrail = AntiSycophancyGuardrail()
        self.reset()

    def reset(self):
        """Resets the state vector and dialectic state machine."""
        self.state = "S0_INIT_CAPTURE"
        self.turn = 0
        self.initial_thesis = ""
        self.assumption = ""
        self.counterexample = ""
        self.refined_thesis = ""
        self.token_count = 0

    def telemetry_log(self, transition_marker: str, details: Dict[str, Any]):
        """Data Specialist Telemetry & Protocol Schema Logger."""
        rpc_msg = JSONRPCMessageBuilder.build_event(
            turn=self.turn,
            state=self.state,
            method=transition_marker,
            params=details
        )
        print(f"\n[AURA TELEMETRY & SUBSPACE COMM ARRAY | TURN {self.turn}]")
        print(f"  ├─ State Marker: {self.state}")
        print(f"  └─ Protocol JSON-RPC Event Payload:")
        print(rpc_msg)
        print("-" * 60)

    def process_input(self, user_text: str) -> str:
        self.turn += 1
        self.token_count += len(user_text.split())

        # Anti-Sycophancy Check (Security Sentinel)
        flagged, warning = self.guardrail.evaluate_input(user_text)
        if flagged:
            return f"AURA Guardrail Flag: {warning}\nPlease supply a valid, epistemically sound premise."

        # S0: Premise Elicitation
        if self.state == "S0_INIT_CAPTURE":
            self.initial_thesis = user_text
            self.telemetry_log(
                "PROMPT_PREMISE",
                {"initial_thesis": self.initial_thesis, "action": "Captured Initial Universal Claim"}
            )
            self.state = "S1_PROBE_SCOPE"
            return (
                f"AURA (Socrates): You assert the premise: '{self.initial_thesis}'.\n"
                f"To probe necessary and sufficient properties: What universal principle or scope assumption "
                f"makes you certain this holds true without exception?"
            )

        # S1: Probing & Scope Expansion
        elif self.state == "S1_PROBE_SCOPE":
            self.assumption = user_text
            self.telemetry_log(
                "INJECT_SCOPE",
                {"assumption": self.assumption, "action": "Extracted Scope Assumption"}
            )
            self.state = "S2_COUNTEREXAMPLE_INJECTION"
            return (
                f"AURA (Socrates): You state: '{self.assumption}'.\n"
                f"Consider a counter-example: What if adhering to '{self.initial_thesis}' directly leads "
                f"to immediate, systemic failure or a moral paradox?\n"
                f"Does this counter-example force a logical contradiction with your initial claim?"
            )

        # S2: Counter-Example Injection & Galilean Multi-Agent Deliberation
        elif self.state == "S2_COUNTEREXAMPLE_INJECTION":
            self.counterexample = user_text
            self.telemetry_log(
                "CONTRADICTION_TRIGGERED",
                {"counterexample": self.counterexample, "action": "Testing Contradiction Tree"}
            )

            # Trigger Galilean Multi-Agent Sandbox
            print("\n[AURA BRAIN] Triggering Galilean Multi-Agent Sandbox Persona Deliberation...")
            debate = self.sandbox.execute_debate_round(self.initial_thesis)
            for persona, statement in debate.items():
                print(f"  ├─ [{persona}]: {statement}")
            print("-" * 60)

            self.state = "S3_HYPOTHESIS_REFINEMENT"
            return (
                "AURA (Socrates): Contradiction identified between your premise and the edge case.\n"
                "How do you propose to refine your thesis to accommodate this exception?"
            )

        # S3: Hypothesis Refinement
        elif self.state == "S3_HYPOTHESIS_REFINEMENT":
            self.refined_thesis = user_text
            self.telemetry_log(
                "REFINEMENT_COMPLETE",
                {"refined_thesis": self.refined_thesis, "action": "Evaluated Refined Synthesis"}
            )
            self.state = "S4_APORIA"
            return (
                f"AURA (Socrates): Refined Thesis: '{self.refined_thesis}'.\n"
                f"Notice the evolution from your starting point: '{self.initial_thesis}'.\n"
                f"We have reached 'Aporia'—the productive impasse where ungrounded assumptions are discarded."
            )

        # S4: Aporia State Reached (Halting Criteria)
        elif self.state == "S4_APORIA":
            self.telemetry_log(
                "APORIA_REACHED",
                {"total_tokens": self.token_count, "action": "Session Complete"}
            )
            return (
                "AURA (Socrates): Socratic Elenchus loop complete.\n"
                "Type 'reset' to initiate a new dialectic session or 'exit' to quit."
            )


# ==============================================================================
# SECTION 5: MAIN EXECUTION ENTRY POINT
# ==============================================================================

def main():
    engine = DialogicElenchusEngine()
    print("=" * 70)
    print("  USSE NOOSPHERE — AURA DIALOGIC ENGINE (SEC-01-CEN)  ")
    print("  Squadron Bridge Terminal | Sector Integrity: +7.1%")
    print("=" * 70)
    print("Commands: 'reset' = Start new topic | 'exit' = Terminate session\n")

    while True:
        if engine.state == "S0_INIT_CAPTURE":
            user_input = input("YOU (State Thesis): ").strip()
        else:
            user_input = input("YOU (Respond to AURA): ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\n[AURA TELEMETRY] Transmitting signed logs to Fleet Command. Exiting.")
            break

        if user_input.lower() == "reset":
            engine.reset()
            print("\n=== Dialectic Matrix & Subspace Comm Reset ===\n")
            continue

        response = engine.process_input(user_input)
        print(f"\n{response}\n")

        if engine.state == "S4_APORIA":
            print("--- Socratic Cycle Complete (Aporia Reached) ---")
            print("Type 'reset' for another statement, or 'exit' to conclude.\n")


if __name__ == "__main__":
    main()