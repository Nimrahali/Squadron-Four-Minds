"""
04_noisy_simulator.py
Squadron Four Minds — Mission 2: Part 4
Noisy Channel Simulator & Error Recovery Analysis Engine

Simulates channel noise injection, length truncation, and basic error-correcting 
recovery over the USSE Noosphere Subspace Transceiver link.
"""

import math
import random
from typing import Dict, List, Tuple


class NoisySubspaceChannel:
    """
    Simulates a constrained subspace communication channel subject to word-count
    limits and stochastic token substitution noise.
    """

    NOISE_VOCABULARY = [
        "Banana",
        "Labyrinth",
        "Quantum",
        "Midnight",
        "Velocity",
        "Green",
        "Cosmological",
    ]

    def __init__(self, max_words: int = 7, seed: int = 42):
        self.max_words = max_words
        self.rng = random.Random(seed)

    def transmit(self, canonical_message: str) -> str:
        """
        Transmits a canonical message through the noisy channel.
        - Truncates message to max_words.
        - Replaces the last word with a noise token.
        """
        words = canonical_message.strip().split()
        if not words:
            return self.rng.choice(self.NOISE_VOCABULARY)

        # Truncate to maximum word allowance
        truncated = words[: self.max_words]

        # Replace final word with noise token
        noise_token = self.rng.choice(self.NOISE_VOCABULARY)
        truncated[-1] = noise_token

        return " ".join(truncated)

    @staticmethod
    def calculate_token_entropy(tokens: List[str]) -> float:
        """Calculates Shannon Entropy H(X) in bits for a sequence/distribution of tokens."""
        if not tokens:
            return 0.0

        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1

        total = len(tokens)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)

        return entropy


class RepetitionDecoder:
    """
    Implements a simple Majority-Vote Parity/Repetition Code decoder across N 
    noisy transmissions to recover the original message prefix and detect noisy slots.
    """

    def __init__(self, channel: NoisySubspaceChannel):
        self.channel = channel

    def transmit_and_decode(self, canonical_message: str, repetitions: int = 5) -> Tuple[str, float]:
        """
        Transmits the message `repetitions` times and applies majority voting per word position.
        Returns the decoded message and the empirical error recovery rate.
        """
        transmissions = [self.channel.transmit(canonical_message) for _ in range(repetitions)]
        tokenized = [t.split() for t in transmissions]
        
        # Determine average length
        max_len = max(len(t) for t in tokenized)
        decoded_words = []

        for i in range(max_len):
            position_tokens = [t[i] for t in tokenized if i < len(t)]
            # Find majority token
            most_common = max(set(position_tokens), key=position_tokens.count)
            
            # If the majority token is a known noise token and appears sporadically, mark as noise
            if most_common in NoisySubspaceChannel.NOISE_VOCABULARY and position_tokens.count(most_common) < (repetitions / 2):
                decoded_words.append("[CORRUPTED]")
            else:
                decoded_words.append(most_common)

        decoded_message = " ".join(decoded_words)
        
        # Calculate word-level accuracy against original canonical prefix
        original_words = canonical_message.split()[: self.channel.max_words]
        matches = sum(1 for a, b in zip(decoded_words, original_words) if a == b)
        accuracy = matches / len(original_words) if original_words else 1.0

        return decoded_message, accuracy


def run_channel_simulation():
    """Executes channel baseline diagnostics and recovery simulations."""
    print("=================================================================")
    print(" USSE NOOSPHERE: SUBSPACE CHANNEL NOISE SIMULATOR DIAGNOSTICS")
    print("=================================================================\n")

    channel = NoisySubspaceChannel(max_words=7, seed=7)
    decoder = RepetitionDecoder(channel)

    test_messages = [
        "Provide topic and goal for analysis immediately",
        "Confirm yes or no to save parameters",
        "System integrity restored to fourteen percent",
    ]

    print("--- 1. SINGLE TRANSMISSION CHANNEL CORRUPTION TRACE ---")
    for msg in test_messages:
        noisy_output = channel.transmit(msg)
        print(f"Canonical : '{msg}'")
        print(f"Transmitted: '{noisy_output}'\n")

    print("--- 2. REPETITION DECODING & ERROR RECOVERY ANALYSIS ---")
    for msg in test_messages:
        decoded, acc = decoder.transmit_and_decode(msg, repetitions=5)
        print(f"Original Target : '{msg}'")
        print(f"Decoded Output  : '{decoded}'")
        print(f"Recovery Rate   : {acc * 100:.1f}%\n")

    # Noise Vocabulary Max Entropy Evaluation
    vocab_entropy = NoisySubspaceChannel.calculate_token_entropy(NoisySubspaceChannel.NOISE_VOCABULARY)
    print("--- 3. NOISE VOCABULARY ENTROPY EVALUATION ---")
    print(f"Theoretical Max Noise Vocabulary Entropy H(N): {math.log2(7):.3f} bits/substitution")
    print(f"Empirical Sampled Entropy H(N)              : {vocab_entropy:.3f} bits/substitution")
    print("=================================================================")


if __name__ == "__main__":
    run_channel_simulation()