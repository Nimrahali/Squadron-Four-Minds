"""
Mission 05 — Part 2: Communications & Protocol Officer
BM25 Lexical Retrieval & Reciprocal Rank Fusion (RRF) Engine

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-05-M01 (Crab Pulsar Remnant / Messier 1)
Subsystem 05: Dual-Path Associative Memory Core
"""

import math
import re
from collections import Counter
from typing import Dict, List, Tuple, Union

import numpy as np


# ==============================================================================
# SECTION 1: BM25 LEXICAL RETRIEVAL ENGINE
# ==============================================================================

class BM25Retriever:
    """
    Okapi BM25 Lexical Search Engine.
    Mathematically scores document relevance based on term frequency (TF) and 
    inverse document frequency (IDF) with length normalization.
    
    Formula:
        Score(D, Q) = sum( IDF(q) * (f(q, D) * (k1 + 1)) / (f(q, D) + k1 * (1 - b + b * (|D| / avgdl))) )
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: Dict[int, int] = {}
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.corpus_size: int = 0
        self.term_freqs: Dict[int, Counter] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Basic alphanumeric tokenization for lexical analysis."""
        return re.findall(r'\b\w+\b', text.lower())

    def fit(self, corpus: Dict[int, str]):
        """Ingests corpus, computes average document length, term frequencies, and IDF."""
        self.corpus_size = len(corpus)
        total_length = 0

        for doc_id, text in corpus.items():
            tokens = self._tokenize(text)
            length = len(tokens)
            self.doc_lengths[doc_id] = length
            total_length += length
            
            tf = Counter(tokens)
            self.term_freqs[doc_id] = tf
            
            for term in tf.keys():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.avgdl = total_length / max(self.corpus_size, 1)

        # Compute IDF for all terms
        for term, freq in self.doc_freqs.items():
            # Robertson-Spärck Jones IDF formulation
            numerator = self.corpus_size - freq + 0.5
            denominator = freq + 0.5
            self.idf[term] = math.log(numerator / denominator + 1.0)

    def score_document(self, query_tokens: List[str], doc_id: int) -> float:
        """Calculates exact BM25 score for a single document against a query."""
        if self.corpus_size == 0:
            return 0.0

        score = 0.0
        doc_len = self.doc_lengths.get(doc_id, 0)
        tf_dict = self.term_freqs.get(doc_id, Counter())

        for token in query_tokens:
            if token not in self.term_freqs[doc_id]:
                continue
            
            tf = tf_dict[token]
            idf = self.idf.get(token, 0.0)
            
            # BM25 Length Normalization
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Executes full lexical scan, returning top-k scored documents."""
        query_tokens = self._tokenize(query)
        scores = []
        for doc_id in self.doc_lengths.keys():
            score = self.score_document(query_tokens, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        
        # Sort descending by score
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ==============================================================================
# SECTION 2: RECIPROCAL RANK FUSION (RRF) COMBINER
# ==============================================================================

def reciprocal_rank_fusion(
    dense_ranks: List[int],
    sparse_ranks: List[int],
    k: int = 60,
    top_n: int = 5
) -> List[Tuple[int, float]]:
    """
    Fuses rankings from dense (HNSW) and sparse (BM25) pipelines.
    
    Formula:
        RRF_Score(d) = sum_{pipeline in {dense, sparse}} 1 / (k + rank(d))
    """
    rrf_scores: Dict[int, float] = {}

    # Process Dense Ranks
    for rank, doc_id in enumerate(dense_ranks, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    # Process Sparse Ranks
    for rank, doc_id in enumerate(sparse_ranks, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))

    fused_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused_results[:top_n]


# ==============================================================================
# SECTION 3: EVALUATION METRICS (MRR & NDCG@10)
# ==============================================================================

def mean_reciprocal_rank(ranked_lists: List[List[int]], ground_truths: List[int]) -> float:
    """Computes MRR across a batch of queries."""
    mrr_sum = 0.0
    for ranked_list, truth in zip(ranked_lists, ground_truths):
        try:
            rank = ranked_list.index(truth) + 1
            mrr_sum += 1.0 / rank
        except ValueError:
            pass # Ground truth not found in ranked list
    return mrr_sum / max(len(ground_truths), 1)

def ndcg_at_k(ranked_list: List[int], ground_truth: int, k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K for a single query 
    with a binary relevance judgement.
    """
    dcg = 0.0
    for i, doc_id in enumerate(ranked_list[:k]):
        if doc_id == ground_truth:
            # Relevance is 1 if it's the ground truth, else 0
            dcg += 1.0 / math.log2(i + 2) # i is 0-indexed, standard formula uses rank + 1
            break
            
    idcg = 1.0 # Ideal DCG for a single perfectly ranked relevant document
    return dcg / idcg


# ==============================================================================
# MAIN VERIFICATION & FLIGHT TEST
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("USSE NOOSPHERE • AURA SUBSYSTEM 05: DENSE-SPARSE HYBRID RETRIEVAL")
    print("Waypoint: Crab Pulsar Remnant (Messier 1 • SEC-05-M01)")
    print("Protocol Officer Directive: BM25 Saturation & Reciprocal Rank Fusion")
    print("=" * 80)

    # 1. Corpus Simulation (Crab Pulsar Telemetry)
    corpus = {
        101: "Crab Pulsar PSR B0531+21 spins at 30.22 rotations per second with period P=33.08 milliseconds.",
        102: "Shield Harmonic Code MAG-SHIELD-77B phase-locks deflector shields at 30.22 Hz counter-phase against synchrotron radiation.",
        103: "Messier 1 is situated 6,500 light-years from Sol in the Perseus Arm of the constellation Taurus.",
        104: "Relativistic magnetic field strength B reaches 3.8e8 Tesla, inducing intense synchrotron X-ray filaments."
    }

    print("\n[TEST 1] Initializing and Fitting BM25 Engine...")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.fit(corpus)
    
    print(f"[✓] Corpus ingested. Size: {bm25.corpus_size}, Avg Doc Length: {bm25.avgdl:.2f} tokens")
    print(f"[✓] Computed Inverse Document Frequencies (IDF) for {len(bm25.idf)} unique terms.")

    query = "shield harmonic counter-phase synchrotron"
    print(f"\n[TEST 2] Executing Lexical Sparse Search (Query: '{query}')...")
    sparse_results = bm25.search(query, top_k=3)
    
    for rank, (doc_id, score) in enumerate(sparse_results, 1):
        print(f"  [{rank}] Document #{doc_id} | BM25 Score: {score:.4f}")
    assert sparse_results[0][0] == 102, "BM25 failed to retrieve correct top document."
    
    print("\n[TEST 3] Reciprocal Rank Fusion (k=60) Matrix Generation...")
    # Simulated dense pipeline results (HNSW vectors)
    mock_dense_ranks = [102, 104, 101]
    mock_sparse_ranks = [doc_id for doc_id, _ in sparse_results]
    
    print(f"  ├─ Dense Subsystem Ranks:  {mock_dense_ranks}")
    print(f"  ├─ Sparse Subsystem Ranks: {mock_sparse_ranks}")
    
    fused_ranks = reciprocal_rank_fusion(mock_dense_ranks, mock_sparse_ranks, k=60)
    for rank, (doc_id, rrf_score) in enumerate(fused_ranks, 1):
        print(f"  [{rank}] Document #{doc_id} | RRF Score: {rrf_score:.6f}")
        
    assert fused_ranks[0][0] == 102, "RRF Combiner failed to prioritize consensus document."
    print("[✓] RRF Matrix successfully smoothed and combined dense/sparse vector arrays.")

    print("\n[TEST 4] Empirical Retrieval Metrics Benchmark...")
    batch_fused_results = [
        [102, 104, 101], # Query 1
        [101, 103, 104], # Query 2
        [103, 102, 104]  # Query 3
    ]
    ground_truths = [102, 101, 104] # Note: Q3 misses top 2
    
    mrr = mean_reciprocal_rank(batch_fused_results, ground_truths)
    ndcg = np.mean([ndcg_at_k(ranks, gt, k=10) for ranks, gt in zip(batch_fused_results, ground_truths)])
    
    print(f"  ├─ Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"  ├─ NDCG@10 Average:            {ndcg:.4f}")
    print("\n" + "=" * 80)
    print("[SUCCESS] Protocol Officer Deliverable Validated: 02_BM25_and_RRF_Combiner.py")
    print("=" * 80)
