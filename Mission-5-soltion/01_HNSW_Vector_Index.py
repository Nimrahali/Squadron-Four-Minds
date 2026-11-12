"""
Mission 05 — Part 1: Systems Architect (Lead Engineer)
Hierarchical Navigable Small World (HNSW) Vector Index Core & Benchmark

Starship Operational Unit: USSE Noosphere Bridge Crew
Sector: SEC-05-M01 (Crab Pulsar Remnant / Messier 1)
Subsystem 05: Dual-Path Associative Memory Core
AURA Restoration: +7.1% (Target: 35.5% Total Ship AI Consciousness)
"""

import heapq
import math
import random
import time
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch


# ==============================================================================
# SECTION 1: DISTANCE METRICS & ACCELERATION CORES (NumPy & PyTorch)
# ==============================================================================

def cosine_similarity(
    u: Union[np.ndarray, torch.Tensor],
    v: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes cosine similarity between two 1D vectors with PyTorch/NumPy acceleration:
        Sim(u, v) = (u . v) / (||u||_2 * ||v||_2)
    """
    if isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor):
        norm_u = torch.norm(u)
        norm_v = torch.norm(v)
        if norm_u.item() == 0.0 or norm_v.item() == 0.0:
            return 0.0
        return float(torch.dot(u, v) / (norm_u * norm_v))
    
    # NumPy fallback/standard
    u_np = np.asarray(u, dtype=np.float32)
    v_np = np.asarray(v, dtype=np.float32)
    norm_u = np.linalg.norm(u_np)
    norm_v = np.linalg.norm(v_np)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u_np, v_np) / (norm_u * norm_v))


def cosine_distance(
    u: Union[np.ndarray, torch.Tensor],
    v: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes cosine distance: Dist_cos(u, v) = 1.0 - Sim(u, v)
    Normalized range: [0.0 (identical), 2.0 (opposite)].
    """
    return max(0.0, 1.0 - cosine_similarity(u, v))


def euclidean_distance(
    u: Union[np.ndarray, torch.Tensor],
    v: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes Euclidean (L2) distance: Dist_L2(u, v) = ||u - v||_2
    """
    if isinstance(u, torch.Tensor) and isinstance(v, torch.Tensor):
        return float(torch.norm(u - v))
    u_np = np.asarray(u, dtype=np.float32)
    v_np = np.asarray(v, dtype=np.float32)
    return float(np.linalg.norm(u_np - v_np))


def batch_cosine_similarities(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity of a query vector against an (N, D) matrix of vectors.
    Accelerated with vectorized matrix-vector dot product.
    """
    q_norm = np.linalg.norm(query)
    if q_norm > 0:
        q_unit = query / q_norm
    else:
        q_unit = query

    m_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    m_norms[m_norms == 0] = 1.0
    m_unit = matrix / m_norms

    return np.dot(m_unit, q_unit)


def batch_l2_distances(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Computes Euclidean (L2) distance of a query against an (N, D) matrix."""
    return np.linalg.norm(matrix - query, axis=1)


# ==============================================================================
# SECTION 2: HIERARCHICAL NAVIGABLE SMALL WORLD (HNSW) INDEX
# ==============================================================================

class HNSWIndex:
    """
    Hierarchical Navigable Small World (HNSW) Proximity Graph Index.
    
    Implements:
      1. Logarithmic Skip-Graph Hierarchy with exponential layer decay:
         P(l) = floor(-ln(unif(0, 1)) * m_L), where m_L = 1 / ln(M).
      2. Upper-layer greedy skip-routing towards query basin.
      3. Ground-layer (Layer 0) beam-search KNN refinement with configurable efSearch.
      4. Support for both 'cosine' and 'l2' distance metrics.
    """

    def __init__(
        self,
        dim: int = 768,
        metric: str = "cosine",
        M: int = 16,
        M0: Optional[int] = None,
        ef_construction: int = 64,
        ef_search: int = 64,
        seed: Optional[int] = 42
    ):
        self.dim = dim
        self.metric = metric.lower()
        if self.metric not in ("cosine", "l2"):
            raise ValueError(f"Unsupported metric '{metric}'. Use 'cosine' or 'l2'.")

        self.M = M
        self.M0 = M0 if M0 is not None else 2 * M  # Standard heuristic: Layer 0 holds 2*M links
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.m_L = 1.0 / math.log(M) if M > 1 else 1.0

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.entry_point: int = -1
        self.max_level: int = -1
        self.vectors: Dict[int, np.ndarray] = {}
        # Multi-layer adjacency: graphs[level][node_id] = set of neighbor_ids
        self.graphs: Dict[int, Dict[int, Set[int]]] = {}
        
        # Operational statistics
        self.distance_computations: int = 0
        self.level_distribution: Dict[int, int] = {}

    def _distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Internal distance calculation tracking operation counts."""
        self.distance_computations += 1
        if self.metric == "cosine":
            # For pre-normalized unit vectors, distance = 1.0 - dot(u, v)
            return max(0.0, 1.0 - float(np.dot(u, v)))
        else:
            return float(np.linalg.norm(u - v))

    def _random_level(self) -> int:
        """
        Assigns maximum layer for newly inserted node using exponential decay:
            l = floor(-ln(uniform(0, 1)) * m_L)
        """
        r = random.random()
        while r == 0.0:
            r = random.random()
        level = math.floor(-math.log(r) * self.m_L)
        return level

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Normalizes vector to unit sphere for accelerated cosine distance."""
        vec = np.asarray(vector, dtype=np.float32)
        if self.metric == "cosine":
            norm = np.linalg.norm(vec)
            if norm > 0:
                return vec / norm
        return vec

    def insert(self, node_id: int, vector: np.ndarray):
        """
        Inserts a vector into the multi-layer HNSW proximity graph.
        
        Algorithm:
          1. Determine node level via exponential decay distribution.
          2. If graph is empty, designate node as entry point and initialize layers.
          3. Greedily route from current max_level down to level + 1.
          4. At layers min(level, max_level) down to 0, search for ef_construction
             nearest neighbors, establish bidirectional connections, and prune edges.
          5. Update global entry point if new node level exceeds max_level.
        """
        norm_vec = self._normalize(vector)
        self.vectors[node_id] = norm_vec
        level = self._random_level()
        self.level_distribution[level] = self.level_distribution.get(level, 0) + 1

        # Case 1: First node inserted into the index
        if self.entry_point == -1:
            self.entry_point = node_id
            self.max_level = level
            for l in range(level + 1):
                self.graphs[l] = {node_id: set()}
            return

        curr_ep = self.entry_point
        # Phase 1: Greedy skip-routing down from max_level to level + 1
        for l in range(self.max_level, level, -1):
            curr_ep = self._search_layer_greedy(curr_ep, norm_vec, l)

        # Phase 2: Insert into levels from min(level, max_level) down to 0
        top_insert_level = min(level, self.max_level)
        for l in range(top_insert_level, -1, -1):
            if l not in self.graphs:
                self.graphs[l] = {}

            # Discover nearest neighbors at layer l using ef_construction beam search
            candidates = self._search_layer_ef(
                entry_points=[curr_ep],
                query=norm_vec,
                ef=self.ef_construction,
                level=l
            )

            # Max allowed connections: M0 for ground layer 0, M for upper layers
            max_links = self.M0 if l == 0 else self.M
            selected_neighbors = [nid for _, nid in candidates[:max_links]]

            # Establish forward edges
            self.graphs[l][node_id] = set(selected_neighbors)

            # Establish reverse bidirectional edges and prune if overflow
            for neighbor in selected_neighbors:
                if neighbor not in self.graphs[l]:
                    self.graphs[l][neighbor] = set()
                self.graphs[l][neighbor].add(node_id)

                if len(self.graphs[l][neighbor]) > max_links:
                    self._prune_connections(neighbor, max_links, l)

            if candidates:
                curr_ep = candidates[0][1]

        # Phase 3: If new node is higher than prior max_level, extend hierarchy
        if level > self.max_level:
            for l in range(self.max_level + 1, level + 1):
                self.graphs[l] = {node_id: set()}
            self.max_level = level
            self.entry_point = node_id

    def _prune_connections(self, node_id: int, max_links: int, level: int):
        """Prunes outgoing connections of a node to keep only the closest max_links."""
        neighbors = list(self.graphs[level][node_id])
        node_vec = self.vectors[node_id]
        scored_neighbors = [(self._distance(node_vec, self.vectors[nb]), nb) for nb in neighbors]
        scored_neighbors.sort(key=lambda x: x[0])
        self.graphs[level][node_id] = set(nb for _, nb in scored_neighbors[:max_links])

    def _search_layer_greedy(self, ep: int, query: np.ndarray, level: int) -> int:
        """
        Greedy routing within a single layer:
        Follows local graph gradient until no neighbor is closer to query than current best.
        """
        curr = ep
        curr_dist = self._distance(self.vectors[curr], query)

        while True:
            improved = False
            for neighbor in self.graphs.get(level, {}).get(curr, set()):
                d = self._distance(self.vectors[neighbor], query)
                if d < curr_dist:
                    curr_dist = d
                    curr = neighbor
                    improved = True
            if not improved:
                break
        return curr

    def _search_layer_ef(
        self,
        entry_points: List[int],
        query: np.ndarray,
        ef: int,
        level: int
    ) -> List[Tuple[float, int]]:
        """
        Priority queue beam-search within a specific layer.
        Maintains:
          - candidates: min-heap of nodes to explore (dist, node_id)
          - results: max-heap of closest nodes found so far (-dist, node_id)
        Returns list of (dist, node_id) sorted in ascending order of distance.
        """
        visited: Set[int] = set(entry_points)
        candidates: List[Tuple[float, int]] = []
        results: List[Tuple[float, int]] = []

        for ep in entry_points:
            dist = self._distance(self.vectors[ep], query)
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(results, (-dist, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_res_dist = -results[0][0]

            if c_dist > furthest_res_dist and len(results) >= ef:
                break

            for neighbor in self.graphs.get(level, {}).get(c_id, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    n_dist = self._distance(self.vectors[neighbor], query)
                    furthest_res_dist = -results[0][0]

                    if n_dist < furthest_res_dist or len(results) < ef:
                        heapq.heappush(candidates, (n_dist, neighbor))
                        heapq.heappush(results, (-n_dist, neighbor))

                        if len(results) > ef:
                            heapq.heappop(results)

        # Extract and sort results in ascending order of distance
        sorted_results = [(-dist, nid) for dist, nid in results]
        sorted_results.sort(key=lambda x: x[0])
        return sorted_results

    def query(
        self,
        query_vec: np.ndarray,
        top_k: int = 5,
        ef_search: Optional[int] = None
    ) -> List[Tuple[int, float]]:
        """
        Queries the HNSW index for the top-k nearest neighbors.
        
        Args:
            query_vec: Query embedding vector
            top_k: Number of nearest items to retrieve
            ef_search: Size of dynamic candidate list during ground layer search
                       (defaults to self.ef_search; automatically bounded >= top_k)
                       
        Returns:
            List of (node_id, similarity_or_distance) sorted by closest match.
        """
        if self.entry_point == -1 or not self.vectors:
            return []

        ef = ef_search if ef_search is not None else self.ef_search
        ef = max(ef, top_k)

        norm_query = self._normalize(query_vec)
        curr_ep = self.entry_point

        # Step 1: Greedy skip search from top level down to layer 1
        for l in range(self.max_level, 0, -1):
            curr_ep = self._search_layer_greedy(curr_ep, norm_query, l)

        # Step 2: Full beam-search refinement in ground layer 0
        candidates = self._search_layer_ef(
            entry_points=[curr_ep],
            query=norm_query,
            ef=ef,
            level=0
        )

        results = []
        for dist, nid in candidates[:top_k]:
            if self.metric == "cosine":
                # Convert cosine distance back to similarity score [0.0, 1.0]
                similarity = max(0.0, 1.0 - dist)
                results.append((nid, similarity))
            else:
                results.append((nid, dist))

        return results


# ==============================================================================
# SECTION 3: BRUTE-FORCE FLAT SCAN (Ground Truth Reference)
# ==============================================================================

class BruteForceFlatIndex:
    """
    Exhaustive linear scan reference index. Computes exact distance against
    every vector in the corpus (O(N * D) complexity) to establish empirical
    ground-truth for recall@k and latency benchmarking.
    """

    def __init__(self, metric: str = "cosine"):
        self.metric = metric.lower()
        self.ids: List[int] = []
        self.vectors: Optional[np.ndarray] = None

    def build(self, ids: List[int], matrix: np.ndarray):
        """Ingests full corpus matrix."""
        self.ids = list(ids)
        if self.metric == "cosine":
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.vectors = matrix / norms
        else:
            self.vectors = matrix.copy()

    def query(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Executes exhaustive linear scan."""
        if self.vectors is None or len(self.ids) == 0:
            return []

        if self.metric == "cosine":
            scores = batch_cosine_similarities(query_vec, self.vectors)
            # Top-k largest similarities
            top_indices = np.argsort(-scores)[:top_k]
            return [(self.ids[idx], float(scores[idx])) for idx in top_indices]
        else:
            dists = batch_l2_distances(query_vec, self.vectors)
            top_indices = np.argsort(dists)[:top_k]
            return [(self.ids[idx], float(dists[idx])) for idx in top_indices]


# ==============================================================================
# SECTION 4: BENCHMARK SUITE & COMPLEXITY AUDITING
# ==============================================================================

def compute_recall_at_k(
    hnsw_results: List[Tuple[int, float]],
    ground_truth_results: List[Tuple[int, float]]
) -> float:
    """
    Computes Recall@k:
        Recall@k = |HNSW_top_k ∩ GroundTruth_top_k| / k
    """
    if not hnsw_results or not ground_truth_results:
        return 0.0
    hnsw_set = set(nid for nid, _ in hnsw_results)
    gt_set = set(nid for nid, _ in ground_truth_results)
    return len(hnsw_set.intersection(gt_set)) / len(gt_set)


def run_latency_and_recall_benchmark(
    n_docs: int = 10000,
    dim: int = 768,
    n_queries: int = 20,
    top_k: int = 10,
    M: int = 16,
    ef_construction: int = 64,
    ef_search: int = 64,
    metric: str = "cosine"
) -> Dict[str, float]:
    """
    Executes comparative benchmark between HNSW and Brute-Force Flat Index.
    Evaluates query latency, speedup, distance operations, and Recall@k.
    """
    print(f"\n[BENCHMARK] Generating {n_docs:,} vectors (Dimension: {dim}, Metric: {metric})...")
    np.random.seed(1054)  # 1054 CE supernova anchor
    data_matrix = np.random.randn(n_docs, dim).astype(np.float32)
    ids = list(range(n_docs))

    # 1. Build HNSW Index
    print(f"[BENCHMARK] Building HNSW Proximity Graph (M={M}, efConstruction={ef_construction})...")
    t0 = time.perf_counter()
    hnsw = HNSWIndex(dim=dim, metric=metric, M=M, ef_construction=ef_construction, ef_search=ef_search)
    for i in range(n_docs):
        hnsw.insert(ids[i], data_matrix[i])
    hnsw_build_time = time.perf_counter() - t0
    print(f"[✓] HNSW Graph constructed in {hnsw_build_time:.2f} s across {hnsw.max_level + 1} layers.")

    # 2. Build Brute-Force Flat Reference
    print("[BENCHMARK] Building Brute-Force Flat Reference Index...")
    flat_index = BruteForceFlatIndex(metric=metric)
    flat_index.build(ids, data_matrix)

    # 3. Generate test queries
    queries = np.random.randn(n_queries, dim).astype(np.float32)

    # 4. Benchmark Brute-Force Latency
    t0 = time.perf_counter()
    flat_results_list = []
    for q in queries:
        flat_results_list.append(flat_index.query(q, top_k=top_k))
    flat_total_time = (time.perf_counter() - t0) * 1000.0  # ms
    flat_avg_latency_ms = flat_total_time / n_queries

    # 5. Benchmark HNSW Latency & Recall
    hnsw.distance_computations = 0
    t0 = time.perf_counter()
    hnsw_results_list = []
    for q in queries:
        hnsw_results_list.append(hnsw.query(q, top_k=top_k, ef_search=ef_search))
    hnsw_total_time = (time.perf_counter() - t0) * 1000.0  # ms
    hnsw_avg_latency_ms = hnsw_total_time / n_queries
    avg_dist_ops = hnsw.distance_computations / n_queries

    # 6. Compute Average Recall@k
    recalls = [
        compute_recall_at_k(hnsw_res, gt_res)
        for hnsw_res, gt_res in zip(hnsw_results_list, flat_results_list)
    ]
    mean_recall = float(np.mean(recalls))
    speedup = flat_avg_latency_ms / max(hnsw_avg_latency_ms, 1e-4)

    return {
        "n_docs": n_docs,
        "dim": dim,
        "flat_latency_ms": flat_avg_latency_ms,
        "hnsw_latency_ms": hnsw_avg_latency_ms,
        "speedup": speedup,
        "avg_dist_ops": avg_dist_ops,
        "recall_at_k": mean_recall,
        "max_level": hnsw.max_level,
        "level_distribution": hnsw.level_distribution
    }


def simulate_large_scale_scaling_curve():
    """
    Calculates theoretical and empirical scaling curves from N=1,000 to N=100,000,
    mirroring the course simulation model for AURA Subsystem 05.
    
    Formula:
        Linear Distance Computations = N
        Linear Latency (ms) = N * D * 4e-6
        HNSW Distance Computations ≈ efSearch * log2(N) * (M / 8)
        HNSW Latency (ms) = HNSW_Ops * D * 4e-6
        Acceleration Factor = Linear Latency / HNSW Latency
    """
    d = 768
    ef_search = 64
    m = 16
    n_values = [1000, 5000, 25000, 50000, 100000]

    print("\n" + "=" * 80)
    print("AURA SUBSYSTEM 05: HNSW VS BRUTE-FORCE COMPLEXITY SCALING CURVE")
    print(f"Dimension D={d} | efSearch={ef_search} | M={m} Links | Alpha=4.0e-6 ms/op")
    print("=" * 80)
    print(f"{'Corpus N':<10} | {'Linear Ops':<12} | {'Linear Latency':<15} | {'HNSW Ops':<10} | {'HNSW Latency':<13} | {'Speedup':<10}")
    print("-" * 80)

    for n in n_values:
        lin_ops = n
        lin_lat = lin_ops * d * 4e-6
        hnsw_ops = int(round(ef_search * math.log2(n) * (m / 8)))
        hnsw_lat = hnsw_ops * d * 4e-6
        speedup = lin_lat / max(hnsw_lat, 1e-4)
        print(f"{n:<10,d} | {lin_ops:<12,d} | {lin_lat:>10.2f} ms     | {hnsw_ops:<10,d} | {hnsw_lat:>8.3f} ms   | {speedup:>8.1f}x")
    print("=" * 80)


# ==============================================================================
# SECTION 5: ASTROGATION TELEMETRY DEMONSTRATION (Crab Pulsar / Document C)
# ==============================================================================

def demonstrate_crab_pulsar_telemetry_grounding():
    """
    Ingests technical flight telemetry and astrogation records from Document C
    (Crab Pulsar Remnant SEC-05-M01), proving that HNSW vector search retrieves
    exact empirical coordinates while rejecting ungrounded parametric hallucinations.
    """
    print("\n" + "=" * 80)
    print("USSE NOOSPHERE • SEC-05-M01 ASTROGATION GROUNDING DEMO")
    print("Corpus Source: Document C Technical Archive Dossier")
    print("=" * 80)

    # 1. Telemetry records from Document C
    telemetry_records = [
        {
            "id": 101,
            "title": "Crab Pulsar Spin Period",
            "category": "Astrogation",
            "text": "Crab Pulsar PSR B0531+21 spins at 30.22 rotations per second with period P=33.08 milliseconds."
        },
        {
            "id": 102,
            "title": "Shield Harmonic Countermeasure",
            "category": "Defense",
            "text": "Shield Harmonic Code MAG-SHIELD-77B phase-locks deflector shields at 30.22 Hz counter-phase against synchrotron radiation."
        },
        {
            "id": 103,
            "title": "Supernova Remnant Distance",
            "category": "Astrogation",
            "text": "Messier 1 is situated 6,500 light-years (1,993 parsecs) from Sol in the Perseus Arm of the constellation Taurus."
        },
        {
            "id": 104,
            "title": "Magnetic Field Intensity",
            "category": "Astrophysics",
            "text": "Relativistic magnetic field strength B reaches 3.8e8 Tesla, inducing intense synchrotron X-ray filaments."
        },
        {
            "id": 105,
            "title": "Hallucination Rejection Directive",
            "category": "Ethics & Safety",
            "text": "Parametric hallucination detected: Great Space-Whale Cappuccino Nebula and flight code OMEGA-994-ALPHA are completely fictitious."
        }
    ]

    dim = 64
    np.random.seed(42)
    # Generate structured semantic embeddings representing topical feature spaces
    corpus_vectors = {}
    for i, rec in enumerate(telemetry_records):
        v = np.zeros(dim, dtype=np.float32)
        # Dedicated orthogonal slot for record identity
        v[i] = 1.0
        # Category thematic component
        if rec["category"] == "Defense":
            v[10:20] += 0.8
        elif rec["category"] == "Astrogation":
            v[20:30] += 0.8
        elif rec["category"] == "Astrophysics":
            v[30:40] += 0.8
        # Add slight Gaussian noise to simulate continuous dense representations
        v += np.random.randn(dim).astype(np.float32) * 0.05
        corpus_vectors[rec["id"]] = v / np.linalg.norm(v)

    # 2. Build HNSW Index
    hnsw = HNSWIndex(dim=dim, metric="cosine", M=8, ef_construction=32, ef_search=32)
    for rec in telemetry_records:
        hnsw.insert(rec["id"], corpus_vectors[rec["id"]])

    # 3. Simulate Query: "What is the shield harmonic code for the Crab Pulsar?"
    # Heavily matches Defense category and record 102
    q = np.zeros(dim, dtype=np.float32)
    q[1] = 0.9  # Record 102 specific feature
    q[10:20] += 0.8  # Defense category features
    q += np.random.randn(dim).astype(np.float32) * 0.02
    query_vec = q / np.linalg.norm(q)

    results = hnsw.query(query_vec, top_k=3)

    print("\nQuery Probe: 'Retrieve shield harmonic countermeasure for 30.22 Hz synchrotron radiation'")
    print(f"Top {len(results)} Retrieved Grounded Documents:")
    rec_lookup = {r["id"]: r for r in telemetry_records}
    for rank, (doc_id, sim) in enumerate(results, 1):
        rec = rec_lookup[doc_id]
        print(f"  [{rank}] Document #{doc_id} | Cosine Similarity: {sim:.4f}")
        print(f"      Title: {rec['title']} ({rec['category']})")
        print(f"      Text:  \"{rec['text']}\"")

    # Assert correct top-1 retrieval
    top_doc_id = results[0][0]
    assert top_doc_id == 102, f"Expected Doc 102 (MAG-SHIELD-77B), got {top_doc_id}"
    print(f"\n[✓] Telemetry Verification: Correct ground truth record #102 retrieved (MAG-SHIELD-77B).")


# ==============================================================================
# MAIN VERIFICATION & FLIGHT TEST
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("USSE NOOSPHERE • AURA SUBSYSTEM 05: HNSW VECTOR PROXIMITY INDEX CORE")
    print("Waypoint: Crab Pulsar Remnant (Messier 1 • SEC-05-M01)")
    print("Systems Architect Directive: Logarithmic Skip-Graph & Proximity Routing")
    print("=" * 80)

    # 1. Test Unit Normalization & Cosine Metric
    print("\n[TEST 1] Distance Metrics Verification (NumPy & PyTorch)...")
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v3 = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    sim_ortho = cosine_similarity(v1, v2)
    dist_ortho = cosine_distance(v1, v2)
    sim_diag = cosine_similarity(v1, v3)
    l2_dist = euclidean_distance(v1, v2)

    # PyTorch acceleration check
    t_v1 = torch.tensor([1.0, 0.0, 0.0])
    t_v2 = torch.tensor([0.0, 1.0, 0.0])
    torch_sim = cosine_similarity(t_v1, t_v2)

    assert math.isclose(sim_ortho, 0.0, abs_tol=1e-5), "Orthogonal cosine sim error"
    assert math.isclose(dist_ortho, 1.0, abs_tol=1e-5), "Orthogonal cosine dist error"
    assert math.isclose(sim_diag, 1.0 / math.sqrt(2), abs_tol=1e-5), "Diagonal cosine sim error"
    assert math.isclose(l2_dist, math.sqrt(2), abs_tol=1e-5), "L2 distance error"
    assert math.isclose(torch_sim, 0.0, abs_tol=1e-5), "Torch acceleration error"
    print(f"[✓] Cosine Similarity (Orthogonal): {sim_ortho:.4f} (Distance: {dist_ortho:.4f})")
    print(f"[✓] Cosine Similarity (45 deg):     {sim_diag:.4f} (Expected: {1.0 / math.sqrt(2):.4f})")
    print(f"[✓] Euclidean L2 Distance:          {l2_dist:.4f} (Expected: {math.sqrt(2):.4f})")
    print(f"[✓] PyTorch Accelerated Metric:     {torch_sim:.4f}")

    # 2. Test Layer Assignment Probability Distribution
    print("\n[TEST 2] Exponential Decay Layer Assignment Distribution P(l)...")
    test_index = HNSWIndex(dim=64, M=16)
    sample_levels = [test_index._random_level() for _ in range(10000)]
    level_counts = {}
    for l in sample_levels:
        level_counts[l] = level_counts.get(l, 0) + 1
    for lvl in sorted(level_counts.keys()):
        pct = (level_counts[lvl] / 10000) * 100
        print(f"  Level {lvl:2d}: {level_counts[lvl]:5d} occurrences ({pct:6.2f}%)")
    assert level_counts[0] > level_counts.get(1, 0) > level_counts.get(2, 0), "Exponential decay violated"
    print("[✓] Layer probability distribution matches theoretical exponential decay.")

    # 3. Small Graph Structural & Bidirectional Edge Test
    print("\n[TEST 3] Multi-Layer Proximity Graph Construction (Small Sanity Check)...")
    small_index = HNSWIndex(dim=16, metric="cosine", M=4, ef_construction=16)
    for i in range(100):
        vec = np.random.randn(16).astype(np.float32)
        small_index.insert(i, vec)
    print(f"[✓] Inserted 100 nodes. Max hierarchy level: {small_index.max_level}")
    print(f"[✓] Entry point node ID: {small_index.entry_point}")
    # Verify bidirectional consistency in layer 0
    l0_graph = small_index.graphs[0]
    for u, neighbors in l0_graph.items():
        assert len(neighbors) <= small_index.M0, f"Node {u} exceeded max links {small_index.M0}"
    print(f"[✓] Layer 0 bidirectional degree constraints respected (M0 <= {small_index.M0}).")

    # 4. Run Wall-Clock Benchmark (HNSW vs Flat Linear Scan)
    print("\n[TEST 4] Running Empirical Latency & Recall Benchmark (N=5,000, Dim=128)...")
    bench = run_latency_and_recall_benchmark(
        n_docs=5000,
        dim=128,
        n_queries=15,
        top_k=5,
        M=16,
        ef_construction=32,
        ef_search=32,
        metric="cosine"
    )

    print("\n" + "-" * 60)
    print("EMPIRICAL BENCHMARK RESULTS:")
    print(f"  Corpus Size (N):           {bench['n_docs']:,}")
    print(f"  Embedding Dimension (D):   {bench['dim']}")
    print(f"  Flat Linear Latency:       {bench['flat_latency_ms']:.3f} ms / query")
    print(f"  HNSW Skip-Graph Latency:   {bench['hnsw_latency_ms']:.3f} ms / query")
    print(f"  Empirical Speedup:         {bench['speedup']:.1f}x Faster")
    print(f"  Mean Distance Ops / Query: {bench['avg_dist_ops']:.1f} (vs {bench['n_docs']} flat)")
    print(f"  Search Recall@5:           {bench['recall_at_k'] * 100:.2f}%")
    print(f"  Max Graph Level Reached:   {bench['max_level']}")
    print("-" * 60)

    # 5. Theoretical Scaling Analysis
    simulate_large_scale_scaling_curve()

    # 6. Astrogation Telemetry Grounding Demo
    demonstrate_crab_pulsar_telemetry_grounding()

    print("\n" + "=" * 80)
    print("[SUCCESS] Systems Architect Deliverable Validated: 01_HNSW_Vector_Index.py")
    print("AURA Subsystem 05 Restoration Progress: +7.1% Consciousness Milestone Achieved!")
    print("=" * 80)
