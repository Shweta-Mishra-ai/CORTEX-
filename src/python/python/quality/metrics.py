"""
CORTEX v2 — Output Quality Preservation
==========================================
Measures and preserves LLM output quality under context compression.

  Reducing context tokens risks degrading LLM output.
  This module quantifies the tradeoff and finds the optimal
  compression point where quality is preserved.

Classes:
  QualityMetrics        — measures file accuracy, hallucination rate
  CompressionCurve      — plots quality vs compression tradeoff
  OptimalBudgetFinder   — finds minimum tokens for target accuracy
  HallucinationDetector — detects when LLM invents non-existent files

This module is used by the experiment scripts to generate
the quality-vs-compression figures for the paper.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Quality metrics ───────────────────────────────────────────────────────────

@dataclass
class QualityScore:
    top1_accuracy:      float   # Did LLM name the correct file first?
    top3_accuracy:      float   # Correct file in top 3?
    any_accuracy:       float   # Correct file anywhere in response?
    hallucination_rate: float   # Fraction of mentioned files that don't exist
    response_length:    int     # Output tokens (proxy for output richness)
    context_tokens:     int     # Input tokens used

    @property
    def efficiency_score(self) -> float:
        """Accuracy per 1K input tokens — higher is better."""
        if self.context_tokens == 0:
            return 0.0
        return (self.top1_accuracy * 100) / (self.context_tokens / 1000)

    def __str__(self) -> str:
        return (f"Top-1: {self.top1_accuracy*100:.1f}% | "
                f"Top-3: {self.top3_accuracy*100:.1f}% | "
                f"Hallucination: {self.hallucination_rate*100:.1f}% | "
                f"Tokens: {self.context_tokens:,} | "
                f"Efficiency: {self.efficiency_score:.2f} acc/1Ktok")


class QualityMetrics:
    """
    Evaluates LLM response quality against ground truth.

    Usage:
        metrics = QualityMetrics(repo_path='/path/to/repo')
        score = metrics.score(
            response="The bug is in src/auth.py in the login() function",
            ground_truth_files=["src/auth.py"],
            context_tokens=8500
        )
        print(score)
    """

    def __init__(self, repo_path: str = ""):
        self._repo = repo_path
        # Cache of real files for hallucination detection
        self._real_files: Optional[set[str]] = None

    def _get_real_files(self) -> set[str]:
        if self._real_files is not None:
            return self._real_files
        if not self._repo or not Path(self._repo).exists():
            return set()
        real = set()
        for fp in Path(self._repo).rglob("*"):
            if fp.is_file():
                real.add(fp.name.lower())
                real.add(str(fp.relative_to(self._repo)).lower())
        self._real_files = real
        return real

    def _extract_file_mentions(self, response: str) -> list[str]:
        """Extract file path mentions from LLM response."""
        patterns = [
            # Quoted paths
            re.compile(r'[\'"`]([^\'"`\n]+\.(?:py|js|ts|go|rb|java|rs|cpp|c|h|kt|swift))[\'"`]'),
            # Backtick code spans
            re.compile(r'`([^`\n]+\.(?:py|js|ts|go|rb|java|rs|cpp|c|h))`'),
            # Plain file paths
            re.compile(r'\b([\w/\-\.]+\.(?:py|js|ts|go|rb|java|rs|cpp|c|h))\b'),
        ]
        mentions = []
        for pat in patterns:
            mentions.extend(pat.findall(response))
        # Deduplicate, preserve order
        seen = set()
        result = []
        for m in mentions:
            m = m.strip()
            if m not in seen and len(m) > 3:
                seen.add(m)
                result.append(m)
        return result

    def _matches(self, mention: str, ground_truth: str) -> bool:
        m  = mention.lower().strip("/").strip()
        gt = ground_truth.lower().strip("/").strip()
        return (gt.endswith(m) or m.endswith(gt) or
                m in gt or gt in m or
                Path(m).stem == Path(gt).stem)

    def score(self, response: str, ground_truth_files: list[str],
              context_tokens: int = 0) -> QualityScore:
        mentions = self._extract_file_mentions(response)
        real     = self._get_real_files()

        # Accuracy
        top1 = any(self._matches(mentions[0], gt)
                   for gt in ground_truth_files) if mentions else False
        top3 = any(self._matches(m, gt)
                   for m in mentions[:3] for gt in ground_truth_files)
        any_ = any(self._matches(m, gt)
                   for m in mentions for gt in ground_truth_files)

        # Hallucination: mentioned files that don't exist in the repo
        if real and mentions:
            hallucinated = sum(
                1 for m in mentions
                if not any(m.lower().endswith(r) or r.endswith(m.lower())
                           for r in real)
            )
            hall_rate = hallucinated / len(mentions)
        else:
            hall_rate = 0.0

        return QualityScore(
            top1_accuracy      = float(top1),
            top3_accuracy      = float(top3),
            any_accuracy       = float(any_),
            hallucination_rate = round(hall_rate, 3),
            response_length    = len(response.split()),
            context_tokens     = context_tokens,
        )

    def score_batch(self, responses: list[dict]) -> dict:
        """
        Score a batch of responses.

        Args:
            responses: List of dicts with keys:
                       response, ground_truth_files, context_tokens

        Returns:
            Aggregate dict with mean scores.
        """
        scores = [
            self.score(r["response"], r["ground_truth_files"],
                       r.get("context_tokens", 0))
            for r in responses
        ]
        n = len(scores)
        if n == 0:
            return {}
        return {
            "n": n,
            "mean_top1":      round(sum(s.top1_accuracy for s in scores) / n, 3),
            "mean_top3":      round(sum(s.top3_accuracy for s in scores) / n, 3),
            "mean_any":       round(sum(s.any_accuracy  for s in scores) / n, 3),
            "mean_halluc":    round(sum(s.hallucination_rate for s in scores) / n, 3),
            "mean_tokens":    round(sum(s.context_tokens for s in scores) / n),
            "mean_efficiency":round(sum(s.efficiency_score for s in scores) / n, 3),
        }


# ── Compression curve analysis ────────────────────────────────────────────────

@dataclass
class CurvePoint:
    budget_tokens: int
    top1_accuracy: float
    hallucination_rate: float
    compression_ratio: float
    method: str


class CompressionCurve:
    """
    Computes the quality-vs-compression tradeoff curve.
    Used to generate Figure B in the paper.

    Finds: at what token budget does quality start to degrade?
    This is the "knee point" — the optimal budget for a given task.

    Usage:
        curve = CompressionCurve()
        points = curve.compute(
            results_by_budget={
                4096:   [{"response": "...", "ground_truth_files": [...]}],
                8192:   [...],
                16384:  [...],
                32768:  [...],
            }
        )
        knee = curve.find_knee(points)
        print(f"Optimal budget: {knee.budget_tokens:,} tokens")
    """

    def compute(self, results_by_budget: dict[int, list[dict]],
                repo_path: str = "") -> list[CurvePoint]:
        metrics = QualityMetrics(repo_path)
        points  = []
        for budget in sorted(results_by_budget):
            agg = metrics.score_batch(results_by_budget[budget])
            if not agg:
                continue
            points.append(CurvePoint(
                budget_tokens      = budget,
                top1_accuracy      = agg["mean_top1"],
                hallucination_rate = agg["mean_halluc"],
                compression_ratio  = 1.0,   # filled by caller
                method             = "unknown",
            ))
        return points

    def find_knee(self, points: list[CurvePoint]) -> Optional[CurvePoint]:
        """
        Find the knee point: smallest budget where accuracy is within
        5 percentage points of the maximum.
        """
        if not points:
            return None
        max_acc = max(p.top1_accuracy for p in points)
        threshold = max_acc - 0.05
        for point in sorted(points, key=lambda p: p.budget_tokens):
            if point.top1_accuracy >= threshold:
                return point
        return points[-1]


# ── Hallucination detector ────────────────────────────────────────────────────

class HallucinationDetector:
    """
    Real-time hallucination detector for LLM file mentions.

    Checks if mentioned files exist in the repository.
    Reports: hallucination rate, invented paths, near-misses.

    Usage:
        detector = HallucinationDetector('/path/to/repo')
        report = detector.analyze(
            "The bug is in src/atuh.py and utils/helper.js"
        )
        print(report['hallucination_rate'])
        print(report['suggestions'])   # did you mean src/auth.py?
    """

    def __init__(self, repo_path: str):
        self._repo  = repo_path
        self._index = self._build_index()

    def _build_index(self) -> dict[str, str]:
        """Build {stem_lower: full_path} index."""
        index = {}
        if not Path(self._repo).exists():
            return index
        for fp in Path(self._repo).rglob("*"):
            if fp.is_file():
                key = fp.stem.lower()
                index[key] = str(fp)
                # Also index by relative path
                try:
                    rel = str(fp.relative_to(self._repo)).lower()
                    index[rel] = str(fp)
                except ValueError:
                    pass
        return index

    def _levenshtein(self, a: str, b: str) -> int:
        if len(a) < len(b):
            return self._levenshtein(b, a)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a):
            curr = [i + 1]
            for j, cb in enumerate(b):
                curr.append(min(prev[j+1]+1, curr[-1]+1,
                                prev[j] + (ca != cb)))
            prev = curr
        return prev[-1]

    def _find_closest(self, mention: str, top_k: int = 3) -> list[str]:
        """Find closest real files to a hallucinated mention."""
        stem = Path(mention).stem.lower()
        scored = []
        for key in self._index:
            dist = self._levenshtein(stem, key)
            scored.append((dist, self._index[key]))
        scored.sort()
        return [path for _, path in scored[:top_k]]

    def analyze(self, response: str) -> dict:
        pattern  = re.compile(
            r'\b([\w/\-\.]+\.(?:py|js|ts|go|rb|java|rs|cpp|c|h|kt))\b'
        )
        mentions = list(dict.fromkeys(pattern.findall(response)))

        real_mentions = []
        hallucinated  = []
        suggestions   = {}

        for m in mentions:
            m_lower = m.lower().strip("/")
            stem    = Path(m).stem.lower()
            if (m_lower in self._index or stem in self._index or
                    any(m_lower.endswith(k) or k.endswith(m_lower)
                        for k in self._index)):
                real_mentions.append(m)
            else:
                hallucinated.append(m)
                suggestions[m] = self._find_closest(m)

        total = len(mentions)
        return {
            "total_mentions":    total,
            "real_files":        real_mentions,
            "hallucinated":      hallucinated,
            "hallucination_rate": len(hallucinated) / total if total else 0.0,
            "suggestions":       suggestions,
        }


# ── Optimal budget finder ─────────────────────────────────────────────────────

class OptimalBudgetFinder:
    """
    Finds the minimum token budget that achieves a target accuracy.
    Used to answer: "How small can the context be without hurting output?"

    Algorithm: binary search over token budgets.

    Usage:
        finder = OptimalBudgetFinder(
            repo_path='/path/to/repo',
            target_top1_accuracy=0.70   # 70% top-1 accuracy target
        )
        optimal = finder.estimate(known_results)
        print(f"Minimum budget for 70% accuracy: {optimal:,} tokens")
    """

    def __init__(self, repo_path: str = "", target_top1_accuracy: float = 0.70):
        self._repo   = repo_path
        self._target = target_top1_accuracy

    def estimate(self, results_by_budget: dict[int, list[dict]]) -> Optional[int]:
        """
        Given results at different budgets, estimate minimum budget
        for target accuracy using interpolation.
        """
        metrics = QualityMetrics(self._repo)
        scored  = []
        for budget in sorted(results_by_budget):
            agg = metrics.score_batch(results_by_budget[budget])
            scored.append((budget, agg.get("mean_top1", 0)))

        # Find first budget where accuracy >= target
        for budget, acc in scored:
            if acc >= self._target:
                return budget

        # If never reached target, return largest tested
        return scored[-1][0] if scored else None

    def extrapolate(self, results_by_budget: dict[int, list[dict]]) -> dict:
        """
        Fit a log-linear model to extrapolate beyond tested budgets.
        Returns model parameters and predicted optimal budget.
        """
        metrics = QualityMetrics(self._repo)
        points  = []
        for budget in sorted(results_by_budget):
            agg = metrics.score_batch(results_by_budget[budget])
            acc = agg.get("mean_top1", 0)
            if budget > 0 and acc > 0:
                points.append((math.log(budget), acc))

        if len(points) < 2:
            return {"error": "Need at least 2 data points"}

        # Linear regression: acc = a * log(budget) + b
        n    = len(points)
        sx   = sum(p[0] for p in points)
        sy   = sum(p[1] for p in points)
        sxy  = sum(p[0]*p[1] for p in points)
        sxx  = sum(p[0]**2 for p in points)
        denom = n * sxx - sx ** 2
        if denom == 0:
            return {"error": "Degenerate fit"}
        a = (n * sxy - sx * sy) / denom
        b = (sy - a * sx) / n

        # Predict budget for target accuracy
        if a <= 0:
            return {"error": "Negative slope — accuracy decreasing with budget"}
        log_pred = (self._target - b) / a
        predicted_budget = int(math.exp(log_pred))

        return {
            "slope_a":          round(a, 6),
            "intercept_b":      round(b, 6),
            "predicted_budget": predicted_budget,
            "target_accuracy":  self._target,
            "r_squared":        self._r_squared(points, a, b),
        }

    def _r_squared(self, points, a, b) -> float:
        y_mean = sum(p[1] for p in points) / len(points)
        ss_tot = sum((p[1] - y_mean)**2 for p in points)
        ss_res = sum((p[1] - (a * p[0] + b))**2 for p in points)
        return round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0


__all__ = [
    "QualityMetrics", "QualityScore",
    "CompressionCurve", "CurvePoint",
    "HallucinationDetector",
    "OptimalBudgetFinder",
]