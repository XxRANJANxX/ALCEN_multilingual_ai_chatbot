"""Evaluation harness.

Dataset (JSONL), one object per line:
 {"question": "...", "lang": "hi", "answerable": true,
  "expected_sources": ["okf:rag"], "keywords": ["evidence"]}

Retrieval metrics need no LLM. Pass generate=True to also score keyword accuracy
(and, when the verifier is on, the share of answers with unsupported claims).
"""

import json
import statistics
import time
from collections import defaultdict


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    return round(vals[int(round(p * (len(vals) - 1)))], 1)


def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def run_eval(pipeline, dataset_path: str, generate: bool = False) -> dict:
    rows = load_dataset(dataset_path)
    results, latencies = [], []
    top_ans, top_unans = [], []
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        prep = await pipeline.prepare(row["question"], session_id=f"eval-{i}", language=row.get("lang"))
        answer, verification = None, None
        if generate:
            answer = await pipeline.generate(prep)
            verification = await pipeline.verify(prep, answer)
        latencies.append((time.perf_counter() - t0) * 1000)

        sources = [p.source for p in prep.passages]
        answerable = row.get("answerable", True)
        item = {"question": row["question"], "lang": row.get("lang", "?"), "answerable": answerable,
                "route": prep.route, "sources": sources}
        top1 = max([p.score for p in prep.raw["vector"]] + [0.0])
        top_okf = max([p.score for p in prep.raw["okf"]] + [0.0])
        (top_ans if answerable else top_unans).append((top1, top_okf))
        if answerable:
            exp = set(row.get("expected_sources", []))
            item["hit"] = bool(exp & set(sources)) if exp else bool(sources)
            if exp:  # reciprocal rank
                rank = next((n for n, s in enumerate(sources, 1) if s in exp), None)
                item["rr"] = 1 / rank if rank else 0.0
            if generate and row.get("keywords"):
                item["keyword_hit"] = all(k.lower() in (answer or "").lower() for k in row["keywords"])
        else:
            item["correct_refusal"] = not sources
        if verification is not None:
            item["supported"] = verification["supported"]
        results.append(item)

    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]
    by_lang: dict[str, list[bool]] = defaultdict(list)
    for r in ans:
        by_lang[r["lang"]].append(r["hit"])

    def rate(items, key):
        vals = [r[key] for r in items if key in r]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "n": len(results),
        "retrieval_recall": rate(ans, "hit"),
        "retrieval_mrr": round(statistics.mean([r["rr"] for r in ans if "rr" in r]), 3)
        if any("rr" in r for r in ans) else None,
        "recall_by_language": {k: round(sum(v) / len(v), 3) for k, v in by_lang.items()},
        "correct_refusal_rate": rate(unans, "correct_refusal"),
        "keyword_accuracy": rate(ans, "keyword_hit"),
        "verified_supported_rate": rate(results, "supported"),
        "latency_ms": {"p50": _pct(latencies, 0.5), "p95": _pct(latencies, 0.95)},
        "calibration": {
            "answerable_min_top_vector_score": round(min((v for v, _ in top_ans), default=0), 3),
            "unanswerable_max_top_vector_score": round(max((v for v, _ in top_unans), default=0), 3),
            "answerable_min_top_okf_score": round(min((o for _, o in top_ans), default=0), 3),
            "unanswerable_max_top_okf_score": round(max((o for _, o in top_unans), default=0), 3),
            "hint": "Set MIN_SCORE / OKF_MIN_SCORE between the unanswerable max and answerable min.",
        },
    }
    return {"summary": summary, "results": results}
