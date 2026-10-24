"""CLI:  python -m app.eval.run [--generate] [--dataset path]"""
import argparse
import asyncio
import json

from app.core.container import get_container
from app.eval.runner import run_eval


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--generate", action="store_true", help="also call the LLM and score answers")
    args = ap.parse_args()
    c = get_container()
    report = await run_eval(c.pipeline, args.dataset or c.settings.eval_dataset, args.generate)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    for r in report["results"]:
        flag = r.get("hit", r.get("correct_refusal"))
        print(("PASS" if flag else "FAIL"), r["lang"], r["question"][:70], "->", r["sources"][:3])


if __name__ == "__main__":
    asyncio.run(main())
