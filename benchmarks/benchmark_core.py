"""Microbenchmark reproduzível dos validadores locais (sem rede e sem LLM)."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_core.citation_validator import validate_citations
from rag_core.numerical_validator import validate_numbers


def _measure(callable_, iterations: int) -> dict:
    started = time.perf_counter()
    for _ in range(iterations):
        callable_()
    elapsed = time.perf_counter() - started
    return {
        "iterations": iterations,
        "elapsed_ms": round(elapsed * 1000, 3),
        "operations_per_second": round(iterations / elapsed, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations deve ser maior ou igual a 1")

    node = SimpleNamespace(
        metadata={"source_file": "regional/boletim.pdf", "page": 3},
        get_content=lambda: "As taxas observadas foram 3,4% e 2,8%.",
    )
    answer = (
        "A diferença foi 3,4% − 2,8% = 0,6 p.p. "
        "(Fonte: boletim.pdf, p. 3)."
    )
    output = {
        "python": platform.python_version(),
        "numeric_validation": _measure(
            lambda: validate_numbers(answer, [node]), args.iterations
        ),
        "citation_validation": _measure(
            lambda: validate_citations(answer, [node]), args.iterations
        ),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
