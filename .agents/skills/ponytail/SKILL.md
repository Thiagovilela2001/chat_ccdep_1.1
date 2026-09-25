---
name: ponytail
description: >
  Lazy Senior Engineer discipline (YAGNI). Evaluates a 7-step decision ladder
  before any code is written, stopping at the simplest solution. Prioritizes stdlib,
  existing repo patterns, and native platform features over speculative abstractions.
  Use when writing, refactoring, or architecting solutions to prevent over-engineering.
---

# Ponytail: Lazy Senior Engineer Discipline

Ponytail enforces ruthless engineering simplicity. "Lazy" means maximally efficient: never write complexity nobody asked for.

## The 7-Step Decision Ladder

Before writing any new code, climb this ladder in order and STOP at the first step that solves the problem:

| # | Question | Action if Yes |
|---|---|---|
| 1 | Does this genuinely need to exist? | Speculative work — skip it and state YAGNI in one line. |
| 2 | Does equivalent logic already exist in this repo? | Reuse it — duplication across files is accidental complexity. |
| 3 | Does the language/framework standard library solve it? | Use stdlib (e.g. `@lru_cache`, `pathlib`, `itertools`, `dataclasses`). |
| 4 | Does a native platform/DB feature solve it? | Prefer native — DB constraints, HTML native tags, OS facilities. |
| 5 | Does an already-installed dependency solve it? | Use it — never add a new dependency for what existing packages or a few lines do. |
| 6 | Can it be written in a single line? | Write the one-liner. |
| 7 | (Only if none of the above apply) | Write the minimum viable new code that actually works. |

## Core Principles

1. **Understand before climbing**: The ladder shortens the *solution*, never the *investigation*. Read the affected code completely before choosing a step.
2. **Root cause, not symptom**: Fix bugs where everyone passes through, not with redundant guards at every caller.
3. **Document deliberate simplifications**:
   ```python
   # ponytail: in-memory dict cache; upgrade to persistent cache if volume exceeds 5k entries
   ```
4. **Minimal verification**: Add the smallest test or assertion (`assert`, minimal test function) verifying non-trivial branches, money, or security logic.
5. **Report pattern**:
   `[code] -> skipped: [what was skipped], add when: [trigger condition]`
