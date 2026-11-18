"""
Quizzes for every qslab tutorial.

Each topic has conceptual multiple-choice questions and *computational*
questions that you answer by running qslab yourself.  The numeric answers
were produced by the package's own benchmarked solvers.

Usage
-----
In a notebook::

    from qslab import quiz
    quiz.topics()                 # list topics
    quiz.show('kitaev')           # print questions
    quiz.check('kitaev', 3, 'b')  # check one answer
    quiz.check('kitaev', 5, -0.787)
    quiz.run('kitaev')            # interactive, with score
    quiz.solutions('kitaev')      # answers + explanations

From the shell::

    qslab-quiz                    # menu
    qslab-quiz kitaev             # run one topic
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

_DIR = os.path.join(os.path.dirname(__file__), "quizzes")

__all__ = ["topics", "load", "show", "check", "run", "solutions", "main"]


def topics() -> List[str]:
    return sorted(f[:-5] for f in os.listdir(_DIR) if f.endswith(".json"))


def load(topic: str) -> Dict[str, Any]:
    path = os.path.join(_DIR, f"{topic}.json")
    if not os.path.exists(path):
        raise KeyError(f"unknown quiz topic {topic!r}; available: {topics()}")
    with open(path) as f:
        return json.load(f)


def _fmt_q(n: int, q: dict) -> str:
    s = [f"Q{n}. [{q['type']}] {q['question']}"]
    if q["type"] == "mcq":
        for k, opt in zip("abcdefgh", q["options"]):
            s.append(f"    ({k}) {opt}")
    else:
        s.append(f"    -> numeric answer (tolerance {q.get('tol', 1e-3)})")
        if q.get("hint"):
            s.append(f"    hint: {q['hint']}")
    return "\n".join(s)


def show(topic: str):
    data = load(topic)
    print(f"=== {data['title']} ===\n{data.get('intro', '')}\n")
    for n, q in enumerate(data["questions"], 1):
        print(_fmt_q(n, q) + "\n")


def _correct(q: dict, ans) -> bool:
    if q["type"] == "mcq":
        a = str(ans).strip().lower()
        if a.isdigit():
            a = "abcdefgh"[int(a)]
        return a == q["answer"]
    try:
        return abs(float(ans) - float(q["answer"])) <= float(q.get("tol", 1e-3))
    except (TypeError, ValueError):
        return False


def check(topic: str, number: int, answer) -> bool:
    """Check answer to question ``number`` (1-based); prints the explanation."""
    q = load(topic)["questions"][number - 1]
    ok = _correct(q, answer)
    print(("Correct!" if ok else f"Not quite. Expected: {q['answer']}") +
          f"\n  {q.get('explanation', '')}")
    return ok


def solutions(topic: str):
    data = load(topic)
    for n, q in enumerate(data["questions"], 1):
        print(f"Q{n}: {q['answer']}  -  {q.get('explanation', '')}")
        if q.get("code"):
            print("    code: " + q["code"].replace("\n", "\n          "))


def run(topic: str, input_fn=input) -> float:
    """Interactive quiz; returns the score fraction."""
    data = load(topic)
    print(f"=== {data['title']} ===\n{data.get('intro', '')}\n"
          "(type 'skip' to skip, 'quit' to stop)\n")
    score, asked = 0, 0
    for n, q in enumerate(data["questions"], 1):
        print(_fmt_q(n, q))
        a = input_fn("your answer: ").strip()
        if a == "quit":
            break
        asked += 1
        if a == "skip":
            print(f"  answer: {q['answer']} - {q.get('explanation', '')}\n")
            continue
        if _correct(q, a):
            score += 1
            print("  correct!  " + q.get("explanation", "") + "\n")
        else:
            print(f"  expected {q['answer']}.  " + q.get("explanation", "") + "\n")
    frac = score / max(asked, 1)
    print(f"Score: {score}/{asked} ({100 * frac:.0f}%)")
    return frac


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        run(argv[0])
        return
    t = topics()
    for n, name in enumerate(t, 1):
        print(f"{n:2d}. {load(name)['title']}  [{name}]")
    c = input("choose a topic number: ").strip()
    run(t[int(c) - 1])


if __name__ == "__main__":
    main()
