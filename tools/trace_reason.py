"""把推理引擎的中间状态打印出来，排查断链用。"""
import math
import sys

from geoagent.schema import Problem
from geoagent import pipeline, reason, relations

P = Problem(
    problem_text="四边形ABCD为正方形，DE∥AC，AE=AC。AE与CD相交于F。求证：CE=CF。",
    given=["四边形ABCD为正方形", "DE∥AC", "AE=AC"],
    goal="求证：CE=CF",
    points=["A", "B", "C", "D", "E", "F"],
    segments=[("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("A", "F"),
              ("B", "C"), ("C", "D"), ("C", "E"), ("C", "F"), ("D", "E")],
    relations=["F在AE上，也在CD上"],
    pos={"A": [20, 20], "B": [20, 80], "C": [80, 80],
         "D": [80, 20], "E": [100, 50], "F": [70, 50]},
)

r = pipeline.resolve(P)
sol = r.solution
q = r.problem
X = dict(sol.coords)
s = math.dist(X["A"], X["B"])
X = {k: (v[0] / s, v[1] / s) for k, v in X.items()}
ctx = reason.Ctx(q, X, relations.derive(q).between)

kb = reason.KB()
reason._seed(q, ctx, kb, ("A", "B"))
for _ in range(16):
    n = 0
    for a in reason._rules(kb, ctx):
        if kb.add(*a[:2], a[2], a[3], a[4]):
            n += 1
    print(f"round {_}: +{n}  facts={len(kb.f)}")
    if not n:
        break

print("--- angval ---")
for f in kb.of("angval"):
    print(f"  ∠{reason._ang_name(f.args[0])} = {f.args[1]}  [{f.rule}]  pre={f.pre}")
print("--- len ---")
for f in kb.of("len"):
    print(f"  {f.args[0]} = {f.args[1]:.4f}  [{f.rule}]  pre={f.pre}")
print("--- eqseg ---")
for f in kb.of("eqseg"):
    print(f"  {f.args} [{f.rule}]")
pr = reason.prove(q, sol)
print("--- prove ---", pr.ok, pr.note)
print("goal:", reason._find_goal(kb, sol))
