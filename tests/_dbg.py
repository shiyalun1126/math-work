import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mathagent.latex import normalize, _CASES, _split_top

s = r"\begin{cases} x+y=3 \\ x-y=1 \end{cases}"
n = normalize(s)
print("in :", s.encode("unicode_escape"))
print("out:", n.encode("unicode_escape"))
m = _CASES.search(n)
print("matched:", bool(m))
if m:
    body = m.group(1)
    print("body:", body.encode("unicode_escape"))
    print("split:", _split_top(body, ("\\\\",)))
