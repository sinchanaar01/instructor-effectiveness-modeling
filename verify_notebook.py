"""Self-check for instructor_effectiveness.ipynb. Run: python verify_notebook.py
Exit code 0 = all checks pass. Any FAIL must be fixed, then re-run nbconvert and this script."""
import json, re, subprocess, sys
NB = "instructor_effectiveness.ipynb"
nb = json.load(open(NB, encoding="utf-8"))
cells = nb["cells"]
src = lambda c: "".join(c["source"])
fails = []
def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok: fails.append(msg)

# 1. execution: no errors, no stderr, sequential execution counts, code cells have outputs where they print
code = [c for c in cells if c["cell_type"] == "code"]
errs = sum(1 for c in code for o in c.get("outputs", []) if o.get("output_type") == "error")
stderr = sum(1 for c in code for o in c.get("outputs", []) if o.get("output_type") == "stream" and o.get("name") == "stderr")
counts = [c.get("execution_count") for c in code]
check(errs == 0, "no error outputs")
check(stderr == 0, "no stderr outputs")
check(counts == list(range(1, len(code) + 1)), "execution counts are 1..N in order (fresh kernel run)")
check(nb["metadata"].get("kernelspec"), "kernelspec present in metadata")
check("kernelspec" not in "".join(src(c) for c in cells if c["cell_type"] == "markdown"), "no kernelspec JSON text inside markdown")

# 2. mandatory questions: Q1..Q5 exactly once, in order
qs = [re.match(r"###\s*Q(\d)\.", src(c)) for c in cells if c["cell_type"] == "markdown"]
qs = [m.group(1) for m in qs if m]
check(qs == ["1", "2", "3", "4", "5"], f"Q1-Q5 exactly once in order (found {qs})")

# 3. structure
mds = [src(c) for c in cells if c["cell_type"] == "markdown"]
check(sum(1 for m in mds if "Early Observations" in m.split("\n")[0]) == 1, "exactly one 'Early Observations' cell")
stacked = [i for i in range(len(cells) - 1) if cells[i]["cell_type"] == "markdown" and cells[i + 1]["cell_type"] == "markdown"
           and src(cells[i]).startswith("## ") and src(cells[i + 1]).startswith("## ")]
check(not stacked, f"no stacked '## ' section headers (at {stacked})")
nums = [int(m.group(1)) for m in (re.match(r"##\s*(\d+)\.", s) for s in mds) if m]
check(nums == list(range(1, len(nums) + 1)), f"section numbers have no gaps/duplicates ({nums})")
# every '## N.' section that contains a code cell must have a markdown cell after its LAST code cell before the next '## '
bad = []
sec_start = None
for i, c in enumerate(cells + [{"cell_type": "markdown", "source": ["## END"]}]):
    if c["cell_type"] == "markdown" and src(c).startswith("## "):
        if sec_start is not None:
            block = cells[sec_start:i]
            idx = [j for j, b in enumerate(block) if b["cell_type"] == "code"]
            if idx and idx[-1] == len(block) - 1:
                bad.append(src(cells[sec_start]).split("\n")[0])
        sec_start = i
check(not bad, f"each section with code ends with a markdown interpretation ({bad})")

# 4. content checks from printed outputs
out_all = "\n".join("".join(o.get("text", [])) for c in code for o in c.get("outputs", []) if o.get("output_type") == "stream")
card = re.findall(r"^\s*I_\d+\s+(Low|Medium|High)\s", out_all, re.M)
check({"Low", "Medium", "High"} <= set(card), f"coaching card covers Low, Medium and High tiers (found {sorted(set(card))})")
check(len(re.findall(r"^\s*I_\d+\s+\w+\s+.*\s+yes\s", out_all, re.M)) >= 2, "coaching card has >= 2 borderline='yes' rows")
summ = next((m for m in mds if m.startswith("### D. Plain-language")), "")
jargon = [w for w in ["macro-f1", "f1", "bootstrap", "stratified", "dirichlet", "baseline", "standard deviation", "ablation", "permutation", "tier flips", "shrinkage", "logistic", "importance", "leakage", "icc"] if w in summ.lower()]
check(summ and not jargon, f"plain-language summary has zero jargon (found {jargon})")
sents = re.split(r"(?<=[.!?])\s+", summ.split("\n", 2)[-1].strip())
check(5 <= len([s for s in sents if s]) <= 7, f"plain-language summary is 5-6 sentences (found {len([s for s in sents if s])})")
check(re.search(r"\d", summ) is not None, "plain-language summary contains real numbers")

# 5. every decimal/percent number quoted in backticks in markdown must appear in printed outputs
outn = out_all.replace(",", "")
missing = []
for m in mds:
    for tok in re.findall(r"`(-?\d+\.\d+%?|\d+\.\d+/\d+\.\d+)`", m):
        for part in tok.split("/"):
            p = part.rstrip("%")
            cand = {p, p.lstrip("-"), p.rstrip("0")}
            if not any(c and c in outn for c in cand):
                # allow 2-decimal rounding of a printed number (e.g. 0.46 from 0.4591)
                found = False
                for x in re.findall(r"-?\d+\.\d+", outn):
                    if abs(float(x) - float(p)) <= 0.5 * 10 ** (-len(p.split(".")[1])) + 1e-9:
                        found = True; break
                if not found: missing.append(tok)
check(not missing, f"all backticked numbers in markdown appear in printed output (missing {sorted(set(missing))})")

# 6. git hygiene
def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()
check(sh("git log --all --oneline -- data/instructor_data.csv") == "", "CSV is in NO commit")
check(sh("git ls-files data") == "", "CSV not tracked")
check("data/instructor_data.csv" in open(".gitignore").read(), "CSV is gitignored")
dirty = sh("git diff --ignore-space-at-eol --ignore-cr-at-eol --stat") + sh("git diff --cached --stat") + sh("git ls-files --others --exclude-standard")
check(dirty == "", f"working tree fully committed (uncommitted: {dirty[:120]!r})")
check(sh("git remote") == "", "no git remote configured / nothing pushed")
print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL(S)")
sys.exit(1 if fails else 0)
