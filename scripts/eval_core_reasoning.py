

import argparse, json, re
from pathlib import Path
from typing import List
import pandas as pd


def load_json(path: Path):
    text = path.read_text(encoding="utf-8")
    if text.strip().startswith("["):
        return json.loads(text)
    return [json.loads(x) for x in text.splitlines() if x.strip()]

def strip_think(text: str) -> str:
    return re.sub(r"(?is)<think>.*?</think>", "", text).strip()

def after_think(text: str) -> str:
    parts = re.split(r"(?is)</think>", text)
    return parts[-1].strip() if parts else text

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  
    return s

def parse_groups(text: str):
    text = text.strip()
    lines = text.splitlines()
    groups = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = re.match(r"^\**([^:*]+?)\**\s*[:\-]\s*(.+)$", line)
        if not m:
            continue

        cat = m.group(1).lower().strip()
        items = [normalize(x) for x in m.group(2).split(",")]
        items = [x for x in items if x]

        if len(items) == 4:
            groups.append(frozenset(items))

    return frozenset(groups)


def precision_recall_f1(preds: List[str], refs: List[str]):
    precs, recs, f1s = [], [], []
    for p, r in zip(preds, refs):
        ps, rs = set(normalize(p).split()), set(normalize(r).split())
        tp = len(ps & rs)
        fp = len(ps - rs)
        fn = len(rs - ps)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        precs.append(prec); recs.append(rec); f1s.append(f1)
    return sum(precs)/len(precs), sum(recs)/len(recs), sum(f1s)/len(f1s)

def process_file(path: Path):
    data = load_json(path)
    preds, refs = [], []
    for ex in data:
        p = ex.get("prediction"); r = ex.get("ground_truth")
        if p is None or r is None:
            raise ValueError(f"Missing prediction/ground_truth in {path}")
        if not isinstance(p, str): p = str(p)
        if not isinstance(r, str): r = str(r)
        p, r = strip_think(p), strip_think(r)
        p, r = after_think(p), after_think(r)
        preds.append(p); refs.append(r)
    prec, rec, f1 = precision_recall_f1(preds, refs)
    return {"Precision": prec, "Recall": rec, "Macro-F1": f1, "n": len(preds)}

def write_csv(rows, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        headers = rows[0].keys()
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r[h]) for h in headers) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", type=Path, default=Path("predictions"))
    ap.add_argument("--output", type=Path, default=Path("predictions/data/results/core_only"))
    args = ap.parse_args()

    experiments = {
        "exp1": ["exp1_baseline", "exp1_permutation", "exp1_full", "exp1_synthetic"],
        "exp2": ["exp2_mixed", "exp2_sequential", "exp2_structured", "exp2_unstructured"],
        "exp3": ["exp3_warmup", "exp3_no_warmup", "exp3_staged"],
    }

    combined = []

    for exp, models in experiments.items():
        rows = []
        for m in models:
            path = args.predictions_dir / f"{m}.json"
            if not path.exists(): 
                print(f"⚠️ Missing file: {path}")
                continue
            metrics = process_file(path)
            row = {"experiment": exp, "model": m, **metrics}
            rows.append(row); combined.append(row)
        if rows:
            write_csv(rows, args.output/exp/"summary_core.csv")
            print(f"Wrote {exp} → {args.output/exp/'summary_core.csv'}")

    if combined:
        write_csv(combined, args.output/"all_core_summary.csv")
        print(f" Wrote combined → {args.output/'all_core_summary.csv'}")

if __name__ == "__main__":
    main()



REASONING_KEYS = ("reasoning","rationale","chain_of_thought","steps")

STEP_PATTERNS = [
    r"\bstep\s*\d+\b",           
    r"^\s*-\s",                  
    r"^\s*\d+\.\s",              
    r"\btherefore\b|\bthus\b|\bbecause\b",
]
STEP_RE = re.compile("|".join(STEP_PATTERNS), flags=re.IGNORECASE|re.MULTILINE)

def _read_json(path: Path):
    text = path.read_text(encoding="utf-8")
    if text.strip().startswith("["):
        return json.loads(text)
    return [json.loads(x) for x in text.splitlines() if x.strip()]

def _has_explicit_reasoning(ex: dict) -> bool:
    for k in REASONING_KEYS:
        v = ex.get(k)
        if isinstance(v, str) and v.strip(): return True
        if isinstance(v, list) and len(v) > 0: return True
    md = ex.get("metadata")
    if isinstance(md, dict):
        for k in REASONING_KEYS:
            v = md.get(k)
            if isinstance(v, str) and v.strip(): return True
            if isinstance(v, list) and len(v) > 0: return True
    return False

def _infer_from_prediction(ex: dict):
    txt = ex.get("prediction") or ""
    if not isinstance(txt, str) or not txt.strip():
        return False, 0
    matches = list(STEP_RE.finditer(txt))
    if not matches:
        return False, 0
    step_count = max(1, len(matches))
    return True, step_count

def eval_reasoning_file(path: Path):
    data = _read_json(path)
    n = len(data)
    present = 0
    step_sum = 0

    for ex in data:
        # explicit reasoning
        if _has_explicit_reasoning(ex):
            present += 1
            v = ex.get("reasoning") or (ex.get("steps") if isinstance(ex.get("steps"), list) else None)
            step_sum += (len(v) if isinstance(v, list) else max(1, str(v).count("\n")+1))
        else:  # infer
            inf, steps = _infer_from_prediction(ex)
            if inf:
                present += 1
                step_sum += steps

    coverage = present / max(1, n)
    avg_steps = (step_sum / present) if present else 0.0

    return {
        "n_examples": n,
        "coverage_ratio": coverage,
        "avg_step_count_if_present": avg_steps,
    }

def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f" Wrote {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, default=Path("results/reasoning_only"))
    args = ap.parse_args()

    experiments = {
        "exp1": ["exp1_baseline", "exp1_permutation", "exp1_full", "exp1_synthetic"],
        "exp2": ["exp2_mixed", "exp2_sequential", "exp2_structured", "exp2_unstructured"],
        "exp3": ["exp3_warmup", "exp3_no_warmup", "exp3_staged"],
    }

    combined = []

    for exp, models in experiments.items():
        rows = []
        for m in models:
            path = args.predictions_dir / f"{m}.json"
            if not path.exists():
                print(f"⚠️ Missing {path}")
                continue

            metrics = eval_reasoning_file(path)
            row = {"experiment": exp, "model": m, **metrics}
            rows.append(row)
            combined.append(row)

        if rows:
            out_file = args.output / exp / "summary_reasoning.csv"
            write_csv(rows, out_file)

    if combined:
        write_csv(combined, args.output / "all_reasoning_summary.csv")

if __name__ == "__main__":
    main()
