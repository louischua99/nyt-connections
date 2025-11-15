
import os, json, time, argparse, itertools
from pathlib import Path
import pandas as pd
import requests
from math import sqrt



SYSTEM_PROMPT = """
You are a strict evaluator comparing two model responses (Answer A vs Answer B) to the SAME puzzle.

You must judge BOTH:

1) Final answer correctness and task success.
2) Reasoning quality:
   - Logical, step-by-step thinking
   - Correct intermediate reasoning
   - No hallucinated or false steps
   - Coherence and structure
   - Faithful to the problem information

Weighting:
- Correctness > Reasoning faithfulness > Clarity > Format

If one answer is correct and the other is wrong, prefer the correct one.
If both are correct, choose the one with better reasoning quality.
If both are incorrect, choose the one with more reasonable and grounded reasoning.
If they are equally good/bad, output TIE.

Respond with ONLY ONE token: WIN_A, WIN_B, or TIE.
"""

def _payload(q, a, b):
    return {
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "messages": [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": f"Question:\n{q}\n\n[Answer A]\n{a}\n\n[Answer B]\n{b}\n\nOutput only: WIN_A | WIN_B | TIE"}
        ],
        "temperature": 0.0,
        "max_tokens": 8
    }



TIMEOUT=120
RETRIES=4
COOLDOWN=2
RATE_LIMIT = 30  # calls/min

def _post(payload):
    key = os.environ.get("DEEPSEEK_API_KEY", None)
    if not key:
        return "TIE"  # offline mode

    url = os.environ.get("DEEPSEEK_URL","https://api.deepseek.com/v1/chat/completions")
    headers = {"Authorization": f"Bearer {key}", "Content-Type":"application/json"}

    for i in range(RETRIES):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip().upper()
            if "WIN_A" in txt: return "WIN_A"
            if "WIN_B" in txt: return "WIN_B"
            return "TIE"
        except:
            time.sleep(COOLDOWN)
    return "TIE"


last_reset = time.time()
calls = 0

def _rate_throttle():
    global calls, last_reset
    now = time.time()
    if now - last_reset >= 60:
        last_reset, calls = now, 0
    if calls >= RATE_LIMIT:
        sleep = 60 - (now - last_reset)
        if sleep > 0: time.sleep(sleep)
        last_reset, calls = time.time(), 0
    calls += 1


def wilson_ci(w, n, z=1.96):
    if n == 0: return (0,0)
    p = w/n
    denom = 1 + z**2/n
    centre = p + z*z/(2*n)
    margin = z*sqrt((p*(1-p) + z*z/(4*n))/n)
    return ((centre-margin)/denom, (centre+margin)/denom)



def run_pairs(files, pairs, out_dir, checkpoint, max_examples):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(checkpoint)

    # load data
    def _readfile(f):
        d = json.loads(Path(f).read_text())
        if isinstance(d, dict):
            for k in ("examples","data","items"):
                if k in d and isinstance(d[k],list): return d[k]
        return d

    data = {name: {ex["puzzle_id"]:(ex["prediction"], ex["user_message"]) 
                   for ex in _readfile(path)} 
            for name, path in files.items()}

    seen=set(); rows=[]
    if ckpt.exists():
        df=pd.read_csv(ckpt)
        for _,r in df.iterrows():
            seen.add((r["pair"], r["id"]))
            rows.append(r.to_dict())

    for A,B in pairs:
        ids = sorted(set(data[A].keys()) & set(data[B].keys()))
        if max_examples: ids = ids[:max_examples]

        for pid in ids:
            tag=f"{A} vs {B}"
            if (tag, pid) in seen: continue

            a,q = data[A][pid]
            b,_ = data[B][pid]

            _rate_throttle()
            vote = _post(_payload(q,a,b))
            rows.append({"pair":tag,"id":pid,"vote":vote})

            if len(rows) % 25 == 0:
                pd.DataFrame(rows).to_csv(ckpt,index=False)

    pd.DataFrame(rows).to_csv(ckpt,index=False)

    out = out_dir / "judge_summary.csv"
    agg=[]
    df=pd.DataFrame(rows)
    for name,g in df.groupby("pair"):
        wA=(g["vote"]=="WIN_A").sum()
        wB=(g["vote"]=="WIN_B").sum()
        t =(g["vote"]=="TIE").sum()
        n=wA+wB+t
        denom=max(1,n-t)
        wr = wA/denom if denom else 0
        lo,hi = wilson_ci(wA,denom)
        agg.append(dict(pair=name, wins_A=wA, wins_B=wB, ties=t, n=n,
                        win_rate_A_excl_ties=wr, ci_low=lo, ci_high=hi))
    pd.DataFrame(agg).to_csv(out,index=False)
    return out



EXPERIMENTS = {
    "exp1": ["exp1_baseline","exp1_permutation","exp1_full","exp1_synthetic"],
    "exp2": ["exp2_mixed","exp2_sequential","exp2_structured","exp2_unstructured"],
    "exp3": ["exp3_warmup","exp3_no_warmup","exp3_staged"],
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred-dir",type=Path,default=Path("."))
    ap.add_argument("--out-dir",type=Path,default=Path("results/judge_only"))
    ap.add_argument("--checkpoint-dir",type=Path,default=Path("results/judge_ckpts"))
    ap.add_argument("--max-examples",type=int,default=None)
    args=ap.parse_args()

    args.out_dir.mkdir(parents=True,exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True,exist_ok=True)

    all_rows=[]

    for exp,models in EXPERIMENTS.items():
        print(f"\n=== {exp} ===")
        files={m:str(args.pred_dir/f"{m}.json") for m in models if (args.pred_dir/f"{m}.json").exists()}
        pairs=list(itertools.combinations(models,2))
        exp_dir=args.out_dir/exp; exp_dir.mkdir(parents=True,exist_ok=True)

        for A,B in pairs:
            ck = args.checkpoint_dir / f"{exp}_{A}_vs_{B}.csv"
            out = run_pairs(files,[(A,B)],exp_dir,ck,args.max_examples)
            df=pd.read_csv(out); df["experiment"]=exp
            all_rows.append(df)

    pd.concat(all_rows).to_csv(args.out_dir/"all_judge_summary.csv",index=False)
    print("\n DONE — results saved.")


if __name__ == "__main__":
    main()

