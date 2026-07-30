#!/usr/bin/env python
"""Summarize the per-timestep diffusion diagnostics and flag the first failure."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run",required=True); args=ap.parse_args()
    run=Path(args.run); rows=[json.loads(x) for x in (run/"train_metrics.jsonl").read_text().splitlines() if x.strip()]
    if not rows: raise SystemExit("no train_metrics.jsonl rows")
    def avg(k): return sum(float(r[k]) for r in rows[-min(10,len(rows)):])/min(10,len(rows))
    diagnosis=[]
    if avg("presence_active_recall") < .8 or avg("presence_inactive_rejection") < .8: diagnosis.append("presence channel is not separating active/inactive strokes")
    if avg("x0_bbox_mse") > avg("x0_shape_mse") * 3: diagnosis.append("bbox state is the dominant geometry error")
    if avg("x0_shape_mse") > .1: diagnosis.append("shape latent is not denoising")
    if avg("noise_mse") > 0.5: diagnosis.append("overall noise mse is high")
    keys=("loss","noise_mse","x0_shape_mse","x0_bbox_mse","presence_sign_acc","presence_active_recall","presence_inactive_rejection","active_count_mean","grad_norm")
    result={"rows":len(rows),"last":rows[-1],"recent_means":{k:avg(k) for k in keys if k in rows[-1]},"diagnosis":diagnosis or ["no threshold failure detected; inspect generated preview"]}
    (run/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        x=[r["step"] for r in rows]; fig,ax=plt.subplots(2,2,figsize=(12,8))
        for a, keys, title in zip(ax.flat, [("loss",), ("x0_shape_mse","x0_bbox_mse"), ("presence_active_recall","presence_inactive_rejection"), ("noise_mse",)], ["loss","x0 errors","presence","noise mse"]):
            for k in keys: a.plot(x,[r[k] for r in rows],label=k)
            a.set_title(title); a.legend(); a.grid(alpha=.2)
        fig.tight_layout(); fig.savefig(run/"diagnostics.png",dpi=140); plt.close(fig)
    except Exception as e: result["plot_error"]=str(e)
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__": main()
