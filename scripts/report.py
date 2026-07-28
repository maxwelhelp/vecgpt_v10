#!/usr/bin/env python
"""Turn a run into a compact, pasteable diagnosis.

    python scripts/report.py runs/v1/log.jsonl

Prints, per stage: how each field's information gain moved over the run,
where it ended up, and an explicit verdict on which fields are reading the
image and which are sitting on the marginal prior. That verdict is the
thing worth arguing about - not a rendered grid, and not the total loss,
which can look fine while half the fields are constants.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

FIELD_ORDER = ["x", "y", "theta", "len", "turn", "width", "r", "g", "b", "eos", "eol"]


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def sparkline(vals, lo=0.0, hi=None):
    if not vals:
        return ""
    chars = " .:-=+*#%@"
    hi = hi if hi is not None else max(max(vals), 1e-6)
    span = max(hi - lo, 1e-6)
    return "".join(chars[min(int((v - lo) / span * (len(chars) - 1)), len(chars) - 1)]
                   if v > lo else chars[0] for v in vals)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", help="path to log.jsonl")
    p.add_argument("--flat-threshold", type=float, default=0.05,
                   help="gain (nats) below which a field counts as 'not learning'")
    a = p.parse_args()

    recs = load(a.log)
    if not recs:
        sys.exit("empty log")

    meta = next((r for r in recs if r.get("kind") == "meta"), {})
    train = [r for r in recs if r.get("kind") == "train"]
    evals = [r for r in recs if r.get("kind") == "eval"]

    print("=" * 78)
    print("VECGPT RUN REPORT")
    print("=" * 78)
    if meta:
        c = meta.get("cfg", {})
        print(f"params {meta.get('params_M')}M  vocab {meta.get('vocab')}  "
              f"device {meta.get('device')}  torch {meta.get('torch')}")
        print(f"d_model {c.get('d_model')}  layers {c.get('n_layers')}  heads {c.get('n_heads')} "
              f"(seg {c.get('n_seg_heads')}/stroke {c.get('n_stroke_heads')})  "
              f"enc_layers {c.get('n_enc_layers')}  image {c.get('image_size')}")
        print(f"batch {c.get('batch_size')}  lr {c.get('lr')}  "
              f"label_smooth_bins {c.get('label_smooth_bins')}  "
              f"schedule {c.get('stage_schedule')}")
        print(f"throughput {meta.get('scenes_per_sec')} scenes/s  "
              f"({meta.get('sec_per_step')} s/step)")
    if train:
        print(f"steps logged: {train[-1]['step'] + 1}  "
              f"wall {train[-1]['elapsed_s'] / 3600:.2f} h (last stage)")

    by_stage = defaultdict(list)
    for r in train:
        by_stage[r["stage"]].append(r)

    for stage in sorted(by_stage):
        rs = by_stage[stage]
        print("\n" + "-" * 78)
        print(f"STAGE {stage}   ({len(rs)} log points, "
              f"steps {rs[0]['stage_step']}..{rs[-1]['stage_step']})")
        print("-" * 78)
        series = defaultdict(list)
        mae = {}
        ce = {}
        hm = {}
        for r in rs:
            for f in r["fields"]:
                series[f["field"]].append(f["gain"])
                mae[f["field"]] = f["mae_bins"]
                ce[f["field"]] = f["ce"]
                hm[f["field"]] = f["h_marginal"]
        print(f"{'field':7s} {'gain over stage':22s} {'final':>7s} {'CE':>7s} "
              f"{'H_marg':>7s} {'mae':>6s}   verdict")
        for f in FIELD_ORDER:
            if f not in series:
                continue
            v = series[f]
            fin = v[-1]
            if f in ("eos", "eol"):
                verdict = "deterministic (gain 0 is correct)"
            elif fin < a.flat_threshold:
                verdict = "FLAT - not reading the image"
            elif fin < 0.3:
                verdict = "weak"
            elif fin < 1.0:
                verdict = "learning"
            else:
                verdict = "strong"
            trend = "rising" if len(v) > 3 and fin > max(v[: max(len(v) // 2, 1)]) else "plateaued"
            print(f"{f:7s} {sparkline(v):22s} {fin:+7.2f} {ce[f]:7.2f} {hm[f]:7.2f} "
                  f"{mae[f]:6.0f}   {verdict}, {trend}")

        se = [e for e in evals if e["stage"] == stage]
        if se:
            print()
            for e in se:
                ind, ood, ceil = e["in_dist"], e["ood"], e.get("ceiling", 0)
                frac = 100 * ind["iou"] / max(ceil, 1e-6)
                print(f"  eval @ {e['stage_step']:6d}  IoU {ind['iou']:.3f} "
                      f"shape {ind.get('iou_shape', float('nan')):.3f} "
                      f"/ ceiling {ceil:.3f} = {frac:3.0f}%   "
                      f"strokes {ind['n_strokes_pred']:.1f}/{ind['n_strokes_true']:.1f}  "
                      f"empty {ind['empty']}   OOD {ood['iou']:.3f}")
                if "families" in ood:
                    fams = ood["families"]
                    parts = []
                    for fn, f in fams.items():
                        parts.append(f"{fn} {f['iou']:.3f}|e{f['empty']}")
                    print(f"    OOD per family: " + "  ".join(parts))

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    if train:
        last = train[-1]
        flat = [f["field"] for f in last["fields"]
                if f["field"] not in ("eos", "eol") and f["gain"] < a.flat_threshold]
        ok = [f["field"] for f in last["fields"]
              if f["field"] not in ("eos", "eol") and f["gain"] >= a.flat_threshold]
        print(f"reading the image : {', '.join(ok) if ok else 'NOTHING'}")
        print(f"flat              : {', '.join(flat) if flat else 'none'}")
    if evals:
        e = evals[-1]
        gap = e["in_dist"]["iou"] - e["ood"]["iou"]
        print(f"final IoU {e['in_dist']['iou']:.3f} vs ceiling {e.get('ceiling', 0):.3f}; "
              f"OOD {e['ood']['iou']:.3f} (gap {gap:+.3f})")
        if gap > 0.15:
            print("  large in-dist/OOD gap -> memorising shape families rather than "
                  "generalising over them")
        fams = e["ood"].get("families")
        if fams:
            print()
            print("  OOD by family (these fail for DIFFERENT reasons - do not average them):")
            hint = {"star": "~20 segments: tests length generalisation",
                    "spiral": "4-6 arcs, monotone curvature: tests smooth shape",
                    "cross": "2 strokes of 1 straight segment: the EASIEST case",
                    "blob": "6 arcs, closed: tests closure",
                    "deeper": "tests depth generalisation",
                    "wider": "tests branching-factor generalisation",
                    "tiny": "tests unseen child/parent scale ratios",
                    "chain": "tests deep narrow composition"}
            for name, v in sorted(fams.items(), key=lambda kv: kv[1]["iou"]):
                print(f"    {name:7s} IoU {v['iou']:.3f} shape "
                      f"{v.get('iou_shape', float('nan')):.3f}  empty {v['empty']:2d}  "
                      f"strokes {v['n_strokes_pred']:.1f}/{v['n_strokes_true']:.1f}"
                      f"   ({hint.get(name, '')})")
            worst = min(fams.items(), key=lambda kv: kv[1]["iou"])
            if fams.get("cross", {}).get("iou", 1) < 0.3:
                print("    -> even `cross` is failing. That is two straight strokes; if the")
                print("       model cannot do that, this is a concept/generalisation problem,")
                print("       not a sequence-length problem.")
            elif worst[0] == "star":
                print("    -> `star` worst while `cross` is fine: this looks like failure to")
                print("       generalise to LONGER sequences, not failure to generalise over")
                print("       shape. Curriculum/data problem, not architecture.")
        if e["in_dist"]["empty"] > 0:
            print(f"  {e['in_dist']['empty']} empty decodes -> the model is emitting EOS "
                  f"immediately for some inputs")


if __name__ == "__main__":
    main()
