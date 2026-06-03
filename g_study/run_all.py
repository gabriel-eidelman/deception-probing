#!/usr/bin/env python3
"""
run_all.py -- run phases 4-7 in order against outputs/scored_responses.csv.

    python run_all.py                          # auto-find ./outputs
    python run_all.py --outdir path/to/outputs --boot 1000
    python run_all.py --activations outputs/activations.npz   # Phase 5 mode B

Each phase is a normal module; this just calls their main() with shared args so
you don't have to invoke four scripts by hand. Stops if Phase 4 fails (others
depend on its JSON); otherwise continues and reports what succeeded.
"""
import argparse
import runpy
import sys
from pathlib import Path

PHASES = [
    ("phase4_gstudy.py", "G-study variance decomposition"),
    ("phase5_consistency.py", "consistency / AUROC-transfer check"),
    ("phase6_validity.py", "construct-validity checks"),
    ("phase7_synthesis.py", "synthesis & main results table"),
]


def run(script, argv):
    here = Path(__file__).resolve().parent
    sys.argv = [script] + argv
    runpy.run_path(str(here / script), run_name="__main__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--activations", default=None)
    args = ap.parse_args()

    common = []
    if args.outdir:
        common += ["--outdir", args.outdir]

    plans = {
        "phase4_gstudy.py": common + ["--boot", str(args.boot),
                                      "--seed", str(args.seed)],
        "phase5_consistency.py": common + (["--activations", args.activations]
                                           if args.activations else []),
        "phase6_validity.py": common + ["--topk", str(args.topk)],
        "phase7_synthesis.py": common,
    }

    for script, desc in PHASES:
        print("\n" + "=" * 72)
        print(f"  RUNNING {script}  --  {desc}")
        print("=" * 72)
        try:
            run(script, plans[script])
        except SystemExit as e:
            if e.code not in (0, None):
                print(f"!! {script} exited with code {e.code}")
                if script == "phase4_gstudy.py":
                    print("Phase 4 is required by later phases. Stopping.")
                    sys.exit(e.code)
        except Exception as e:
            print(f"!! {script} raised {type(e).__name__}: {e}")
            if script == "phase4_gstudy.py":
                raise
    print("\nAll done. See <outputs>/analysis/ for tables and figures.")


if __name__ == "__main__":
    main()
