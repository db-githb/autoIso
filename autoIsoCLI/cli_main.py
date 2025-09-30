import os, torch
caps = {torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())}
os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(f"{m}.{n}" for m, n in sorted(caps))

import argparse
from pathlib import Path
from autoIsoPY.main.autoIso_main import autoIso

def main():
    parser = argparse.ArgumentParser(prog="autoIso", description="autoIso image processing tools")
    sub = parser.add_subparsers(dest="command", required=True)
    
    p_cull = sub.add_parser("isolate-car", help="Create car mask from 3DGS model")
    p_cull.add_argument("--load-config", "-l", required=True,
                        help="path to 3DGS model's yaml configuration file")
    p_cull.add_argument("--output-dir", "-o", default=None,
                        help="Path to output directory")
    p_cull.add_argument("--stat-thresh", "-s", type=float, default=0.5,
                        help="Cull Gaussians whose distance from the scene’s median center exceeds this value, measured in per-axis standard deviations")

    args = parser.parse_args()
    if args.command == "isolate-car":
        dc = autoIso(Path(args.load_config), output_dir=args.output_dir, stat_thresh=args.stat_thresh)
        dc.run_iso()

with torch.inference_mode():
    if __name__ == "__main__":
        main()