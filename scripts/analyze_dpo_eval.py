"""分析 DPO 评估结果：对比 original ProteinMPNN vs DPO 微调后的指标。

Usage:
    python scripts/analyze_dpo_eval.py results/dpo/eval_results.json
    python scripts/analyze_dpo_eval.py results/dpo/eval_results.json --output results/dpo/analysis
"""

import json
import argparse
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(description="Analyze DPO evaluation results")
    parser.add_argument("eval_json", type=str, help="Path to eval_results.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save summary CSV and report (default: same dir as JSON)")
    args = parser.parse_args()

    with open(args.eval_json) as f:
        data = json.load(f)

    orig = data["original"]
    dpo = data["dpo"]

    # 按 pdb_name 对齐（顺序应一致）
    by_name = {}
    for r in orig:
        by_name[r["pdb_name"]] = {"original": r}
    for r in dpo:
        if r["pdb_name"] in by_name:
            by_name[r["pdb_name"]]["dpo"] = r

    pairs = [v for v in by_name.values() if "dpo" in v]
    n = len(pairs)

    # 提取数组
    o_mean = np.array([p["original"]["proagg_mean"] for p in pairs])
    o_max = np.array([p["original"]["proagg_max"] for p in pairs])
    o_logp = np.array([p["original"]["mpnn_logprob_mean"] for p in pairs])
    o_div = np.array([p["original"]["diversity"] for p in pairs])

    d_mean = np.array([p["dpo"]["proagg_mean"] for p in pairs])
    d_max = np.array([p["dpo"]["proagg_max"] for p in pairs])
    d_logp = np.array([p["dpo"]["mpnn_logprob_mean"] for p in pairs])
    d_div = np.array([p["dpo"]["diversity"] for p in pairs])

    delta_mean = d_mean - o_mean
    delta_max = d_max - o_max
    delta_logp = d_logp - o_logp
    delta_div = d_div - o_div

    # 汇总统计
    print("=" * 60)
    print("DPO 评估结果分析")
    print("=" * 60)
    print(f"Test PDB 数量: {n}")
    print()

    print("--- ProAgg 分数 (越高越不易聚集) ---")
    print(f"  ProAgg 均分:")
    print(f"    Original   mean={o_mean.mean():.4f}  std={o_mean.std():.4f}")
    print(f"    DPO        mean={d_mean.mean():.4f}  std={d_mean.std():.4f}")
    print(f"    Delta      mean={delta_mean.mean():+.4f}  std={delta_mean.std():.4f}")
    print(f"  ProAgg 最高分 (每 backbone 最优序列):")
    print(f"    Original   mean={o_max.mean():.4f}  std={o_max.std():.4f}")
    print(f"    DPO        mean={d_max.mean():.4f}  std={d_max.std():.4f}")
    print(f"    Delta      mean={delta_max.mean():+.4f}  std={delta_max.std():.4f}")
    print()

    print("--- 逆折叠质量 (MPNN log-prob, 越接近 0 越好) ---")
    print(f"  Original   mean={o_logp.mean():.4f}  std={o_logp.std():.4f}")
    print(f"  DPO        mean={d_logp.mean():.4f}  std={d_logp.std():.4f}")
    print(f"  Delta      mean={delta_logp.mean():+.4f}  (负=略降, 可接受)")
    print()

    print("--- 序列多样性 (平均归一化 Hamming 距离) ---")
    print(f"  Original   mean={o_div.mean():.4f}  std={o_div.std():.4f}")
    print(f"  DPO        mean={d_div.mean():.4f}  std={d_div.std():.4f}")
    print(f"  Delta      mean={delta_div.mean():+.4f}")
    print()

    # 比例统计
    better_mean = (delta_mean > 0).sum()
    better_max = (delta_max > 0).sum()
    print("--- 按 backbone 统计 ---")
    print(f"  ProAgg 均分 提升的 backbone 比例: {better_mean}/{n} = {100*better_mean/n:.1f}%")
    print(f"  ProAgg 最高分 提升的 backbone 比例: {better_max}/{n} = {100*better_max/n:.1f}%")
    print("=" * 60)

    # 保存 CSV 和报告
    out_dir = args.output if args.output else os.path.dirname(args.eval_json)
    os.makedirs(out_dir, exist_ok=True)

    names = [p["original"]["pdb_name"] for p in pairs]
    csv_path = os.path.join(out_dir, "dpo_eval_summary.csv")
    with open(csv_path, "w") as f:
        f.write("pdb_name,proagg_mean_orig,proagg_mean_dpo,delta_mean,proagg_max_orig,proagg_max_dpo,delta_max,logprob_orig,logprob_dpo,diversity_orig,diversity_dpo\n")
        for i, name in enumerate(names):
            f.write(f"{name},{o_mean[i]:.6f},{d_mean[i]:.6f},{delta_mean[i]:.6f},"
                    f"{o_max[i]:.6f},{d_max[i]:.6f},{delta_max[i]:.6f},"
                    f"{o_logp[i]:.6f},{d_logp[i]:.6f},"
                    f"{o_div[i]:.6f},{d_div[i]:.6f}\n")
    print(f"\nPer-PDB 汇总已保存: {csv_path}")

    report_path = os.path.join(out_dir, "dpo_eval_report.txt")
    with open(report_path, "w") as f:
        f.write("DPO Evaluation Report\n")
        f.write("====================\n")
        f.write(f"eval_json: {args.eval_json}\n")
        f.write(f"N (test backbones): {n}\n\n")
        f.write(f"ProAgg mean:  orig={o_mean.mean():.4f}  dpo={d_mean.mean():.4f}  delta={delta_mean.mean():+.4f}\n")
        f.write(f"ProAgg max:   orig={o_max.mean():.4f}  dpo={d_max.mean():.4f}  delta={delta_max.mean():+.4f}\n")
        f.write(f"MPNN logprob: orig={o_logp.mean():.4f}  dpo={d_logp.mean():.4f}  delta={delta_logp.mean():+.4f}\n")
        f.write(f"Diversity:    orig={o_div.mean():.4f}  dpo={d_div.mean():.4f}  delta={delta_div.mean():+.4f}\n")
        f.write(f"\nBackbones with improved ProAgg mean: {better_mean}/{n} ({100*better_mean/n:.1f}%)\n")
        f.write(f"Backbones with improved ProAgg max:  {better_max}/{n} ({100*better_max/n:.1f}%)\n")
    print(f"文字报告已保存: {report_path}")


if __name__ == "__main__":
    main()
