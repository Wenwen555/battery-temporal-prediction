import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_prediction_dirs(root:Path, prefix="Predictor_"):
    out = []
    low_prefix = prefix.lower()
    for d in sorted(root.iterdir()):
        # 只匹配目录的 basename，忽略大小写，避免父路径影响
        if d.is_dir() and d.name.lower().startswith(low_prefix):
            out.append(d)
    print(out)
    return out

def parse_text_for_metric(path:Path, metric_label:str):
    # 支持形式： "Test MAPE loss: 3.45%", "Test RMSE loss = 0.123", 等
    text = path.read_text(errors='ignore')
    # 构造不区分大小写的正则：metric_label 后可能跟 ":" "=" 或空格，然后数字（可带小数、可带%）
    pat = re.compile(r'(?i)'+re.escape(metric_label)+r'[^0-9\-\n\r%]*([-+]?\d+(?:\.\d+)?)(?:\s*%?)')
    vals = [float(m.group(1)) for m in pat.finditer(text)]
    return vals

def collect_metrics_for_dir(top_dir:Path, metric_labels):
    files = list(top_dir.rglob("*.log")) + list(top_dir.rglob("*.txt"))
    res = {lbl: [] for lbl in metric_labels}
    for f in files:
        for lbl in metric_labels:
            try:
                vals = parse_text_for_metric(f, lbl)
            except:
                vals = []
            if vals:
                res[lbl].extend(vals)
    return res

def summarize(root, out_dir:Path, prefix="Predictor_"):
    root = Path(root)
    dirs = find_prediction_dirs(root, prefix)

    if not dirs:
        print("未找到任何以指定前缀开头的目录。")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # 要解析的 metric 标签（与日志中字段完全或部分匹配）
    metric_labels = ["Test MAPE loss", "Test RMSE loss"]
    rows = []
    per_model_vals = {lbl: {} for lbl in metric_labels}

    for d in dirs:
        collected = collect_metrics_for_dir(d, metric_labels)
        short_name = d.name  # 只保留 basename，去掉父路径
        row = {"model": short_name}
        for lbl in metric_labels:
            vals = np.array(collected.get(lbl, []) or [], dtype=float)
            # 如果没找到值，尝试只用短标签（MAPE, RMSE）
            if vals.size == 0:
                vals = np.array(collect_metrics_for_dir(d, [lbl.split()[-1]])[lbl.split()[-1]] or [], dtype=float)
            if vals.size == 0:
                row[lbl + "_n"] = 0
                row[lbl + "_mean"] = np.nan
                row[lbl + "_std"] = np.nan
            else:
                row[lbl + "_n"] = vals.size
                row[lbl + "_mean"] = float(vals.mean())
                row[lbl + "_std"] = float(vals.std(ddof=0))
                per_model_vals[lbl][short_name] = vals  # 用短名作为 key
        rows.append(row)

    df = pd.DataFrame(rows)
    # 打印汇总表
    print("\nSummary per model (mean ± std, n):")
    display_cols = []
    for lbl in metric_labels:
        display_cols += [f"{lbl}_n", f"{lbl}_mean", f"{lbl}_std"]
    print(df[["model"] + display_cols].to_string(index=False))

    # 为每 metric 做 box plot（只包含有数据的模型），不画原始点，只画箱线并叠加均值和 std
    for lbl in metric_labels:
        data = per_model_vals[lbl]
        if not data:
            print(f"\n{lbl}: 未找到数据，跳过绘图。")
            continue

        # 按 mean 排序，保证可读性
        models_sorted = sorted(data.items(), key=lambda kv: float(kv[1].mean()))
        model_names = [m for m, _ in models_sorted]
        means = [float(vals.mean()) for _, vals in models_sorted]
        stds = [float(vals.std(ddof=0)) for _, vals in models_sorted]

        # 展开用于 seaborn.boxplot 的 DataFrame（boxplot 仍需样本数据）
        exp = []
        for m, vals in models_sorted:
            for v in vals:
                exp.append({"model": m, "value": float(v)})
        exp_df = pd.DataFrame(exp)

        out_box = out_dir / f"{lbl.replace(' ','_')}_box.png"
        plt.figure(figsize=(8,5))
        # 不显示离群点（showfliers=False），不绘制原始点
        ax = sns.boxplot(data=exp_df, x='model', y='value', palette='pastel', order=model_names, showfliers=False)
        # 在 boxplot 上叠加均值和 std（errorbar）
        x_pos = np.arange(len(model_names))
        ax.errorbar(x_pos, means, yerr=stds, fmt='o', color='red', capsize=4, label='mean ± std')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(lbl)
        ax.set_title(f"{lbl} distribution across runs (mean±std overlay)")
        plt.tight_layout()
        plt.legend()
        plt.savefig(str(out_box), dpi=150)
        plt.close()
        print(f"\n{lbl} boxplot saved: {out_box}")

        # 保存该 metric 的 summary csv（每个 prediction 的均值/std/n）
        summary_rows = []
        for m, vals in models_sorted:
            summary_rows.append({"model": m, "n": len(vals), "mean": float(vals.mean()), "std": float(vals.std(ddof=0))})
        summary_df = pd.DataFrame(summary_rows).sort_values("mean")
        out_csv = out_dir / f"summary_{lbl.replace(' ','_')}.csv"
        summary_df.to_csv(out_csv, index=False)
        print(f"{lbl} summary saved: {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/wenjt5/TC-SOH/experiments_logs/xjtu", help="experiments root")
    parser.add_argument("--out", default="/mnt/wenjt5/TC-SOH/experiments_logs/xjtu/compare_predictors_prediction", help="output directory for plots and csv")
    parser.add_argument("--prefix", default="Predictor_", help="top-level folder prefix to include (case-insensitive)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summarize(args.root, out_dir, args.prefix)