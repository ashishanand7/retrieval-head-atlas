#!/usr/bin/env python3
"""Build publication-ready retrieval-circuit tables and figures from committed artifacts."""

from __future__ import annotations

import csv
import html
import json
import math
import statistics as stats
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = REPO_ROOT / "artifacts_phase2"
OUT_DIR = ARTIFACTS / "report_assets"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

SETTINGS = [
    {"target": 8192, "frac": 0.1, "frac_code": "0p1", "label": "8k / 0.1"},
    {"target": 8192, "frac": 0.5, "frac_code": "0p5", "label": "8k / 0.5"},
    {"target": 8192, "frac": 0.9, "frac_code": "0p9", "label": "8k / 0.9"},
    {"target": 16384, "frac": 0.1, "frac_code": "0p1", "label": "16k / 0.1"},
    {"target": 16384, "frac": 0.5, "frac_code": "0p5", "label": "16k / 0.5"},
    {"target": 16384, "frac": 0.9, "frac_code": "0p9", "label": "16k / 0.9"},
]

FUNCTIONAL_GROUPS = [
    "answer_address",
    "non_address_core",
    "core19",
    "strong13",
    "query_tail",
    "first_token_sink",
    "answer_address_inactive_control",
    "query_tail_inactive_control",
]

SINGLE_HEAD_GROUPS = [
    "address_L22H7",
    "address_L22H10",
    "address_L21H11",
    "support_L20H7",
    "support_L18H3",
    "support_L17H10",
    "support_L20H8",
    "support_L22H0",
    "support_L22H4",
    "activation_control_L20H5",
    "inactive_control_L21H2",
    "inactive_control_L22H8",
    "inactive_control_L22H3",
]

ATTENTION_HEADS = ["L22H7", "L22H10", "L21H11", "L20H7", "L18H3", "L17H10", "L22H0", "L22H4"]

ROLE_LABELS = {
    "answer_address": "Address heads",
    "non_address_core": "Non-address core",
    "query_tail": "Query-tail support",
    "first_token_sink": "First-token/sink",
    "core19": "Core 19",
    "strong13": "Strong 13",
    "answer_address_inactive_control": "Address inactive control",
    "query_tail_inactive_control": "Query-tail inactive control",
}

HEAD_LABELS = {
    "address_L22H7": "L22H7",
    "address_L22H10": "L22H10",
    "address_L21H11": "L21H11",
    "activation_control_L20H5": "L20H5 control",
    "inactive_control_L21H2": "L21H2 inactive",
    "inactive_control_L22H8": "L22H8 inactive",
    "inactive_control_L22H3": "L22H3 inactive",
}

PALETTE = {
    "address": "#2563eb",
    "support": "#dc2626",
    "query": "#f97316",
    "sink": "#16a34a",
    "green": "#16a34a",
    "companion": "#7c3aed",
    "control": "#6b7280",
    "grid": "#e5e7eb",
    "ink": "#111827",
    "muted": "#6b7280",
}


def suffix(setting: dict) -> str:
    if setting["target"] == 8192 and setting["frac_code"] == "0p1":
        return ""
    return f"_frac{setting['frac_code']}"


def artifact_path(kind: str, setting: dict, summary: bool = True) -> Path:
    ext = "_summary.json" if summary else ".csv"
    target = setting["target"]
    suf = suffix(setting)
    if kind == "ablation":
        return ARTIFACTS / f"semantic_group_ablation_functional_controls_{target}_n8{suf}_query{ext}"
    if kind == "patch":
        return ARTIFACTS / f"semantic_component_patching_functional_{target}_n8{suf}{ext}"
    if kind == "single_patch":
        return ARTIFACTS / f"semantic_single_head_patching_{target}_n8{suf}{ext}"
    if kind == "attention":
        return ARTIFACTS / f"semantic_attention_trace_core19_{target}_n8{suf}{ext}"
    if kind == "activation":
        return ARTIFACTS / f"semantic_activation_delta_functional_{target}_n8{suf}{ext}"
    raise ValueError(f"Unknown artifact kind: {kind}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def ci(values: list[float], z: float = 1.96) -> tuple[float, float, float, float]:
    values = [float(v) for v in values]
    if not values:
        return (math.nan, math.nan, math.nan, math.nan)
    mean = float(stats.fmean(values))
    if len(values) == 1:
        return mean, mean, mean, 0.0
    sd = float(stats.stdev(values))
    half_width = z * sd / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width, sd


def values_for(path: Path, group: str, field: str) -> list[float]:
    return [float(row[field]) for row in read_rows(path) if row.get("group") == group]


def attention_by_head(summary: dict) -> dict[str, dict]:
    return {f"L{row['layer']}H{row['head']}": row for row in summary["by_head"]}


def build_tables() -> dict[str, pd.DataFrame]:
    main_rows = []
    ablation_rows = []
    patch_rows = []
    single_rows = []
    attention_activation_rows = []
    stats_rows = []

    for setting in SETTINGS:
        label = setting["label"]
        target = setting["target"]
        frac = setting["frac"]

        ab_summary = load_json(artifact_path("ablation", setting))
        patch_summary = load_json(artifact_path("patch", setting))
        single_summary = load_json(artifact_path("single_patch", setting))
        attention_summary = load_json(artifact_path("attention", setting))
        activation_summary = load_json(artifact_path("activation", setting))

        ab_csv = artifact_path("ablation", setting, summary=False)
        patch_csv = artifact_path("patch", setting, summary=False)
        single_csv = artifact_path("single_patch", setting, summary=False)
        single_rows_raw = read_rows(single_csv)
        attn_heads = attention_by_head(attention_summary)

        prompt_tokens = [float(row["prompt_tokens"]) for row in single_rows_raw]
        clean_lp = [float(row["clean_logprob"]) for row in single_rows_raw]
        corrupt_clean_lp = [float(row["corrupt_clean_logprob"]) for row in single_rows_raw]
        corrupt_gold_lp = [float(row["corrupt_gold_logprob"]) for row in single_rows_raw]
        recovery_gap = [float(row["recovery_denominator"]) for row in single_rows_raw]
        l22h7_values = values_for(single_csv, "address_L22H7", "delta_patch")
        l22h7_mean, l22h7_low, l22h7_high, _ = ci(l22h7_values)

        main_rows.append(
            {
                "setting": label,
                "target_tokens": target,
                "needle_frac": frac,
                "n_examples": single_summary["n_examples"],
                "prompt_tokens_mean": np.mean(prompt_tokens),
                "prompt_tokens_min": min(prompt_tokens),
                "prompt_tokens_max": max(prompt_tokens),
                "clean_logprob": np.mean(clean_lp),
                "corrupt_clean_logprob": np.mean(corrupt_clean_lp),
                "corrupt_gold_logprob": np.mean(corrupt_gold_lp),
                "recovery_gap": np.mean(recovery_gap),
                "answer_address_ablation": ab_summary["by_group"]["answer_address"]["mean_delta_logprob"],
                "non_address_core_ablation": ab_summary["by_group"]["non_address_core"]["mean_delta_logprob"],
                "query_tail_ablation": ab_summary["by_group"]["query_tail"]["mean_delta_logprob"],
                "answer_address_patch": patch_summary["by_group"]["answer_address"]["mean_delta_patch"],
                "non_address_core_patch": patch_summary["by_group"]["non_address_core"]["mean_delta_patch"],
                "l22h7_patch": l22h7_mean,
                "l22h7_patch_ci_low": l22h7_low,
                "l22h7_patch_ci_high": l22h7_high,
                "l22h7_positive_examples": single_summary["by_group"]["address_L22H7"]["positive_patch_examples"],
                "l22h7_gold_attention": attn_heads["L22H7"]["mean_gold_mass"],
                "l22h7_needle_attention": attn_heads["L22H7"]["mean_needle_mass"],
                "l22h7_activation_relative_diff": activation_summary["by_head"]["L22H7"]["mean_relative_diff_l2"],
            }
        )

        for group in FUNCTIONAL_GROUPS:
            if group in ab_summary["by_group"]:
                vals = values_for(ab_csv, group, "delta_logprob")
                mean, low, high, sd = ci(vals)
                item = ab_summary["by_group"][group]
                ablation_rows.append(
                    {
                        "setting": label,
                        "target_tokens": target,
                        "needle_frac": frac,
                        "group": group,
                        "group_label": ROLE_LABELS.get(group, group),
                        "n_heads": item["n_heads"],
                        "mean_delta_logprob": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "std": sd,
                        "negative_examples": sum(v < 0 for v in vals),
                        "n": len(vals),
                    }
                )

            if group in patch_summary["by_group"]:
                vals = values_for(patch_csv, group, "delta_patch")
                mean, low, high, sd = ci(vals)
                item = patch_summary["by_group"][group]
                patch_rows.append(
                    {
                        "setting": label,
                        "target_tokens": target,
                        "needle_frac": frac,
                        "group": group,
                        "group_label": ROLE_LABELS.get(group, group),
                        "n_heads": item["n_heads"],
                        "mean_delta_patch": mean,
                        "ci_low": low,
                        "ci_high": high,
                        "std": sd,
                        "mean_recovery_fraction": item["mean_recovery_fraction"],
                        "positive_examples": sum(v > 0 for v in vals),
                        "n": len(vals),
                    }
                )

        for group in SINGLE_HEAD_GROUPS:
            vals = values_for(single_csv, group, "delta_patch")
            mean, low, high, sd = ci(vals)
            item = single_summary["by_group"][group]
            head = item["heads"][0]
            single_rows.append(
                {
                    "setting": label,
                    "target_tokens": target,
                    "needle_frac": frac,
                    "group": group,
                    "head": f"L{head[0]}H{head[1]}",
                    "head_label": HEAD_LABELS.get(group, group.replace("_", " ")),
                    "mean_delta_patch": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "std": sd,
                    "mean_recovery_fraction": item["mean_recovery_fraction"],
                    "positive_examples": sum(v > 0 for v in vals),
                    "n": len(vals),
                }
            )

        for head in ATTENTION_HEADS:
            att = attn_heads[head]
            act = activation_summary["by_head"][head]
            attention_activation_rows.append(
                {
                    "setting": label,
                    "target_tokens": target,
                    "needle_frac": frac,
                    "head": head,
                    "gold_attention": att["mean_gold_mass"],
                    "needle_attention": att["mean_needle_mass"],
                    "distractor_attention": att["mean_distractor_mass"],
                    "argmax_in_needle_rate": att["argmax_in_needle_rate"],
                    "gold_rank": att["mean_gold_rank"],
                    "activation_relative_diff": act["mean_relative_diff_l2"],
                    "activation_cosine": act["mean_cosine_similarity"],
                }
            )

        for metric, csv_path, group, field, direction in [
            ("single_head_patch", single_csv, "address_L22H7", "delta_patch", "positive"),
            ("group_ablation", ab_csv, "non_address_core", "delta_logprob", "negative"),
            ("group_ablation", ab_csv, "query_tail", "delta_logprob", "negative"),
            ("group_ablation", ab_csv, "answer_address", "delta_logprob", "negative"),
        ]:
            vals = values_for(csv_path, group, field)
            mean, low, high, sd = ci(vals)
            stats_rows.append(
                {
                    "setting": label,
                    "target_tokens": target,
                    "needle_frac": frac,
                    "metric": metric,
                    "group": group,
                    "mean": mean,
                    "ci_low": low,
                    "ci_high": high,
                    "std": sd,
                    "supporting_examples": sum((v > 0) if direction == "positive" else (v < 0) for v in vals),
                    "n": len(vals),
                }
            )

    return {
        "main_results": pd.DataFrame(main_rows),
        "functional_ablation": pd.DataFrame(ablation_rows),
        "functional_patching": pd.DataFrame(patch_rows),
        "single_head_patching": pd.DataFrame(single_rows),
        "attention_activation": pd.DataFrame(attention_activation_rows),
        "statistical_checks": pd.DataFrame(stats_rows),
    }


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#d1d5db",
            "axes.labelcolor": PALETTE["ink"],
            "axes.titlecolor": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "grid.color": "#eef2f7",
            "grid.linewidth": 0.9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "retrieval-head-atlas-report-assets",
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    for ext in ["svg", "png"]:
        metadata = {"Date": None} if ext == "svg" else None
        out_path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(out_path, bbox_inches="tight", facecolor="white", metadata=metadata)
        if ext == "svg":
            out_path.write_text("\n".join(line.rstrip() for line in out_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def add_header(fig: plt.Figure, title: str, subtitle: str, x: float = 0.08) -> None:
    fig.suptitle(title, x=x, y=0.99, ha="left", fontsize=15, weight="bold", color=PALETTE["ink"])
    fig.text(x, 0.945, subtitle, ha="left", va="top", fontsize=9, color=PALETTE["muted"])


def annotate_bars(ax, fmt="{:.2f}", dy=0.02) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * dy
    for patch in ax.patches:
        height = patch.get_height()
        if np.isnan(height):
            continue
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_y() + height
        if height < 0.055:
            continue
        ax.text(x, y + offset, fmt.format(height), ha="center", va="bottom", fontsize=8, color=PALETTE["muted"])


def build_figures(tables: dict[str, pd.DataFrame]) -> None:
    setup_style()
    main = tables["main_results"].copy()
    ablation = tables["functional_ablation"].copy()
    patch = tables["functional_patching"].copy()
    single = tables["single_head_patching"].copy()
    attn = tables["attention_activation"].copy()

    # Figure 1: Role decomposition at 16k, averaged over positions.
    selected_groups = ["answer_address", "non_address_core", "query_tail", "first_token_sink"]
    ab_16 = ablation[(ablation.target_tokens == 16384) & (ablation.group.isin(selected_groups))].copy()
    pa_16 = patch[(patch.target_tokens == 16384) & (patch.group.isin(selected_groups))].copy()
    role = (
        ab_16.groupby(["group", "group_label"], as_index=False)["mean_delta_logprob"].mean()
        .rename(columns={"mean_delta_logprob": "necessity"})
        .merge(pa_16.groupby(["group", "group_label"], as_index=False)["mean_delta_patch"].mean(), on=["group", "group_label"])
    )
    role["necessity"] = -role["necessity"]
    role = role.melt(id_vars=["group", "group_label"], value_vars=["necessity", "mean_delta_patch"], var_name="metric", value_name="value")
    role["metric"] = role["metric"].map({"necessity": "Necessity (-ablation)", "mean_delta_patch": "Sufficiency (clean patch)"})
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    sns.barplot(data=role, x="group_label", y="value", hue="metric", palette=[PALETTE["support"], PALETTE["address"]], ax=ax)
    add_header(fig, "Role decomposition at 16k context", "Support heads are necessary but weak answer donors; address heads patch answer identity.")
    fig.subplots_adjust(top=0.82)
    ax.set_xlabel("")
    ax.set_ylabel("Mean logprob delta magnitude")
    ax.tick_params(axis="x", labelrotation=12)
    ax.legend(title="", loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1)
    annotate_bars(ax)
    savefig(fig, "fig_01_role_decomposition_16k")

    # Figure 2: L22H7 generalization with CI.
    l22 = single[single.group == "address_L22H7"].copy()
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for target, color, marker in [(8192, PALETTE["address"], "o"), (16384, PALETTE["companion"], "s")]:
        subset = l22[l22.target_tokens == target].sort_values("needle_frac")
        ax.errorbar(
            subset["needle_frac"],
            subset["mean_delta_patch"],
            yerr=[subset["mean_delta_patch"] - subset["ci_low"], subset["ci_high"] - subset["mean_delta_patch"]],
            marker=marker,
            markersize=7,
            linewidth=2.5,
            capsize=4,
            label=f"{target // 1024}k context",
            color=color,
        )
    add_header(fig, "L22H7 remains the dominant answer-content donor", "Single-head clean patch effect with 95% normal-approx intervals.")
    fig.subplots_adjust(top=0.82)
    ax.set_xlabel("Needle position fraction")
    ax.set_ylabel("Patch delta")
    ax.set_xticks([0.1, 0.5, 0.9])
    ax.set_ylim(0, max(l22["ci_high"]) * 1.18)
    ax.legend(loc="upper left")
    savefig(fig, "fig_02_l22h7_generalization")

    # Figure 3: single-head decomposition.
    head_groups = ["address_L22H7", "address_L22H10", "address_L21H11", "activation_control_L20H5"]
    head_df = single[single.group.isin(head_groups)].copy()
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    sns.barplot(
        data=head_df,
        x="setting",
        y="mean_delta_patch",
        hue="head_label",
        hue_order=["L22H7", "L22H10", "L21H11", "L20H5 control"],
        palette=[PALETTE["address"], PALETTE["companion"], PALETTE["green"], PALETTE["control"]],
        ax=ax,
    )
    add_header(fig, "Single-head decomposition of answer patching", "L22H7 carries most transplantable answer signal; L22H10 is a smaller companion.")
    fig.subplots_adjust(top=0.82)
    ax.set_xlabel("Context length / needle position")
    ax.set_ylabel("Patch delta")
    ax.tick_params(axis="x", labelrotation=18)
    ax.legend(title="")
    savefig(fig, "fig_03_single_head_decomposition")

    # Figure 4: L22H7 attention/activation/patch alignment.
    align = main[["setting", "l22h7_patch", "l22h7_needle_attention", "l22h7_activation_relative_diff"]].copy()
    align = align.melt("setting", var_name="metric", value_name="value")
    align["metric"] = align["metric"].map(
        {
            "l22h7_patch": "Patch delta",
            "l22h7_needle_attention": "Needle attention",
            "l22h7_activation_relative_diff": "Activation relative diff",
        }
    )
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 7.2), sharex=True)
    metric_colors = {
        "Patch delta": PALETTE["address"],
        "Needle attention": PALETTE["companion"],
        "Activation relative diff": "#0891b2",
    }
    for ax, metric in zip(axes, ["Patch delta", "Needle attention", "Activation relative diff"]):
        subset = align[align.metric == metric]
        sns.lineplot(data=subset, x="setting", y="value", marker="o", linewidth=2.5, color=metric_colors[metric], ax=ax)
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=18)
    add_header(fig, "L22H7 aligns attention, activation, and causal patching", "The same head attends the needle, changes with answer identity, and restores clean-answer probability.")
    fig.subplots_adjust(top=0.86, hspace=0.34)
    savefig(fig, "fig_04_l22h7_attention_activation_alignment")

    # Figure 5: necessity vs sufficiency.
    scatter_groups = ["answer_address", "non_address_core", "query_tail", "first_token_sink"]
    points = []
    for _, row in ablation[ablation.group.isin(scatter_groups)].iterrows():
        patch_row = patch[(patch.setting == row.setting) & (patch.group == row.group)].iloc[0]
        points.append(
            {
                "setting": row.setting,
                "group_label": row.group_label,
                "necessity": -row.mean_delta_logprob,
                "sufficiency": patch_row.mean_delta_patch,
            }
        )
    points = pd.DataFrame(points)
    color_map = {
        "Address heads": PALETTE["address"],
        "Non-address core": PALETTE["support"],
        "Query-tail support": PALETTE["query"],
        "First-token/sink": "#16a34a",
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    sns.scatterplot(
        data=points,
        x="necessity",
        y="sufficiency",
        hue="group_label",
        palette=color_map,
        s=105,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )
    add_header(fig, "Necessity and sufficiency separate cleanly", "Support heads are necessary without being content donors; address heads are content donors.")
    fig.subplots_adjust(top=0.82)
    ax.set_xlabel("Necessity magnitude (-ablation delta)")
    ax.set_ylabel("Sufficiency (clean patch delta)")
    ax.legend(title="")
    savefig(fig, "fig_05_necessity_vs_sufficiency")

    # Figure 6: heatmap of main role metrics across all settings.
    heat = main.set_index("setting")[
        [
            "answer_address_patch",
            "l22h7_patch",
            "non_address_core_patch",
            "non_address_core_ablation",
            "query_tail_ablation",
            "l22h7_needle_attention",
            "l22h7_activation_relative_diff",
        ]
    ].copy()
    heat["non_address_core_ablation"] = -heat["non_address_core_ablation"]
    heat["query_tail_ablation"] = -heat["query_tail_ablation"]
    heat = heat.rename(
        columns={
            "answer_address_patch": "Address patch",
            "l22h7_patch": "L22H7 patch",
            "non_address_core_patch": "Support patch",
            "non_address_core_ablation": "Non-address necessity",
            "query_tail_ablation": "Query-tail necessity",
            "l22h7_needle_attention": "L22H7 needle attn.",
            "l22h7_activation_relative_diff": "L22H7 act. diff",
        }
    )
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    sns.heatmap(heat.T, annot=True, fmt=".2f", cmap="Blues", cbar_kws={"label": "Metric value"}, linewidths=0.8, linecolor="white", ax=ax)
    add_header(fig, "Retrieval-circuit evidence matrix", "Core role-decomposition metrics across context length and needle position.")
    fig.subplots_adjust(top=0.82)
    ax.set_xlabel("Context length / needle position")
    ax.set_ylabel("")
    savefig(fig, "fig_06_evidence_matrix")


def save_tables(tables: dict[str, pd.DataFrame]) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        out = df.copy()
        out.to_csv(TABLE_DIR / f"table_{name}.csv", index=False)
        out.to_markdown(TABLE_DIR / f"table_{name}.md", index=False, floatfmt=".4f")
        out.to_latex(TABLE_DIR / f"table_{name}.tex", index=False, float_format="%.4f")


def html_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    shown = df.head(max_rows).copy()
    return shown.to_html(index=False, float_format=lambda x: f"{x:.4f}", classes="data-table", border=0)


def write_preview(tables: dict[str, pd.DataFrame]) -> None:
    figures = [
        ("fig_01_role_decomposition_16k.svg", "Role decomposition at 16k"),
        ("fig_02_l22h7_generalization.svg", "L22H7 generalization"),
        ("fig_03_single_head_decomposition.svg", "Single-head patch decomposition"),
        ("fig_04_l22h7_attention_activation_alignment.svg", "Attention/activation/patch alignment"),
        ("fig_05_necessity_vs_sufficiency.svg", "Necessity vs sufficiency"),
        ("fig_06_evidence_matrix.svg", "Evidence matrix"),
    ]
    rows = []
    for file, title in figures:
        rows.append(f"<section><h2>{html.escape(title)}</h2><img src='figures/{file}' alt='{html.escape(title)}'></section>")
    main_table = html_table(tables["main_results"])
    stats_table = html_table(tables["statistical_checks"])
    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Retrieval Circuit Report Assets</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 40px; color: #111827; background: #f8fafc; }}
    h1 {{ font-size: 32px; margin-bottom: 8px; }}
    h2 {{ font-size: 22px; margin-top: 34px; }}
    p {{ color: #4b5563; max-width: 900px; line-height: 1.5; }}
    section {{ background: white; padding: 24px; margin: 24px 0; border: 1px solid #e5e7eb; border-radius: 12px; box-shadow: 0 1px 3px rgba(15,23,42,0.06); }}
    img {{ max-width: 100%; height: auto; display: block; }}
    .data-table {{ border-collapse: collapse; font-size: 13px; background: white; }}
    .data-table th, .data-table td {{ padding: 7px 9px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
    .data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
    .data-table th {{ background: #f3f4f6; color: #374151; }}
  </style>
</head>
<body>
  <h1>Retrieval Circuit Report Assets</h1>
  <p>Generated from committed experiment artifacts. The figures are designed for the paper/report narrative: a stable role-decomposed semantic retrieval circuit with a dominant answer-content head and separate support heads.</p>
  {''.join(rows)}
  <section><h2>Main Results Table</h2>{main_table}</section>
  <section><h2>Statistical Checks</h2>{stats_table}</section>
</body>
</html>
"""
    (OUT_DIR / "retrieval_circuit_report_preview.html").write_text(html_doc)


def write_readme() -> None:
    text = """# Retrieval Circuit Report Assets

Generated by:

```bash
.venv/bin/python scripts/build_report_assets.py
```

Open `retrieval_circuit_report_preview.html` for a quick visual review.

## Figures

- `figures/fig_01_role_decomposition_16k.svg` / `.png`
- `figures/fig_02_l22h7_generalization.svg` / `.png`
- `figures/fig_03_single_head_decomposition.svg` / `.png`
- `figures/fig_04_l22h7_attention_activation_alignment.svg` / `.png`
- `figures/fig_05_necessity_vs_sufficiency.svg` / `.png`
- `figures/fig_06_evidence_matrix.svg` / `.png`

## Tables

Each table is emitted as `.csv`, `.md`, and `.tex`:

- `tables/table_main_results.*`
- `tables/table_functional_ablation.*`
- `tables/table_functional_patching.*`
- `tables/table_single_head_patching.*`
- `tables/table_attention_activation.*`
- `tables/table_statistical_checks.*`
"""
    (OUT_DIR / "README.md").write_text(text)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    tables = build_tables()
    save_tables(tables)
    build_figures(tables)
    write_preview(tables)
    write_readme()

    print(f"Wrote report assets to {OUT_DIR}")
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
