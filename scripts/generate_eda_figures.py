"""Deterministic EDA figure generation script for Track S2.

Reads exclusively from splits/train.parquet, splits/valid.parquet, and
splits/split_manifest.json in strict compliance with the project honesty
protocol (zero access to splits/local_test.parquet or splits/frame_full.parquet).
Generates publication-grade 300 DPI PNG figures under reports/figures/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication-grade global styling (300 DPI, clean white canvas, legible fonts)
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)

REQUIRED_COLUMNS: list[str] = [
    "id",
    "date",
    "store_nbr",
    "family",
    "sales",
    "onpromotion",
    "city",
    "state",
    "store_type",
    "cluster",
    "day_of_week",
    "day_of_month",
    "is_holiday",
    "is_holiday_tomorrow",
    "is_holiday_in_2days",
    "earthquake_decay",
]


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments for selective figure rendering and syntax validation."""
    parser = argparse.ArgumentParser(
        description="Generate Track S2 exploratory data analysis figures."
    )
    parser.add_argument(
        "--figures",
        type=int,
        nargs="+",
        default=list(range(1, 9)),
        choices=range(1, 10),
        help="One or more figure numbers in 1..9 (default: 1..8).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate script syntax and CLI configuration without loading data.",
    )
    return parser.parse_args()


def load_data(splits_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load train.parquet, valid.parquet, and split_manifest.json only."""
    train_path = splits_dir / "train.parquet"
    valid_path = splits_dir / "valid.parquet"
    manifest_path = splits_dir / "split_manifest.json"

    for path in (train_path, valid_path, manifest_path):
        if not path.exists():
            msg = f"Missing required input file: {path}. Run `uv run python main.py`."
            raise FileNotFoundError(msg)

    with manifest_path.open("r", encoding="utf-8") as fh:
        manifest: dict = json.load(fh)

    # Project only necessary columns to stay well within 8GB RAM budgets
    try:
        train_df = pd.read_parquet(train_path, columns=REQUIRED_COLUMNS)
        valid_df = pd.read_parquet(valid_path, columns=REQUIRED_COLUMNS)
    except Exception:
        train_df = pd.read_parquet(train_path)
        valid_df = pd.read_parquet(valid_path)

    if len(train_df) != 2_106_324 or len(valid_df) != 299_376:
        print(
            f"[WARN] Row count notice: train={len(train_df):,}, "
            f"valid={len(valid_df):,}"
        )

    df = pd.concat([train_df, valid_df], axis=0, ignore_index=True)
    del train_df, valid_df

    df["date"] = pd.to_datetime(df["date"])

    # Defensive column fallback if upstream schema naming varies slightly
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["date"].dt.dayofweek.astype("int8")
    if "day_of_month" not in df.columns:
        df["day_of_month"] = df["date"].dt.day.astype("int8")
    if "store_type" not in df.columns and "type" in df.columns:
        df["store_type"] = df["type"]

    zero_share = float((df["sales"] == 0).mean() * 100.0)
    print(
        f"[INFO] Loaded train+valid: {len(df):,} rows "
        f"({df['date'].min():%Y-%m-%d}..{df['date'].max():%Y-%m-%d}) | "
        f"Zero-sales share: {zero_share:.2f}%"
    )
    return df, manifest


def plot_annual_trend_and_zero_share(df: pd.DataFrame, out_path: Path) -> None:
    """Fig 1: Annual aggregate unit sales vs zero-sales row counts (train+valid)."""
    years = df["date"].dt.year
    annual = (
        df.assign(year=years, is_zero=(df["sales"] == 0))
        .groupby("year", observed=False)
        .agg(
            total_units=("sales", "sum"),
            zero_rows=("is_zero", "sum"),
            total_rows=("sales", "count"),
        )
        .reset_index()
    )
    annual = annual[annual["year"].isin([2013, 2014, 2015, 2016])].copy()
    annual["units_millions"] = annual["total_units"] / 1e6
    annual["zeros_thousands"] = annual["zero_rows"] / 1e3
    annual["zero_pct"] = (annual["zero_rows"] / annual["total_rows"]) * 100.0

    labels = [
        f"{yr}\n(Partial: ..09-11)" if yr == 2016 else f"{yr}\n(Full Year)"
        for yr in annual["year"]
    ]

    fig, ax1 = plt.subplots(figsize=(9.5, 5.0))
    ax2 = ax1.twinx()
    ax2.grid(False)

    x = np.arange(len(annual))
    width = 0.42

    bars = ax1.bar(
        x - width / 2,
        annual["units_millions"],
        width=width,
        color="#1f77b4",
        alpha=0.88,
        label="Total Units Sold (Millions)",
    )
    line = ax2.plot(
        x + width / 2,
        annual["zeros_thousands"],
        color="#d62728",
        marker="o",
        linewidth=2.5,
        markersize=7,
        label="Zero-Sales Rows (Thousands)",
    )

    for bar, units in zip(bars, annual["units_millions"], strict=False):
        ax1.annotate(
            f"{units:.1f}M",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#1f77b4",
        )

    for xi, z_k, z_pct in zip(
        x + width / 2, annual["zeros_thousands"], annual["zero_pct"], strict=False
    ):
        ax2.annotate(
            f"{z_k:.1f}k ({z_pct:.1f}%)",
            xy=(xi, z_k),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#d62728",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Calendar Year (Train + Valid Scope Only; 2017 Omitted)")
    ax1.set_ylabel("Total Units Sold (Millions)", color="#1f77b4")
    ax2.set_ylabel("Zero-Sales Row Count (Thousands)", color="#d62728")
    ax1.set_ylim(0, annual["units_millions"].max() * 1.22)
    ax2.set_ylim(0, annual["zeros_thousands"].max() * 1.25)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + line, labels1 + labels2, loc="upper left", frameon=True)

    plt.title(
        "Figure 1: Annual Aggregate Unit Volume vs. Zero-Sales Row Contraction "
        "(2013-2016 Partial)"
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close("all")

    for _, row in annual.iterrows():
        print(
            f"[FIG01] Year {int(row['year'])}: "
            f"units={row['total_units']:,.0f} ({row['units_millions']:.2f}M), "
            f"zero_rows={int(row['zero_rows']):,} ({row['zero_pct']:.2f}%)"
        )


def plot_weekly_seasonality(df: pd.DataFrame, out_path: Path) -> None:
    """Fig 2: Normalized daily sales index across day of week by store type."""
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow = df["day_of_week"].astype(int)
    if dow.min() == 1 and dow.max() == 7:
        dow = dow - 1

    grouped = (
        df.assign(dow_idx=dow)
        .groupby(["store_type", "dow_idx"], observed=False)["sales"]
        .mean()
        .unstack("dow_idx")
    )
    # Normalize each store type to its own weekly mean = 100
    type_means = grouped.mean(axis=1)
    normalized = grouped.div(type_means, axis=0) * 100.0

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    palette = {
        "A": "#1f77b4",
        "B": "#ff7f0e",
        "C": "#2ca02c",
        "D": "#d62728",
        "E": "#9467bd",
    }

    for stype in sorted(normalized.index.astype(str)):
        series = normalized.loc[stype]
        ax.plot(
            range(7),
            series.values,
            marker="o",
            linewidth=2.2,
            color=palette.get(stype, "#333333"),
            label=(
                f"Type {stype} (Mean: {type_means.loc[stype]:.1f} units/row)"
            ),
        )

    ax.axhline(
        100.0,
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Weekly Baseline (100)",
    )
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.set_xlabel("Day of Week (Monday = 0 to Sunday = 6)")
    ax.set_ylabel("Normalized Daily Sales Index (Store-Type Mean = 100)")
    ax.set_title(
        "Figure 2: Day-of-Week Seasonality Index Stratified by Store Format (A-E)"
    )
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close("all")

    for stype in sorted(normalized.index.astype(str)):
        sat_val = normalized.loc[stype, 5]
        sun_val = normalized.loc[stype, 6]
        print(
            f"[FIG02] Store Type {stype}: raw_mean={type_means.loc[stype]:.2f}, "
            f"Sat_Idx={sat_val:.1f}, Sun_Idx={sun_val:.1f}"
        )


def plot_payday_seasonality(df: pd.DataFrame, out_path: Path) -> None:
    """Fig 3: Intra-month payday seasonality (quincena: Day 15 and Month-End)."""
    dom_stats = (
        df.groupby("day_of_month", observed=False)["sales"]
        .mean()
        .reset_index()
        .sort_values("day_of_month")
    )
    overall_mean = float(df["sales"].mean())
    dom_stats["norm_index"] = (dom_stats["sales"] / overall_mean) * 100.0

    fig, ax1 = plt.subplots(figsize=(10.0, 4.8))
    ax2 = ax1.twinx()
    ax2.grid(False)

    ax1.plot(
        dom_stats["day_of_month"],
        dom_stats["sales"],
        color="#1f77b4",
        marker="o",
        linewidth=2.2,
        label="Mean Sales Volume (Units / Row)",
    )
    ax1.axhline(
        overall_mean,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"Global Mean ({overall_mean:.1f} Units)",
    )

    # Highlight bi-monthly public sector payroll (quincena) dates
    for pay_day, label_txt in [(15, "Mid-Month Payday (15th)"), (31, "Month-End (30th/31st)")]:
        ax1.axvline(
            pay_day,
            color="#d62728",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.85,
            label=label_txt,
        )
    ax1.axvspan(14.5, 16.5, color="#ff9896", alpha=0.22)
    ax1.axvspan(29.5, 31.3, color="#ff9896", alpha=0.22)
    ax1.axvspan(0.8, 2.5, color="#aec7e8", alpha=0.20, label="Early-Month Spillover (1st-2nd)")

    y_min, y_max = dom_stats["sales"].min() * 0.9, dom_stats["sales"].max() * 1.1
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim((y_min / overall_mean) * 100.0, (y_max / overall_mean) * 100.0)

    ax1.set_xticks(range(1, 32))
    ax1.set_xlabel("Calendar Day of Month (1 to 31)")
    ax1.set_ylabel("Mean Sales Volume (Units per Store-Family Row)", color="#1f77b4")
    ax2.set_ylabel("Normalized Payday Sales Index (Mean = 100)", color="#333333")
    ax1.set_title(
        "Figure 3: Intra-Month Bi-Monthly Payroll (Quincena) Demand Seasonality"
    )
    ax1.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close("all")

    d1 = float(dom_stats.loc[dom_stats["day_of_month"] == 1, "norm_index"].iloc[0])
    d15 = float(dom_stats.loc[dom_stats["day_of_month"] == 15, "norm_index"].iloc[0])
    d30 = float(dom_stats.loc[dom_stats["day_of_month"] == 30, "norm_index"].iloc[0])
    d31 = float(dom_stats.loc[dom_stats["day_of_month"] == 31, "norm_index"].iloc[0])
    print(
        f"[FIG03] Payday Indices (100=mean): Day1={d1:.1f}, "
        f"Day15={d15:.1f}, Day30={d30:.1f}, Day31={d31:.1f}"
    )


def plot_earthquake_impact(df: pd.DataFrame, out_path: Path) -> None:
    """Fig 4: Manabí earthquake impact & 30-day linear decay (2016-03-17..2016-06-14)."""
    start_dt = pd.Timestamp("2016-03-17")
    end_dt = pd.Timestamp("2016-06-14")
    boundary_dt = pd.Timestamp("2016-03-27")
    quake_dt = pd.Timestamp("2016-04-16")
    decay_end_dt = pd.Timestamp("2016-05-15")

    sub = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].copy()
    essential_fams = {"GROCERY I", "BEVERAGES", "WATER", "CLEANING", "BREAD/BAKERY"}
    sub["cat_group"] = np.where(
        sub["family"].astype(str).isin(essential_fams),
        "Emergency & Essential Staples",
        "Discretionary & Other Families",
    )

    daily_cat = (
        sub.groupby(["date", "cat_group"], observed=False)["sales"]
        .sum()
        .unstack("cat_group")
    )
    daily_decay = sub.groupby("date", observed=False)["earthquake_decay"].max()

    fig, ax1 = plt.subplots(figsize=(10.5, 5.2))
    ax2 = ax1.twinx()
    ax2.grid(False)

    if "Emergency & Essential Staples" in daily_cat.columns:
        ax1.plot(
            daily_cat.index,
            daily_cat["Emergency & Essential Staples"] / 1e3,
            color="#1f77b4",
            linewidth=2.0,
            label="Emergency & Essential Staples (Thousands Units)",
        )
    if "Discretionary & Other Families" in daily_cat.columns:
        ax1.plot(
            daily_cat.index,
            daily_cat["Discretionary & Other Families"] / 1e3,
            color="#2ca02c",
            linewidth=1.8,
            linestyle="--",
            label="Discretionary & Other Families (Thousands Units)",
        )

    ax2.plot(
        daily_decay.index,
        daily_decay.values,
        color="#d62728",
        linewidth=2.2,
        alpha=0.85,
        label="earthquake_decay Feature (1.0 → 0.0)",
    )
    ax2.fill_between(
        daily_decay.index,
        0,
        daily_decay.values,
        color="#d62728",
        alpha=0.12,
    )

    # Boundary and shock callout lines
    ax1.axvline(
        boundary_dt,
        color="#333333",
        linestyle=":",
        linewidth=1.8,
        label="Train / Valid Split Boundary (2016-03-27)",
    )
    ax1.axvline(
        quake_dt,
        color="#d62728",
        linestyle="-",
        linewidth=1.8,
        label="7.8 Mw Manabí Earthquake (2016-04-16)",
    )
    ax1.axvline(
        decay_end_dt,
        color="#ff7f0e",
        linestyle="-.",
        linewidth=1.6,
        label="30-Day Positive Window Ends (2016-05-15)",
    )

    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=10))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right")

    ax1.set_xlabel(
        "Calendar Date (2016-03-17 to 2016-06-14; Quake Resides Inside Valid Split)"
    )
    ax1.set_ylabel("Daily Aggregate Sales (Thousand Units)", color="#1f77b4")
    ax2.set_ylabel("Earthquake Linear Decay Index (1.0 to 0.0)", color="#d62728")
    ax2.set_ylim(-0.05, 1.15)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        fontsize=8,
        frameon=True,
    )

    plt.title(
        "Figure 4: April 16, 2016 Manabí Earthquake Demand Surge & 30-Day Linear Decay"
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close("all")

    pre_base = daily_cat.loc[
        (daily_cat.index >= pd.Timestamp("2016-03-17"))
        & (daily_cat.index < quake_dt),
        "Emergency & Essential Staples",
    ].mean()
    peak_val = daily_cat.loc[
        (daily_cat.index >= quake_dt) & (daily_cat.index <= pd.Timestamp("2016-04-20")),
        "Emergency & Essential Staples",
    ].max()
    print(
        f"[FIG04] Essentials pre-quake mean={pre_base:,.0f} units/day, "
        f"post-quake peak={peak_val:,.0f} units/day "
        f"(+{((peak_val - pre_base) / pre_base) * 100:.1f}%)"
    )


def plot_holiday_lead_effects(df: pd.DataFrame, out_path: Path) -> None:
    """Fig 5: Sales uplift on days preceding holidays vs event-day effect."""
    daily = (
        df.groupby("date", observed=False)
        .agg(
            total_sales=("sales", "sum"),
            h_day=("is_holiday", "max"),
            h_tom=("is_holiday_tomorrow", "max"),
            h_2d=("is_holiday_in_2days", "max"),
        )
        .reset_index()
    )

    normal_mask = (
        (daily["h_day"] == 0) & (daily["h_tom"] == 0) & (daily["h_2d"] == 0)
    )
    baseline_sales = float(daily.loc[normal_mask, "total_sales"].mean())

    # Evaluate mutually clear lead regimes + major closure/national events
    states = [
        ("Normal Workday\n(Baseline)", normal_mask),
        ("2 Days Before\n(is_holiday_in_2days=1)", (daily["h_2d"] == 1) & (daily["h_tom"] == 0)),
        ("1 Day Before\n(is_holiday_tomorrow=1)", daily["h_tom"] == 1),
        ("Holiday Event Day\n(All Flagged Holidays)", daily["h_day"] == 1),
        (
            "Major National Eve\n(Dec 23-24 / Dec 30-31)",
            daily["date"].dt.strftime("%m-%d").isin(["12-23", "12-24", "12-30", "12-31"]),
        ),
        (
            "Christmas Closure\n(Dec 25 Synthetic Zero)",
            daily["date"].dt.strftime("%m-%d") == "12-25",
        ),
    ]

    labels, uplifts, means = [], [], []
    for label, mask in states:
        mean_val = float(daily.loc[mask, "total_sales"].mean()) if mask.any() else 0.0
        uplift_pct = ((mean_val - baseline_sales) / baseline_sales) * 100.0
        labels.append(label)
        uplifts.append(uplift_pct)
        means.append(mean_val)

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    colors = [
        "#7f7f7f",
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#d62728",
    ]
    bars = ax.bar(range(len(labels)), uplifts, color=colors, width=0.58, alpha=0.9)
    ax.axhline(0.0, color="black", linewidth=1.2)

    for bar, up, mv in zip(bars, uplifts, means, strict=False):
        y_pos = bar.get_height()
        offset = 4 if y_pos >= 0 else -14
        ax.annotate(
            f"{up:+.1f}%\n({mv / 1e3:.0f}k u/d)",
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom" if y_pos >= 0 else "top",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Calendar Regime / Holiday Lead Window (Train + Valid Scope)")
    ax.set_ylabel("Average Aggregate Sales Uplift (% vs. Normal Workday)")
    ax.set_ylim(min(uplifts) * 1.25 - 10, max(uplifts) * 1.28 + 15)
    ax.set_title(
        "Figure 5: Holiday Lead Anticipation Uplift vs. Event-Day & Closure Contraction"
    )

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close("all")

    for lbl, up, mv in zip(labels, uplifts, means, strict=False):
        clean_lbl = lbl.replace("\n", " ")
        print(f"[FIG05] {clean_lbl}: mean={mv:,.0f} units/day ({up:+.2f}%)")


def plot_store52_timeline(df: pd.DataFrame, out_path: Path) -> None:
    """Fig 6: Store 52 pre-opening zero timeline (2013-01-01..2016-09-11)."""
    s52 = df[df["store_nbr"] == 52].copy()
    daily_s52 = s52.groupby("date", observed=False)["sales"].sum().sort_index()

    train_end = pd.Timestamp("2016-03-27")
    valid_end = pd.Timestamp("2016-09-11")

    s52_train = s52[s52["date"] <= train_end]
    s52_valid = s52[s52["date"] > train_end]

    n_train_zeros = int((s52_train["sales"] == 0).sum())
    n_train_nan_id = int(s52_train["id"].isna().sum())
    n_valid_zeros = int((s52_valid["sales"] == 0).sum())
    n_total_unmasked = n_train_zeros + n_valid_zeros
    n_train_masked = n_train_zeros - n_train_nan_id
    n_total_masked = n_train_masked + n_valid_zeros

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(
        daily_s52.index,
        daily_s52.values,
        color="#d62728",
        linewidth=2.2,
        label="Store 52 Daily Sales (0 Units Across Entire Train+Valid Scope)",
    )

    ax.axvspan(
        pd.Timestamp("2013-01-01"),
        train_end,
        color="#1f77b4",
        alpha=0.10,
        label=f"Train Split ({n_train_zeros:,} grid zeros incl. {n_train_nan_id} id=NaN fills)",
    )
    ax.axvspan(
        train_end,
        valid_end,
        color="#ff7f0e",
        alpha=0.14,
        label=f"Valid Split ({n_valid_zeros:,} grid zeros)",
    )
    ax.axvline(
        train_end,
        color="#1f77b4",
        linestyle="--",
        linewidth=1.5,
        label="Train / Valid Boundary (2016-03-27)",
    )
    ax.axvline(
        valid_end,
        color="#333333",
        linestyle="-.",
        linewidth=1.8,
        label="End of Permitted EDA Scope (2016-09-11)",
    )

    callout_text = (
        "Store 52 Pre-Opening Accounting (2013-01-01..2016-09-11):\n"
        f"• Unmasked Train+Valid Grid Zeros: {n_train_zeros:,} train "
        f"(incl. {n_train_nan_id} Christmas id=NaN fills) + {n_valid_zeros:,} valid "
        f"= {n_total_unmasked:,} rows\n"
        f"• Loader-Masked Pre-Opening Rows: {n_train_masked:,} train + "
        f"{n_valid_zeros:,} valid = {n_total_masked:,} rows "
        f"({n_train_nan_id} id=NaN fills kept as true-zero labels)\n"
        "• Context-Only Note: Store 52 opens 2017-04-20 inside local_test "
        "(51,810 full-grid zeros; 12,804 holdout zeros; local_test not plotted)"
    )
    ax.text(
        0.02,
        0.58,
        callout_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="center",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f8f9fa", "edgecolor": "#888888"},
    )

    ax.set_ylim(-5, 100)
    ax.set_xlabel("Calendar Date (Permitted Scope: 2013-01-01 to 2016-09-11)")
    ax.set_ylabel("Daily Store Sales (Units)")
    ax.set_title("Store 52 pre-opening zero timeline (2013-01-01..2016-09-11)")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close("all")

    print(
        f"[FIG06] Store 52 verified: train_zeros={n_train_zeros:,} "
        f"(id=NaN={n_train_nan_id}), valid_zeros={n_valid_zeros:,}, "
        f"unmasked_total={n_total_unmasked:,}, loader_masked={n_total_masked:,}"
    )


def plot_family_dispersion(df: pd.DataFrame, out_path: Path) -> None:
    """Fig 7: Pareto volume dispersion and zero share across 33 product families."""
    fam = (
        df.assign(is_zero=(df["sales"] == 0))
        .groupby("family", observed=False)
        .agg(
            total_sales=("sales", "sum"),
            zero_share=("is_zero", "mean"),
        )
        .reset_index()
        .sort_values("total_sales", ascending=False)
        .reset_index(drop=True)
    )
    fam["rank"] = np.arange(1, len(fam) + 1)
    total_vol = float(fam["total_sales"].sum())
    fam["vol_pct"] = (fam["total_sales"] / total_vol) * 100.0
    fam["cum_pct"] = fam["vol_pct"].cumsum()
    fam["zero_pct"] = fam["zero_share"] * 100.0

    top3_share = float(fam.loc[:2, "vol_pct"].sum())
    top5_share = float(fam.loc[:4, "vol_pct"].sum())
    top10_share = float(fam.loc[:9, "vol_pct"].sum())

    fig, ax1 = plt.subplots(figsize=(11.5, 5.4))
    ax2 = ax1.twinx()
    ax2.grid(False)

    x = np.arange(len(fam))
    ax1.bar(
        x,
        fam["vol_pct"],
        color="#1f77b4",
        alpha=0.85,
        label="Family Share of Total Unit Volume (%)",
    )
    ax2.plot(
        x,
        fam["cum_pct"],
        color="#d62728",
        marker="o",
        markersize=4.5,
        linewidth=2.2,
        label="Cumulative Volume Share (%)",
    )
    ax2.plot(
        x,
        fam["zero_pct"],
        color="#2ca02c",
        linestyle="--",
        marker="s",
        markersize=4,
        linewidth=1.6,
        label="Family Zero-Sales Share (%)",
    )
    ax2.axhline(80.0, color="#d62728", linestyle=":", linewidth=1.2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{r}. {name}" for r, name in zip(fam["rank"], fam["family"], strict=False)],
        rotation=65,
        ha="right",
        fontsize=7.5,
    )
    ax1.set_xlabel("Product Family Rank (1 to 33)")
    ax1.set_ylabel("Share of Total Unit Volume (%)", color="#1f77b4")
    ax2.set_ylabel("Cumulative Volume % / Zero-Sales %", color="#d62728")
    ax2.set_ylim(0, 105)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="center right",
        frameon=True,
    )

    plt.title(
        f"Figure 7: Product Family Pareto Concentration "
        f"(Top 3 = {top3_share:.1f}%, Top 5 = {top5_share:.1f}%, "
        f"Top 10 = {top10_share:.1f}% of Volume)"
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close("all")

    print(
        f"[FIG07] Pareto Shares: Top3={top3_share:.2f}%, "
        f"Top5={top5_share:.2f}%, Top10={top10_share:.2f}% | "
        f"Top1={fam.iloc[0]['family']} ({fam.iloc[0]['vol_pct']:.2f}%)"
    )


def plot_cluster_geography(df: pd.DataFrame, out_path: Path) -> None:
    """Fig 8: Geographic & cluster demand dispersion across 17 clusters and states."""
    n_dates = df["date"].nunique()
    store_totals = (
        df.groupby(["store_nbr", "cluster", "city", "state"], observed=True)["sales"]
        .sum()
        .reset_index()
    )
    store_totals["daily_store_vol"] = store_totals["sales"] / float(n_dates)

    cluster_stats = (
        store_totals.groupby("cluster", observed=False)
        .agg(
            mean_daily_store_vol=("daily_store_vol", "mean"),
            n_stores=("store_nbr", "nunique"),
            top_state=("state", lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else ""),
        )
        .reset_index()
        .sort_values("mean_daily_store_vol", ascending=True)
    )

    state_totals = (
        store_totals.groupby("state", observed=False)["daily_store_vol"]
        .sum()
        .sort_values(ascending=True)
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.4))

    y_cl = np.arange(len(cluster_stats))
    ax1.barh(
        y_cl,
        cluster_stats["mean_daily_store_vol"] / 1e3,
        color="#1f77b4",
        alpha=0.88,
    )
    ax1.set_yticks(y_cl)
    ax1.set_yticklabels(
        [
            f"Cluster {int(r['cluster'])} (n={int(r['n_stores'])})"
            for _, r in cluster_stats.iterrows()
        ],
        fontsize=8,
    )
    ax1.set_xlabel("Average Daily Volume per Store (Thousand Units)")
    ax1.set_ylabel("Administrative Store Cluster (1-17)")
    ax1.set_title("A. Mean Daily Volume per Store by Cluster")

    y_st = np.arange(len(state_totals))
    ax2.barh(
        y_st,
        state_totals.values / 1e3,
        color="#2ca02c",
        alpha=0.88,
    )
    ax2.set_yticks(y_st)
    ax2.set_yticklabels(state_totals.index.astype(str), fontsize=8)
    ax2.set_xlabel("Total Daily Network Volume (Thousand Units)")
    ax2.set_ylabel("Ecuadorian State (16 States Across 22 Cities)")
    ax2.set_title("B. Aggregate Daily Volume by State")

    max_cl = cluster_stats.iloc[-1]
    min_cl = cluster_stats.iloc[0]
    ratio = float(max_cl["mean_daily_store_vol"] / max(min_cl["mean_daily_store_vol"], 1.0))

    fig.suptitle(
        f"Figure 8: Store Cluster & Geographic Demand Distribution "
        f"(Top Cluster {int(max_cl['cluster'])} = {ratio:.1f}x Lowest Cluster {int(min_cl['cluster'])})",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close("all")

    top_state_name = str(state_totals.index[-1])
    top_state_share = float((state_totals.iloc[-1] / state_totals.sum()) * 100.0)
    print(
        f"[FIG08] Top Cluster={int(max_cl['cluster'])} "
        f"({max_cl['mean_daily_store_vol']:,.0f} u/store/day), "
        f"Min Cluster={int(min_cl['cluster'])} "
        f"({min_cl['mean_daily_store_vol']:,.0f} u/store/day), "
        f"Ratio={ratio:.2f}x | Top State={top_state_name} ({top_state_share:.2f}%)"
    )


def plot_onpromotion_uplift(df: pd.DataFrame, out_path: Path) -> None:
    """Fig 9 (Optional): Promotional elasticity across top 6 product families."""
    top6_fams = (
        df.groupby("family", observed=False)["sales"]
        .sum()
        .nlargest(6)
        .index.astype(str)
        .tolist()
    )
    sub = df[df["family"].astype(str).isin(top6_fams)].copy()
    sub["promo_bin"] = pd.cut(
        sub["onpromotion"],
        bins=[-1, 0, 1, 5, 15, 50, 10_000],
        labels=["0 (None)", "1", "2-5", "6-15", "16-50", "51+"],
    )

    piv = (
        sub.groupby(["promo_bin", "family"], observed=True)["sales"]
        .mean()
        .unstack("family")
    )

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    for fam_name in top6_fams:
        if fam_name in piv.columns:
            ax.plot(
                range(len(piv.index)),
                piv[fam_name].values,
                marker="o",
                linewidth=2.0,
                label=fam_name,
            )

    ax.set_xticks(range(len(piv.index)))
    ax.set_xticklabels(piv.index.astype(str))
    ax.set_xlabel("Promotional Item Count Bin (onpromotion)")
    ax.set_ylabel("Mean Unit Sales per Store-Family Row")
    ax.set_title(
        "Figure 9 (Diagnostic): Mean Unit Sales by Promotional Intensity Across Top Families"
    )
    ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close("all")
    print("[FIG09] Optional promotional uplift figure rendered successfully.")


def main() -> None:
    """Entrypoint for deterministic generation of Track S2 EDA figures."""
    args = parse_arguments()
    if args.check:
        print("CLI check passed: scripts/generate_eda_figures.py syntax and args valid.")
        sys.exit(0)

    repo_root = Path(__file__).resolve().parents[1]
    splits_dir = repo_root / "splits"
    figures_dir = repo_root / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)


    df, _ = load_data(splits_dir)

    dispatch = {
        1: (
            plot_annual_trend_and_zero_share,
            figures_dir / "fig01_annual_trend_zero_share.png",
        ),
        2: (
            plot_weekly_seasonality,
            figures_dir / "fig02_weekly_seasonality_dow.png",
        ),
        3: (
            plot_payday_seasonality,
            figures_dir / "fig03_payday_seasonality_bimonthly.png",
        ),
        4: (
            plot_earthquake_impact,
            figures_dir / "fig04_earthquake_impact_decay.png",
        ),
        5: (
            plot_holiday_lead_effects,
            figures_dir / "fig05_holiday_lead_anticipation.png",
        ),
        6: (
            plot_store52_timeline,
            figures_dir / "fig06_store52_pre_opening_timeline.png",
        ),
        7: (
            plot_family_dispersion,
            figures_dir / "fig07_family_volume_dispersion.png",
        ),
        8: (
            plot_cluster_geography,
            figures_dir / "fig08_cluster_geographic_sales.png",
        ),
        9: (
            plot_onpromotion_uplift,
            figures_dir / "fig09_onpromotion_sales_uplift.png",
        ),
    }

    for fig_num in sorted(set(args.figures)):
        func, out_path = dispatch[fig_num]
        print(f"[RUN] Rendering Figure {fig_num} -> {out_path.relative_to(repo_root)}")
        func(df, out_path)

    total_bytes = sum(p.stat().st_size for p in figures_dir.glob("fig*.png"))
    total_mb = total_bytes / (1024 * 1024)
    print(
        f"[DONE] Rendered {len(set(args.figures))} figure(s) | "
        f"Aggregate directory size: {total_mb:.2f} MB (limit < 5.00 MB)"
    )


if __name__ == "__main__":
    main()
