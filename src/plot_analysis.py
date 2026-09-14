#!/usr/bin/env python3
"""Набор отдельных графиков со справкой в углу (русский язык)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"

# Основные дистанции для сравнения (эстафету отдельно не мешаем в гипотезу)
COMPARE = ["21.1 км Ж", "21.1 км М", "10 км", "5 км"]
GROUP_ORDER = ["21.1 км", "10 км", "5 км"]

# Понятные подписи категорий для осей (без жаргона протокола)
RACE_LABEL = {
    "21.1 км Ж": "Полумарафон,\nженщины",
    "21.1 км М": "Полумарафон,\nмужчины",
    "10 км": "10 км",
    "5 км": "5 км",
    "21.1 км": "Полумарафон",
    "эстафета": "Эстафета",
}

COLORS = {
    "21.1 км Ж": "#0B6E4F",
    "21.1 км М": "#1B4332",
    "21.1 км": "#1B4332",
    "10 км": "#C44900",
    "5 км": "#2C6EAF",
    "эстафета": "#6B5B95",
}

BG = "#F7F4EF"
INK = "#1C1917"
MUTED = "#57534E"
PANEL = "#FFFCF7"
ACCENT = "#9A3412"


def setup_style() -> None:
    # DejaVu хорошо покрывает кириллицу и служебные символы в подписях
    family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BG,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": "#D6D3D1",
            "grid.color": "#E7E5E4",
            "font.family": family,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "medium",
            "axes.labelsize": 11,
            "figure.dpi": 140,
            "savefig.dpi": 180,
        }
    )


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA / "results.csv")
    df = df[df["finished"]].copy()
    for col in (
        "norm_minmax",
        "norm_p1p99",
        "norm_vs_winner",
        "norm_vs_median",
        "percentile",
        "riegel_10k_s",
        "pace_s_per_km",
        "finish_s",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(round(seconds % 60))
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def add_note(ax, text: str, loc: str = "lower right") -> None:
    """Развёрнутая справка в углу графика."""
    anchors = {
        "lower right": (0.98, 0.03, "right", "bottom"),
        "lower left": (0.02, 0.03, "left", "bottom"),
        "upper right": (0.98, 0.97, "right", "top"),
        "upper left": (0.02, 0.97, "left", "top"),
    }
    x, y, ha, va = anchors[loc]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=8.2,
        color=MUTED,
        linespacing=1.35,
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor="#FFF7ED",
            edgecolor="#FED7AA",
            alpha=0.96,
        ),
        zorder=10,
    )


def save(fig: plt.Figure, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("→", path)
    return path


def plot_pace_scatter(df: pd.DataFrame) -> None:
    """Главный интуитивный график: каждый бегун — точка, темп в мин/км."""
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(12.5, 7.6))

    turtle_xy = None  # самая медленная точка для стрелки «черепаха»
    for i, race in enumerate(COMPARE):
        g = df[df["race"] == race].copy()
        pace = (g["pace_s_per_km"] / 60.0).astype(float)
        y_hi = pace.quantile(0.995)
        shown = pace[pace <= y_hi * 1.15]
        x = i + rng.normal(0, 0.12, size=len(shown))
        ax.scatter(
            x,
            shown,
            s=16,
            alpha=0.38,
            color=COLORS[race],
            edgecolors="none",
            zorder=2,
            label=RACE_LABEL[race].replace("\n", " "),
        )
        outliers = pace[pace > y_hi * 1.15]
        if len(outliers):
            xo = i + rng.normal(0, 0.08, size=len(outliers))
            ax.scatter(
                xo,
                outliers,
                s=42,
                alpha=0.9,
                color=COLORS[race],
                edgecolors="#1C1917",
                linewidths=0.45,
                zorder=4,
                marker="D",
            )
            # запомним самого медленного среди выбросов
            j = int(np.argmax(outliers.to_numpy()))
            cand = (float(xo[j]), float(outliers.iloc[j]), race)
            if turtle_xy is None or cand[1] > turtle_xy[1]:
                turtle_xy = cand

        med = float(pace.median())
        win = float(pace.min())
        ax.hlines(med, i - 0.28, i + 0.28, colors=ACCENT, lw=2.2, zorder=5)
        ax.scatter([i], [win], s=110, marker="*", color="#CA8A04", edgecolors="#1C1917", linewidths=0.5, zorder=6)
        # стрелка к медиане — только у крайних колонок, чтобы не зашумить
        if i in (0, 3):
            ax.annotate(
                f"медиана {med:.1f}",
                xy=(i + 0.28, med),
                xytext=(i + 0.45, med + (0.35 if i == 0 else -0.15)),
                fontsize=8,
                color=ACCENT,
                va="center",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1),
            )
        else:
            ax.text(i + 0.32, med, f"{med:.1f}", fontsize=7.5, color=ACCENT, va="center")

        # стрелка к победителю в каждой категории
        dy = 0.55 if i % 2 == 0 else 0.9
        ax.annotate(
            f"лидер {win:.1f}",
            xy=(i, win),
            xytext=(i - 0.15 if i else i + 0.15, win + dy),
            fontsize=7.5,
            color="#854D0E",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#854D0E", lw=1),
        )

    if turtle_xy is not None:
        tx, ty, _race = turtle_xy
        ax.annotate(
            f"выброс {ty:.1f}\n(«черепаха»)",
            xy=(tx, ty),
            xytext=(tx - 0.85, ty - 0.35),
            fontsize=8,
            color="#7F1D1D",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#7F1D1D", lw=1.2),
            zorder=7,
        )

    ax.set_xticks(range(len(COMPARE)))
    ax.set_xticklabels([RACE_LABEL[r] for r in COMPARE])
    ax.set_ylabel("Темп бегуна, минут на километр\n(меньше = быстрее; единая шкала для всех дистанций)")
    ax.set_xlabel("Категория забега")
    ax.set_title("Каждый финишёр — одна точка: темп (мин/км) по четырём категориям")
    ax.grid(True, axis="y", alpha=0.45)
    all_pace = df[df["race"].isin(COMPARE)]["pace_s_per_km"] / 60.0
    ax.set_ylim(all_pace.min() * 0.90, min(float(all_pace.quantile(0.995)) * 1.08, float(all_pace.max()) * 1.02))
    ax.invert_yaxis()  # быстрые сверху

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=COLORS[r],
            markersize=8,
            label=RACE_LABEL[r].replace("\n", " "),
        )
        for r in COMPARE
    ]
    handles += [
        plt.Line2D([0], [0], color=ACCENT, lw=2.2, label="медиана категории"),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#CA8A04",
            markeredgecolor="#1C1917",
            markersize=12,
            label="победитель (лидер)",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="#57534E",
            markeredgecolor="#1C1917",
            markersize=8,
            label="сильный выброс",
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower left", fontsize=8.5)

    add_note(
        ax,
        "СПРАВКА\n"
        "Ось Y — реальный темп в мин/км (не нормировка 0…1).\n"
        "Быстрые вверху, медленные внизу. Четыре облака\n"
        "можно сравнивать глазами на одной шкале.\n"
        "Звезда + стрелка — лидер; черта — типичный темп;\n"
        "ромб / «черепаха» — аномально медленный, он НЕ\n"
        "сжимает шкалу остальных точек.",
        "lower right",
    )
    save(fig, "00_pace_scatter")


def plot_speed_scatter(df: pd.DataFrame) -> None:
    """Тот же scatter, но в км/ч — кому привычнее скорость."""
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(12, 7.2))

    for i, race in enumerate(COMPARE):
        g = df[df["race"] == race]
        speed = 3600.0 / g["pace_s_per_km"].astype(float)  # км/ч
        y_lo = speed.quantile(0.005)
        shown = speed[speed >= y_lo * 0.85]
        x = i + rng.normal(0, 0.12, size=len(shown))
        ax.scatter(x, shown, s=14, alpha=0.35, color=COLORS[race], edgecolors="none", zorder=2)
        outliers = speed[speed < y_lo * 0.85]
        if len(outliers):
            xo = i + rng.normal(0, 0.08, size=len(outliers))
            ax.scatter(
                xo,
                outliers,
                s=36,
                alpha=0.85,
                color=COLORS[race],
                edgecolors="#1C1917",
                linewidths=0.4,
                zorder=4,
                marker="D",
            )
        med = float(speed.median())
        win = float(speed.max())
        ax.hlines(med, i - 0.28, i + 0.28, colors=ACCENT, lw=2.2, zorder=5)
        ax.scatter([i], [win], s=90, marker="*", color="#CA8A04", edgecolors="#1C1917", linewidths=0.5, zorder=6)
        ax.annotate(
            f"медиана {med:.1f}",
            xy=(i + 0.28, med),
            xytext=(i + 0.42, med),
            fontsize=8,
            color=ACCENT,
            va="center",
            arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1),
        )

    ax.set_xticks(range(len(COMPARE)))
    ax.set_xticklabels([RACE_LABEL[r] for r in COMPARE])
    ax.set_ylabel("Средняя скорость бегуна, км/ч\n(больше = быстрее)")
    ax.set_xlabel("Категория забега")
    ax.set_title("Каждый финишёр — одна точка: скорость (км/ч) по четырём категориям")
    ax.grid(True, axis="y", alpha=0.45)
    all_spd = 3600.0 / df[df["race"].isin(COMPARE)]["pace_s_per_km"]
    ax.set_ylim(max(0, all_spd.quantile(0.005) * 0.9), all_spd.max() * 1.05)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[r], markersize=8, label=RACE_LABEL[r].replace("\n", " "))
        for r in COMPARE
    ]
    handles += [
        plt.Line2D([0], [0], color=ACCENT, lw=2.2, label="медиана категории"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#CA8A04", markeredgecolor="#1C1917", markersize=12, label="победитель"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8.5)
    add_note(
        ax,
        "СПРАВКА\n"
        "Та же картинка, что scatter по темпу, но в км/ч.\n"
        "Единая шкала для всех дистанций: можно глазами\n"
        "сравнить облака и увидеть выпадающих бегунов.\n"
        "Скорость = 60 / темп(мин/км).",
        "lower left",
    )
    save(fig, "00b_speed_scatter")


def plot_finish_minutes_panels(df: pd.DataFrame) -> None:
    """Абсолютные минуты финиша — по панели на дистанцию."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharey=False)
    axes = axes.ravel()
    for ax, race in zip(axes, COMPARE):
        minutes = df[df["race"] == race]["finish_s"] / 60.0
        minutes = minutes[minutes <= minutes.quantile(0.99)]
        ax.hist(minutes, bins=28, color=COLORS[race], edgecolor="white", linewidth=0.4)
        ax.axvline(minutes.median(), color=ACCENT, lw=2, label=f"медиана {minutes.median():.0f} мин")
        ax.axvline(minutes.min(), color="#CA8A04", lw=1.5, ls="--", label=f"победитель {minutes.min():.0f} мин")
        ax.set_title(RACE_LABEL[race].replace("\n", " "))
        ax.set_xlabel("Финишное время, минуты")
        ax.set_ylabel("Число участников")
        ax.grid(True, axis="y", alpha=0.4)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Сколько минут бежали участники (каждая дистанция — своя шкала)", y=1.01)
    fig.text(
        0.01,
        -0.01,
        "СПРАВКА: минуты нельзя сравнивать между панелями напрямую (разная длина дистанции). "
        "Для сравнения на одной шкале смотрите scatter темпа/скорости (00, 00b).",
        fontsize=8.2,
        color=MUTED,
    )
    save(fig, "00c_finish_minutes_panels")


def plot_cdf_minmax(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for race in COMPARE:
        g = df[df["race"] == race]["norm_minmax"].dropna().sort_values()
        if g.empty:
            continue
        y = np.linspace(0, 1, len(g), endpoint=False)
        ax.plot(g, y, lw=2.2, color=COLORS[race], label=f"{race} (n={len(g)})")
    ax.axvline(0.2, color="#A8A29E", ls="--", lw=1)
    ax.axvline(0.8, color="#A8A29E", ls="--", lw=1)
    ax.set_xlabel("Нормировка min–max: (t − t_min) / (t_max − t_min)")
    ax.set_ylabel("Доля финишёров (CDF)")
    ax.set_title("Форма поля: CDF нормированных времён")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.5)
    ax.legend(frameon=False, loc="center right")
    add_note(
        ax,
        "СПРАВКА\n"
        "Внимание: шкала 0…1 завязана на самого медленного.\n"
        "Одна «черепаха» среди тысячи быстрых растягивает\n"
        "ось и меняет вид кривой. Для глаз смотрите 00/00b\n"
        "(реальный темп). Этот график — только для сравнения\n"
        "формы между дистанциями, с оговоркой.",
        "lower right",
    )
    save(fig, "01_cdf_minmax")


def plot_kde_p1p99(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = np.linspace(0, 1, 400)
    for race in COMPARE:
        g = df[df["race"] == race]["norm_p1p99"].dropna().astype(float)
        if len(g) < 20:
            continue
        kde = stats.gaussian_kde(g.clip(0, 1))
        ax.plot(xs, kde(xs), lw=2.2, color=COLORS[race], label=race)
        ax.fill_between(xs, kde(xs), alpha=0.12, color=COLORS[race])
    ax.set_xlabel("Робастная нормировка: (t − p01) / (p99 − p01), обрезка [0; 1]")
    ax.set_ylabel("Плотность")
    ax.set_title("Плотность времён без влияния единичных аутлайеров")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.45)
    ax.legend(frameon=False)
    add_note(
        ax,
        "СПРАВКА\n"
        "Min–max чувствителен к одному ультра-медленному финишу.\n"
        "Здесь границы — 1-й и 99-й перцентили: форма поля\n"
        "устойчивее. Пик ближе к 0 → типичный участник ближе\n"
        "к лидерам; пик ближе к 1 → типичный ближе к хвосту.",
        "upper right",
    )
    save(fig, "02_kde_p1p99")


def plot_share_edges(df: pd.DataFrame) -> None:
    rows = []
    for race in COMPARE:
        g = df[df["race"] == race]["norm_p1p99"].dropna()
        rows.append(
            {
                "race": race,
                "left": (g <= 0.2).mean() * 100,
                "right": (g >= 0.8).mean() * 100,
            }
        )
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    x = np.arange(len(tab))
    w = 0.36
    ax.bar(x - w / 2, tab["left"], w, color="#15803D", label="Быстрые: x ≤ 0.2")
    ax.bar(x + w / 2, tab["right"], w, color="#C2410C", label="Медленные: x ≥ 0.8")
    ax.set_xticks(x)
    ax.set_xticklabels(tab["race"])
    ax.set_ylabel("Доля финишёров, %")
    ax.set_title("Масса у быстрого и медленного края (шкала p01–p99)")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.45)
    for i, r in tab.iterrows():
        ax.text(i - w / 2, r["left"] + 0.6, f"{r['left']:.1f}%", ha="center", fontsize=8)
        ax.text(i + w / 2, r["right"] + 0.6, f"{r['right']:.1f}%", ha="center", fontsize=8)
    add_note(
        ax,
        "СПРАВКА\n"
        "Гипотеза: полумарафон — больше «спортсменов» → выше доля\n"
        "у левого края; 10/5 км — массовее → больше справа.\n"
        "Считаем на робастной шкале p01–p99: один ультра-медленный\n"
        "финиш не схлопывает всю шкалу (как бывает у min–max).",
        "upper left",
    )
    save(fig, "03_share_fast_slow")


def plot_metrics_table(df: pd.DataFrame) -> None:
    rows = []
    for race in COMPARE:
        g = df[df["race"] == race]
        t = g["finish_s"]
        x = g["norm_p1p99"]
        rows.append(
            {
                "Дистанция": race,
                "n": len(g),
                "Медиана": fmt_hms(t.median()),
                "Победитель": fmt_hms(t.min()),
                "mean/median": f"{t.mean() / t.median():.3f}",
                "skew": f"{t.skew():.2f}",
                "p90/p10": f"{t.quantile(0.9) / t.quantile(0.1):.2f}",
                "≤0.2*": f"{(x <= 0.2).mean() * 100:.1f}%",
                "≥0.8*": f"{(x >= 0.8).mean() * 100:.1f}%",
                "med/win": f"{t.median() / t.min():.2f}",
            }
        )
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    ax.set_title("Сводные метрики формы распределения", pad=14)
    table = ax.table(
        cellText=tab.values,
        colLabels=tab.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#E7E5E4")
        if r == 0:
            cell.set_facecolor("#1B4332")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F4")
        else:
            cell.set_facecolor("#FFFcf7")
    fig.text(
        0.02,
        0.04,
        "СПРАВКА: * доли ≤0.2 / ≥0.8 считаются на шкале p01–p99. "
        "skew > 0 — длинный хвост медленных; med/win — типичный участник vs победитель.",
        fontsize=8.2,
        color=MUTED,
        wrap=True,
    )
    save(fig, "04_summary_metrics")


def plot_vs_winner(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    data, labels, colors = [], [], []
    for race in COMPARE:
        g = df[df["race"] == race]["norm_vs_winner"].dropna().astype(float)
        # обрежем визуально редкий ультра-хвост
        g = g[g <= g.quantile(0.99)]
        data.append(g)
        labels.append(race)
        colors.append(COLORS[race])
    parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.8)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.55)
        body.set_edgecolor(color)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        if key in parts:
            parts[key].set_color(MUTED)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("t / t_победителя")
    ax.set_title("Глубина поля относительно победителя")
    ax.grid(True, axis="y", alpha=0.45)
    add_note(
        ax,
        "СПРАВКА\n"
        "Значение 1.0 — время победителя. Медиана 1.5 значит,\n"
        "что типичный финишёр на 50% медленнее лидера.\n"
        "Уже «облако» у единицы — плотная конкуренция в топе;\n"
        "растянутое вверх — большое любительское поле.\n"
        "Показаны данные до 99-го перцентиля.",
        "upper left",
    )
    save(fig, "05_violin_vs_winner")


def plot_pace(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for race in COMPARE:
        g = df[df["race"] == race]["pace_s_per_km"].dropna() / 60.0
        g = g[g <= g.quantile(0.99)]
        ax.hist(g, bins=35, density=True, alpha=0.35, color=COLORS[race], label=race)
    ax.set_xlabel("Темп, мин/км")
    ax.set_ylabel("Плотность")
    ax.set_title("Распределение темпа (мин/км)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.45)
    add_note(
        ax,
        "СПРАВКА\n"
        "Темп удобен глазу, но не уравнивает физиологию:\n"
        "тот же человек на 21.1 км почти всегда медленнее,\n"
        "чем на 5–10 км. Для сравнения «уровня» смотрите\n"
        "нормировки и эквивалент Ригеля. Хвост обрезан по p99.",
        "upper right",
    )
    save(fig, "06_pace_hist")


def plot_riegel(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = np.linspace(0, 1, 400)
    for race in COMPARE:
        g = df[df["race"] == race]["riegel_10k_s"].dropna().astype(float) / 60.0
        if len(g) < 20:
            continue
        # CDF на общей шкале минут эквивалента 10 км
        g = g.sort_values()
        y = np.linspace(0, 1, len(g), endpoint=False)
        ax.plot(g, y, lw=2.2, color=COLORS[race], label=race)
    ax.set_xlabel("Эквивалент 10 км по Ригелю, минуты")
    ax.set_ylabel("CDF")
    ax.set_title("Сопоставимый уровень: эквивалент 10 км (Riegel, b=1.06)")
    ax.grid(True, alpha=0.45)
    ax.legend(frameon=False, loc="lower right")
    add_note(
        ax,
        "СПРАВКА\n"
        "T₂ = T₁ · (D₂/D₁)^1.06 — перевод результата в «как будто 10 км».\n"
        "Позволяет грубо сравнить уровень между дистанциями.\n"
        "Это не про состав поля, а про силу показанных времён.\n"
        "Формула эмпирическая, для элиты и любителей груба.",
        "upper left",
    )
    save(fig, "07_riegel_cdf")


def plot_gender(df: pd.DataFrame) -> None:
    # для 21.1 пол зашит в название; для 10/5 — колонка gender
    fig, ax = plt.subplots(figsize=(10, 6))
    races = ["21.1 км", "10 км", "5 км"]
    shares = []
    for rg in races:
        g = df[df["race_group"] == rg]
        # объединяем Ж/М для 21.1
        women = (g["gender"] == "Ж").sum()
        men = (g["gender"] == "М").sum()
        total = women + men
        shares.append((rg, 100 * women / total if total else 0, 100 * men / total if total else 0, total))
    x = np.arange(len(shares))
    ax.bar(x, [s[1] for s in shares], color="#BE185D", label="Женщины")
    ax.bar(x, [s[2] for s in shares], bottom=[s[1] for s in shares], color="#1D4ED8", label="Мужчины")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s[0]}\nn={s[3]}" for s in shares])
    ax.set_ylabel("Доля, %")
    ax.set_ylim(0, 100)
    ax.set_title("Гендерный состав финишёров")
    ax.legend(frameon=False, loc="upper right")
    for i, s in enumerate(shares):
        ax.text(i, s[1] / 2, f"{s[1]:.0f}%", ha="center", color="white", fontsize=10)
        ax.text(i, s[1] + s[2] / 2, f"{s[2]:.0f}%", ha="center", color="white", fontsize=10)
    add_note(
        ax,
        "СПРАВКА\n"
        "Состав поля влияет на форму распределения времени.\n"
        "Если на коротких дистанциях больше новичков/семейных\n"
        "стартов, хвост медленных обычно тяжелее.",
        "lower left",
    )
    save(fig, "08_gender_mix")


def plot_categories(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    for ax, rg in zip(axes, GROUP_ORDER):
        g = df[df["race_group"] == rg]
        vc = g["category"].fillna("—").value_counts().head(10).sort_values()
        ax.barh(vc.index.astype(str), vc.values, color=COLORS.get(rg, ACCENT))
        ax.set_title(f"{rg} (n={len(g)})")
        ax.grid(True, axis="x", alpha=0.4)
    axes[0].set_xlabel("Число финишёров")
    fig.suptitle("Топ возрастных категорий", y=1.02)
    fig.text(
        0.01,
        -0.02,
        "СПРАВКА: категории взяты из протокола Startimer. Сдвиг возрастного состава "
        "между дистанциями — ещё один канал селекции «кто выходит на старт».",
        fontsize=8.2,
        color=MUTED,
    )
    save(fig, "09_age_categories")


def plot_elite_depth(df: pd.DataFrame) -> None:
    thresholds = [1.05, 1.10, 1.20, 1.50]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = np.arange(len(COMPARE))
    width = 0.18
    for i, thr in enumerate(thresholds):
        vals = []
        for race in COMPARE:
            g = df[df["race"] == race]["norm_vs_winner"]
            vals.append((g <= thr).mean() * 100)
        ax.bar(x + (i - 1.5) * width, vals, width, label=f"≤ {thr:.2f}× победителя")
    ax.set_xticks(x)
    ax.set_xticklabels(COMPARE)
    ax.set_ylabel("Доля финишёров, %")
    ax.set_title("Глубина элиты: кто уложился в +5% / +10% / +20% / +50%")
    ax.legend(frameon=False, ncols=2)
    ax.grid(True, axis="y", alpha=0.45)
    add_note(
        ax,
        "СПРАВКА\n"
        "Доля людей в пределах фиксированного отставания от победителя.\n"
        "Высокий % на +10% — «плотный» спортивный забег.\n"
        "Низкий % — лидер сильно оторвался от массы.",
        "upper left",
    )
    save(fig, "10_elite_depth")


def plot_cv_gini(df: pd.DataFrame) -> None:
    def gini(arr: np.ndarray) -> float:
        a = np.sort(arr.astype(float))
        n = len(a)
        if n == 0 or a.sum() == 0:
            return float("nan")
        idx = np.arange(1, n + 1)
        return (2 * np.sum(idx * a) / (n * np.sum(a))) - (n + 1) / n

    rows = []
    for race in COMPARE:
        t = df[df["race"] == race]["finish_s"].astype(float)
        rows.append({"race": race, "CV": t.std() / t.mean(), "Gini": gini(t.values)})
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(tab))
    ax.bar(x - 0.2, tab["CV"], 0.4, color="#0F766E", label="CV = std / mean")
    ax.bar(x + 0.2, tab["Gini"], 0.4, color="#B45309", label="Gini по времени")
    ax.set_xticks(x)
    ax.set_xticklabels(tab["race"])
    ax.set_ylabel("Индекс")
    ax.set_title("Неравенство результатов внутри дистанции")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.45)
    add_note(
        ax,
        "СПРАВКА\n"
        "CV — относительный разброс. Gini — неравенство «как доходы»:\n"
        "0 — все финишировали одинаково, ближе к 1 — сильный разрыв\n"
        "между быстрыми и медленными. Массовые дистанции обычно\n"
        "дают большее неравенство.",
        "upper left",
    )
    save(fig, "11_cv_gini")


def plot_cities(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    g = df[df["race"].isin(COMPARE)].copy()
    g["city_norm"] = g["city"].fillna("не указан").astype(str).str.strip()
    top = g["city_norm"].value_counts().head(12).sort_values()
    ax.barh(top.index, top.values, color="#334155")
    ax.set_xlabel("Финишёры")
    ax.set_title("Откуда участники (топ городов, все сравниваемые дистанции)")
    ax.grid(True, axis="x", alpha=0.4)
    add_note(
        ax,
        "СПРАВКА\n"
        "География старта: доля Новосибирска vs иногородних\n"
        "косвенно говорит о «выездном» спортивном ядре.\n"
        "Город как в протоколе, без ручной нормализации написаний.",
        "lower right",
    )
    save(fig, "12_cities")


def plot_clubs(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    g = df[df["race"].isin(COMPARE)].copy()
    club = g["club"].fillna("").astype(str).str.strip()
    with_club = (club != "") & (club.str.lower() != "nan")
    share = with_club.mean() * 100
    by_race = (
        g.assign(has_club=with_club.values)
        .groupby("race")["has_club"]
        .mean()
        .reindex(COMPARE)
        * 100
    )
    ax.bar(range(len(by_race)), by_race.values, color=[COLORS[r] for r in by_race.index])
    ax.set_xticks(range(len(by_race)))
    ax.set_xticklabels(by_race.index)
    ax.set_ylabel("Доля с указанным клубом, %")
    ax.set_title(f"Клубные участники (всего {share:.0f}% указали клуб)")
    ax.grid(True, axis="y", alpha=0.45)
    add_note(
        ax,
        "СПРАВКА\n"
        "Указанный клуб — грубый прокси «организованного» бегуна.\n"
        "Если на полумарафоне доля клубов выше, это поддерживает\n"
        "гипотезу о более спортивном составе длинной дистанции.",
        "upper right",
    )
    save(fig, "13_club_share")


def plot_qq_hm_vs_10k(df: pd.DataFrame) -> None:
    # сравниваем объединённый 21.1 и 10 км на шкале p1p99
    a = df[df["race_group"] == "21.1 км"]["norm_p1p99"].dropna().astype(float)
    b = df[df["race"] == "10 км"]["norm_p1p99"].dropna().astype(float)
    qs = np.linspace(0.02, 0.98, 80)
    fig, ax = plt.subplots(figsize=(7.5, 7.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#A8A29E", label="y = x")
    ax.scatter(np.quantile(a, qs), np.quantile(b, qs), s=28, color=ACCENT, alpha=0.85)
    ax.set_xlabel("Квантили 21.1 км (норм. p01–p99)")
    ax.set_ylabel("Квантили 10 км (норм. p01–p99)")
    ax.set_title("QQ: одинакова ли форма 21.1 и 10 км?")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.45)
    ax.legend(frameon=False)
    add_note(
        ax,
        "СПРАВКА\n"
        "Точки на диагонали — формы совпадают.\n"
        "Выше диагонали в правой части: на 10 км верхние\n"
        "квантили «медленнее» (тяжелее правый хвост).\n"
        "Ниже — наоборот.",
        "lower right",
    )
    save(fig, "14_qq_21_vs_10")


def plot_hypothesis_verdict(df: pd.DataFrame) -> None:
    """Итоговая панель под гипотезу."""
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.axis("off")
    lines = ["Итог по гипотезе (робастная шкала p01–p99)", ""]
    for race in COMPARE:
        g = df[df["race"] == race]
        x = g["norm_p1p99"]
        t = g["finish_s"]
        left = (x <= 0.2).mean() * 100
        right = (x >= 0.8).mean() * 100
        lines.append(
            f"{race}: n={len(g)}, skew={t.skew():+.2f}, "
            f"доля ≤0.2 = {left:.1f}%, доля ≥0.8 = {right:.1f}%, "
            f"med/win={t.median()/t.min():.2f}"
        )

    hm = df[df["race_group"] == "21.1 км"]
    ten = df[df["race"] == "10 км"]
    five = df[df["race"] == "5 км"]

    def pack(name, g):
        x = g["norm_p1p99"]
        return name, (x <= 0.2).mean() * 100, (x >= 0.8).mean() * 100, g["finish_s"].skew()

    packs = [pack("21.1 км (Ж+М)", hm), pack("10 км", ten), pack("5 км", five)]
    lines += ["", "Агрегировано:"]
    for name, left, right, sk in packs:
        lines.append(f"  {name}: ≤0.2={left:.1f}%  ≥0.8={right:.1f}%  skew={sk:+.2f}")

    lines += [
        "",
        "Как читать: выше доля слева и/или ниже доля справа → поле ближе к быстрым.",
        "Min–max на 21.1 М искажён одним ультра-медленным — ориентируйтесь на p01–p99.",
        "Смотрите графики 01–03 и таблицу 04.",
    ]
    text = "\n".join(lines)
    ax.text(0.03, 0.95, text, va="top", ha="left", family="DejaVu Sans", fontsize=11, color=INK)
    rect = FancyBboxPatch(
        (0.02, 0.08),
        0.96,
        0.84,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="#FFFCF7",
        edgecolor="#D6D3D1",
        zorder=0,
    )
    ax.add_patch(rect)
    ax.set_title("Карточка гипотезы", pad=8)
    add_note(
        ax,
        "СПРАВКА\n"
        "Описательная сводка, не формальный тест.\n"
        "Для устойчивости к хвостам основная метрика — p01–p99,\n"
        "а не сырой min–max.",
        "lower right",
    )
    save(fig, "15_hypothesis_card")


def plot_finish_minutes_box(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    data = []
    for race in COMPARE:
        g = df[df["race"] == race]["finish_s"] / 60.0
        g = g[g <= g.quantile(0.99)]
        data.append(g)
    bp = ax.boxplot(
        data,
        tick_labels=COMPARE,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=ACCENT, lw=2),
    )
    for patch, race in zip(bp["boxes"], COMPARE):
        patch.set_facecolor(COLORS[race])
        patch.set_alpha(0.55)
    ax.set_ylabel("Финишное время, минуты")
    ax.set_title("Абсолютные времена (без нормировки)")
    ax.grid(True, axis="y", alpha=0.45)
    add_note(
        ax,
        "СПРАВКА\n"
        "Сырые минуты нельзя напрямую сравнивать между\n"
        "дистанциями — разный километраж. График нужен как\n"
        "контекст «какие вообще были результаты». Усы без выбросов,\n"
        "отрезано по 99-му перцентилю.",
        "upper left",
    )
    save(fig, "16_boxplot_minutes")


def main() -> None:
    setup_style()
    df = load()
    print("Финишёров:", len(df))
    # Сначала интуитивные графики в реальных единицах
    plot_pace_scatter(df)
    plot_speed_scatter(df)
    plot_finish_minutes_panels(df)
    plot_cdf_minmax(df)
    plot_kde_p1p99(df)
    plot_share_edges(df)
    plot_metrics_table(df)
    plot_vs_winner(df)
    plot_pace(df)
    plot_riegel(df)
    plot_gender(df)
    plot_categories(df)
    plot_elite_depth(df)
    plot_cv_gini(df)
    plot_cities(df)
    plot_clubs(df)
    plot_qq_hm_vs_10k(df)
    plot_hypothesis_verdict(df)
    plot_finish_minutes_box(df)
    print("Готово:", FIG)


if __name__ == "__main__":
    main()
