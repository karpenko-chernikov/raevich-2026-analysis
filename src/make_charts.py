#!/usr/bin/env python3
"""Графики анализа полумарафона Раевича — каждый график отдельным PNG со справкой."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"

# Палитра: не «AI-purple», спокойные контрастные цвета
COLORS = {
    "5 км": "#1f7a6e",
    "10 км": "#c45c26",
    "21.1 км": "#2c4a7c",
    "21.1 км Ж": "#8b3a62",
    "21.1 км М": "#2c4a7c",
    "эстафета": "#6b5b4b",
}

ORDER_GROUPS = ["5 км", "10 км", "21.1 км"]
ORDER_RACES = ["5 км", "10 км", "21.1 км Ж", "21.1 км М", "эстафета"]


def setup_style() -> None:
    # предпочитаем шрифты с кириллицей
    candidates = [
        "Arial Unicode MS",
        "Helvetica Neue",
        "DejaVu Sans",
        "SF Pro Display",
        "Arial",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update(
        {
            "figure.facecolor": "#f7f4ef",
            "axes.facecolor": "#f7f4ef",
            "axes.edgecolor": "#3a3a3a",
            "axes.labelcolor": "#222",
            "text.color": "#222",
            "xtick.color": "#333",
            "ytick.color": "#333",
            "axes.grid": True,
            "grid.color": "#d9d2c5",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.titleweight": "medium",
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
            "savefig.facecolor": "#f7f4ef",
        }
    )


def fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def add_help_box(ax, text: str, loc: str = "upper right") -> None:
    """Развёрнутая справка в углу графика."""
    anchors = {
        "upper right": (0.98, 0.98, "right", "top"),
        "upper left": (0.02, 0.98, "left", "top"),
        "lower right": (0.98, 0.02, "right", "bottom"),
        "lower left": (0.02, 0.02, "left", "bottom"),
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
        linespacing=1.35,
        color="#2a2a2a",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#fffdf8",
            "edgecolor": "#b7aa96",
            "alpha": 0.92,
        },
        zorder=10,
    )


def save(fig, name: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, pad_inches=0.25)
    plt.close(fig)
    print("→", path.name)
    return path


def finishers(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["finished"]].copy()


def plot_01_minmax_hist(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(ORDER_GROUPS)]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bins = np.linspace(0, 1, 25)
    for grp in ORDER_GROUPS:
        x = fin.loc[fin["race_group"] == grp, "norm_minmax"].astype(float)
        ax.hist(
            x,
            bins=bins,
            density=True,
            alpha=0.45,
            color=COLORS[grp],
            label=f"{grp} (n={len(x)})",
            edgecolor="white",
            linewidth=0.4,
        )
    ax.set_xlabel("Нормированное время (min–max внутри дистанции)")
    ax.set_ylabel("Плотность")
    ax.set_title("Распределение min–max: где «толпится» поле")
    ax.legend(frameon=False)
    add_help_box(
        ax,
        "Справка\n"
        "x = (t − tmin) / (tmax − tmin).\n"
        "0 — у победителя дистанции, 1 — у самого\n"
        "медленного финишёра. Если полумарафон\n"
        "«спортивнее», масса должна быть ближе к 0;\n"
        "у массовых 5/10 км — сильнее правый хвост.\n"
        "Чувствителен к одному ультра-медленному\n"
        "результату → смотрите также p1–p99.",
        "upper right",
    )
    save(fig, "01_minmax_histogram.png")


def plot_02_p1p99_kde(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(ORDER_GROUPS)]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = np.linspace(0, 1, 300)
    for grp in ORDER_GROUPS:
        x = fin.loc[fin["race_group"] == grp, "norm_p1p99"].astype(float).clip(0, 1)
        if len(x) < 5:
            continue
        kde = stats.gaussian_kde(x)
        ax.plot(xs, kde(xs), color=COLORS[grp], lw=2.2, label=f"{grp} (n={len(x)})")
        ax.fill_between(xs, kde(xs), color=COLORS[grp], alpha=0.12)
    ax.set_xlabel("Робастная нормировка p1–p99")
    ax.set_ylabel("Плотность (KDE)")
    ax.set_title("Форма поля без влияния экстремальных хвостов")
    ax.legend(frameon=False)
    add_help_box(
        ax,
        "Справка\n"
        "x = (t − p01) / (p99 − p01), обрезано в [0,1].\n"
        "1% самых быстрых и 1% самых медленных\n"
        "не растягивают шкалу. Удобно сравнивать\n"
        "«типичную» форму дистанций и проверять,\n"
        "не артефакт ли смещение из‑за одного\n"
        "очень медленного финиша.",
        "upper right",
    )
    save(fig, "02_p1p99_kde.png")


def plot_03_cdf(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(ORDER_GROUPS)]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for grp in ORDER_GROUPS:
        x = np.sort(fin.loc[fin["race_group"] == grp, "norm_p1p99"].astype(float))
        y = np.linspace(0, 1, len(x), endpoint=False)
        ax.plot(x, y, color=COLORS[grp], lw=2.2, label=grp)
    ax.set_xlabel("Нормированное время p1–p99")
    ax.set_ylabel("Доля финишёров (CDF)")
    ax.set_title("CDF нормированных времён: кто быстрее набирает «массу»")
    ax.legend(frameon=False, loc="lower right")
    add_help_box(
        ax,
        "Справка\n"
        "Кривая показывает, какая доля людей\n"
        "финишировала не медленнее данного x.\n"
        "Чем левее/выше кривая на старте, тем\n"
        "больше людей сгруппировано у быстрых\n"
        "времён. Гипотеза: 21.1 км растёт круче\n"
        "слева, 5/10 км дольше «догоняют» справа.",
        "upper left",
    )
    save(fig, "03_cdf_normalized.png")


def plot_04_share_bins(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(ORDER_GROUPS)]
    rows = []
    for grp in ORDER_GROUPS:
        x = fin.loc[fin["race_group"] == grp, "norm_p1p99"].astype(float)
        rows.append(
            {
                "Дистанция": grp,
                "Быстрые [0–0.2]": (x <= 0.2).mean() * 100,
                "Середина (0.2–0.8)": ((x > 0.2) & (x < 0.8)).mean() * 100,
                "Медленные [0.8–1]": (x >= 0.8).mean() * 100,
            }
        )
    tab = pd.DataFrame(rows).set_index("Дистанция")
    fig, ax = plt.subplots(figsize=(11, 6.2))
    bottom = np.zeros(len(tab))
    palette = ["#2f6f5e", "#c4a35a", "#a3442d"]
    for col, color in zip(tab.columns, palette):
        ax.bar(tab.index, tab[col], bottom=bottom, color=color, edgecolor="white", label=col)
        bottom += tab[col].values
    ax.set_ylabel("Доля финишёров, %")
    ax.set_title("Куда смещена масса: быстрые / середина / медленные")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 100)
    add_help_box(
        ax,
        "Справка\n"
        "Доли на шкале p1–p99. «Быстрые» — нижние\n"
        "20% шкалы (ближе к минимуму), «медленные» —\n"
        "верхние 20%. Если полумарафон спортивнее,\n"
        "доля быстрых выше, а медленный хвост\n"
        "тоньше, чем на 5/10 км.",
        "lower right",
    )
    save(fig, "04_share_fast_mid_slow.png")


def plot_05_vs_winner(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(ORDER_GROUPS)]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    data, labels, colors = [], [], []
    for grp in ORDER_GROUPS:
        x = fin.loc[fin["race_group"] == grp, "norm_vs_winner"].astype(float)
        # обрежем визуальный хвост для читаемости
        x = x[x <= x.quantile(0.99)]
        data.append(x)
        labels.append(grp)
        colors.append(COLORS[grp])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylabel("t / t_победителя")
    ax.set_title("Глубина поля относительно победителя")
    add_help_box(
        ax,
        "Справка\n"
        "1.0 — время победителя. Медиана 1.5 значит,\n"
        "что типичный финишёр на 50% медленнее\n"
        "лидера. Узкий ящик у верха = плотный\n"
        "спортивный пелотон; широкий и высоко\n"
        "поднятый = большое отставание массы.\n"
        "Выбросы выше p99 скрыты для читаемости.",
        "upper left",
    )
    save(fig, "05_vs_winner_boxplot.png")


def plot_06_pace_violin(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(ORDER_GROUPS)]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    data, labels, colors = [], [], []
    for grp in ORDER_GROUPS:
        x = fin.loc[fin["race_group"] == grp, "pace_s_per_km"].astype(float) / 60.0
        x = x[x <= x.quantile(0.99)]
        data.append(x)
        labels.append(grp)
        colors.append(COLORS[grp])
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for body, c in zip(parts["bodies"], colors):
        body.set_facecolor(c)
        body.set_alpha(0.55)
        body.set_edgecolor("#333")
    parts["cmedians"].set_color("#222")
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("Темп, мин/км")
    ax.set_title("Темп по дистанциям (не эквивалент уровня)")
    add_help_box(
        ax,
        "Справка\n"
        "Темп = время / дистанцию. Удобен для\n"
        "интуиции, но один и тот же человек на\n"
        "21.1 км почти всегда медленнее, чем на\n"
        "5–10 км. Для сравнения «силы» лучше\n"
        "нормировки и формула Ригеля.",
        "upper right",
    )
    save(fig, "06_pace_violin.png")


def plot_07_riegel(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(ORDER_GROUPS)]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    bins = np.linspace(
        fin["riegel_10k_s"].quantile(0.01) / 60,
        fin["riegel_10k_s"].quantile(0.99) / 60,
        35,
    )
    for grp in ORDER_GROUPS:
        x = fin.loc[fin["race_group"] == grp, "riegel_10k_s"].astype(float) / 60.0
        ax.hist(
            x,
            bins=bins,
            density=True,
            alpha=0.4,
            color=COLORS[grp],
            label=grp,
            edgecolor="white",
            linewidth=0.3,
        )
    ax.set_xlabel("Эквивалент 10 км по Ригелю, минуты")
    ax.set_ylabel("Плотность")
    ax.set_title("Сравнение уровня: все дистанции → эквивалент 10 км")
    ax.legend(frameon=False)
    add_help_box(
        ax,
        "Справка\n"
        "T2 = T1 · (D2/D1)^1.06 (Riegel).\n"
        "Переводит результат в «какой был бы\n"
        "примерно 10 км». Это про уровень, не про\n"
        "форму старта. Сдвиг влево = более сильное\n"
        "поле на этой дистанции.",
        "upper right",
    )
    save(fig, "07_riegel_equivalent_10k.png")


def plot_08_metrics_table(df: pd.DataFrame) -> None:
    fin = finishers(df)
    rows = []
    for race in ORDER_RACES:
        g = fin[fin["race"] == race]
        if g.empty:
            continue
        t = g["finish_s"].astype(float)
        x = g["norm_p1p99"].astype(float)
        rows.append(
            {
                "Дистанция": race,
                "n": len(g),
                "Медиана": fmt_hms(t.median()),
                "mean/median": round(t.mean() / t.median(), 3),
                "Skew": round(t.skew(), 2),
                "p90/p10": round(t.quantile(0.9) / t.quantile(0.1), 2),
                "% у быстрых": f"{(x <= 0.2).mean()*100:.1f}%",
                "% у медленных": f"{(x >= 0.8).mean()*100:.1f}%",
                "med/winner": round(t.median() / t.min(), 2),
            }
        )
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.axis("off")
    table = ax.table(
        cellText=tab.values,
        colLabels=tab.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.05, 1.6)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c4a7c")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#efe8dc")
        else:
            cell.set_facecolor("#f7f4ef")
    ax.set_title("Сводные метрики формы распределения по дистанциям", pad=18)
    fig.text(
        0.99,
        0.02,
        "Справка: mean/median>1 и большой Skew = тяжёлый хвост медленных; "
        "высокая доля «у быстрых» на p1–p99 поддерживает гипотезу о более спортивном поле.",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333",
    )
    save(fig, "08_summary_metrics.png")
    tab.to_csv(DATA / "metrics_by_race.csv", index=False)


def plot_09_gender(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(ORDER_GROUPS)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # доли
    ax = axes[0]
    shares = (
        fin.groupby(["race_group", "gender"]).size().unstack(fill_value=0).reindex(ORDER_GROUPS)
    )
    shares = shares.div(shares.sum(axis=1), axis=0) * 100
    shares[["М", "Ж"]].plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#2c4a7c", "#8b3a62"],
        edgecolor="white",
        legend=True,
    )
    ax.set_ylabel("Доля, %")
    ax.set_xlabel("")
    ax.set_title("Пол: состав поля")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False, title="")

    # медианный темп
    ax = axes[1]
    for gender, color in [("М", "#2c4a7c"), ("Ж", "#8b3a62")]:
        meds = []
        for grp in ORDER_GROUPS:
            x = fin[(fin["race_group"] == grp) & (fin["gender"] == gender)]["pace_s_per_km"]
            meds.append((x.median() / 60.0) if len(x) else np.nan)
        ax.plot(ORDER_GROUPS, meds, marker="o", lw=2, color=color, label=gender)
    ax.set_ylabel("Медианный темп, мин/км")
    ax.set_title("Медианный темп М/Ж")
    ax.legend(frameon=False)

    fig.suptitle("Гендерный профиль дистанций", y=1.02, fontsize=15)
    add_help_box(
        axes[0],
        "Справка\n"
        "Слева — кто пришёл на старт.\n"
        "Справа — медианный темп.\n"
        "На полумарафоне протоколы\n"
        "уже разделены на М и Ж.",
        "lower left",
    )
    save(fig, "09_gender_composition.png")


def plot_10_categories(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(["5 км", "10 км", "21.1 км"])]
    fin = fin[fin["category"].notna() & (fin["category"].astype(str).str.len() > 0)]
    # укрупним: вытащим М/Ж и возрастную группу как есть, топ категорий
    top = fin["category"].value_counts().head(12).index
    sub = fin[fin["category"].isin(top)]
    ct = pd.crosstab(sub["category"], sub["race_group"])
    ct = ct.reindex(columns=[c for c in ORDER_GROUPS if c in ct.columns])
    ct = ct.loc[ct.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(11, 7))
    left = np.zeros(len(ct))
    for grp in ct.columns:
        ax.barh(ct.index, ct[grp], left=left, color=COLORS[grp], edgecolor="white", label=grp)
        left += ct[grp].values
    ax.set_xlabel("Число финишёров")
    ax.set_title("Топ возрастных категорий по дистанциям")
    ax.legend(frameon=False)
    add_help_box(
        ax,
        "Справка\n"
        "Показывает, какие возрастные группы\n"
        "наполняют каждую дистанцию. Сдвиг к\n"
        "более «взрослым» категориям на 21.1 км\n"
        "часто означает больший опыт/селекцию.",
        "lower right",
    )
    save(fig, "10_age_categories.png")


def plot_11_clubs(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(ORDER_GROUPS)]
    club = fin["club"].fillna("").astype(str).str.strip()
    no_club = club.isin(["", "Нет", "нет", "—", "-", "None"])
    rates = []
    for grp in ORDER_GROUPS:
        m = fin["race_group"] == grp
        rates.append(
            {
                "Дистанция": grp,
                "С клубом": (~no_club[m]).mean() * 100,
                "Без клуба": no_club[m].mean() * 100,
            }
        )
    tab = pd.DataFrame(rates)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(tab["Дистанция"], tab["С клубом"], color="#2f6f5e", label="Указан клуб")
    ax.bar(
        tab["Дистанция"],
        tab["Без клуба"],
        bottom=tab["С клубом"],
        color="#c4a35a",
        label="Без клуба / «Нет»",
    )
    ax.set_ylabel("Доля, %")
    ax.set_ylim(0, 100)
    ax.set_title("Клубная принадлежность как прокси «серьёзности»")
    ax.legend(frameon=False)
    add_help_box(
        ax,
        "Справка\n"
        "Доля участников с указанным клубом.\n"
        "Не идеальный индикатор, но на массовых\n"
        "дистанциях обычно больше «разовых»\n"
        "без клуба. Сравнивайте вместе с формой\n"
        "распределения времён.",
        "upper right",
    )
    save(fig, "11_club_share.png")


def plot_12_cities(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(ORDER_GROUPS)].copy()

    def norm_city(c):
        if pd.isna(c):
            return "не указан"
        s = str(c).strip().lower().replace("г.", "").replace("г ", "")
        s = " ".join(s.split())
        if "новосиб" in s or s in {"novosibirsk", "novosobirsk"}:
            return "Новосибирск"
        return str(c).strip()

    fin["city_n"] = fin["city"].map(norm_city)
    local = fin["city_n"].eq("Новосибирск")
    rows = []
    for grp in ORDER_GROUPS:
        m = fin["race_group"] == grp
        rows.append(
            {
                "Дистанция": grp,
                "Новосибирск": local[m].mean() * 100,
                "Иногородние": (~local[m]).mean() * 100,
            }
        )
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar(tab["Дистанция"], tab["Новосибирск"], color="#2c4a7c", label="Новосибирск")
    ax.bar(
        tab["Дистанция"],
        tab["Иногородние"],
        bottom=tab["Новосибирск"],
        color="#c45c26",
        label="Другие города",
    )
    ax.set_ylabel("Доля, %")
    ax.set_ylim(0, 100)
    ax.set_title("География старта: местные vs иногородние")
    ax.legend(frameon=False)
    add_help_box(
        ax,
        "Справка\n"
        "Города нормализованы эвристикой\n"
        "(опечатки вроде Novosobirsk → Новосибирск).\n"
        "Выше доля иногородних на длинной\n"
        "дистанции часто = более целевой приезд.",
        "upper right",
    )
    save(fig, "12_city_local_vs_other.png")


def plot_13_elite_depth(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(ORDER_GROUPS)]
    thresholds = [1.05, 1.10, 1.20, 1.50]
    fig, ax = plt.subplots(figsize=(11, 6.2))
    x = np.arange(len(thresholds))
    width = 0.25
    for i, grp in enumerate(ORDER_GROUPS):
        g = fin[fin["race_group"] == grp]
        winner = g["finish_s"].min()
        shares = [((g["finish_s"] / winner) <= thr).mean() * 100 for thr in thresholds]
        ax.bar(
            x + i * width,
            shares,
            width=width,
            color=COLORS[grp],
            label=grp,
            edgecolor="white",
        )
    ax.set_xticks(x + width, [f"≤{int((t-1)*100)}% к лидеру" for t in thresholds])
    ax.set_ylabel("Доля финишёров, %")
    ax.set_title("Плотность элиты: сколько людей рядом с победителем")
    ax.legend(frameon=False)
    add_help_box(
        ax,
        "Справка\n"
        "Доля людей в пределах +5/+10/+20/+50%\n"
        "от времени победителя. Высокие значения\n"
        "у коротких порогов = глубокий конкурентный\n"
        "топ. На массовых дистанциях топ часто\n"
        "тоньше относительно всей массы.",
        "upper left",
    )
    save(fig, "13_elite_depth.png")


def plot_14_qq(df: pd.DataFrame) -> None:
    fin = finishers(df)
    a = np.sort(fin.loc[fin["race_group"] == "10 км", "norm_p1p99"].astype(float))
    b = np.sort(fin.loc[fin["race_group"] == "21.1 км", "norm_p1p99"].astype(float))
    # интерполируем на общие квантили
    qs = np.linspace(0.01, 0.99, 200)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    fig, ax = plt.subplots(figsize=(7.5, 7.2))
    ax.scatter(qa, qb, s=18, color="#2c4a7c", alpha=0.75)
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.2)
    ax.set_xlabel("Квантили 10 км (p1–p99)")
    ax.set_ylabel("Квантили 21.1 км (p1–p99)")
    ax.set_title("QQ-plot: 10 км vs полумарафон")
    ax.set_aspect("equal", adjustable="box")
    add_help_box(
        ax,
        "Справка\n"
        "Точки выше диагонали: на этом квантиле\n"
        "полумарафон относительно медленнее\n"
        "(правее на своей шкале). Ниже диагонали —\n"
        "наоборот. Систематический изгиб показывает\n"
        "различие форм, а не только сдвиг.",
        "lower right",
    )
    save(fig, "14_qq_10k_vs_half.png")


def plot_15_ks_stats(df: pd.DataFrame) -> None:
    fin = finishers(df)
    pairs = [("5 км", "10 км"), ("5 км", "21.1 км"), ("10 км", "21.1 км")]
    rows = []
    for a, b in pairs:
        xa = fin.loc[fin["race_group"] == a, "norm_p1p99"].astype(float)
        xb = fin.loc[fin["race_group"] == b, "norm_p1p99"].astype(float)
        ks = stats.ks_2samp(xa, xb)
        rows.append({"Пара": f"{a} vs {b}", "KS": ks.statistic, "p-value": ks.pvalue})
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    table = ax.table(cellText=[[r["Пара"], f"{r['KS']:.3f}", f"{r['p-value']:.2e}"] for _, r in tab.iterrows()],
                     colLabels=["Сравнение", "Статистика KS", "p-value"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f7a6e")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f0ebe3" if r % 2 else "#f7f4ef")
    ax.set_title("Колмогоров–Смирнов: отличаются ли нормированные распределения?")
    fig.text(
        0.5,
        0.08,
        "Справка: KS сравнивает CDF двух выборок на шкале p1–p99. "
        "Малый p-value → формы статистически различимы (не доказывает причину).",
        ha="center",
        fontsize=8.5,
    )
    save(fig, "15_ks_tests.png")
    tab.to_csv(DATA / "ks_tests.csv", index=False)


def plot_16_hypothesis_verdict(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(ORDER_GROUPS)]
    metrics = []
    for grp in ORDER_GROUPS:
        t = fin.loc[fin["race_group"] == grp, "finish_s"].astype(float)
        x = fin.loc[fin["race_group"] == grp, "norm_p1p99"].astype(float)
        metrics.append(
            {
                "grp": grp,
                "skew": t.skew(),
                "fast": (x <= 0.2).mean() * 100,
                "slow": (x >= 0.8).mean() * 100,
                "mean_med": t.mean() / t.median(),
            }
        )
    m = pd.DataFrame(metrics).set_index("grp")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2))
    axes[0].bar(m.index, m["fast"], color=[COLORS[g] for g in m.index])
    axes[0].set_title("Доля у быстрых [0–0.2]")
    axes[0].set_ylabel("%")
    axes[1].bar(m.index, m["slow"], color=[COLORS[g] for g in m.index])
    axes[1].set_title("Доля у медленных [0.8–1]")
    axes[2].bar(m.index, m["skew"], color=[COLORS[g] for g in m.index])
    axes[2].set_title("Асимметрия (skew) сырого времени")
    for ax in axes:
        ax.tick_params(axis="x", rotation=0)

    # вердикт
    half_fast = m.loc["21.1 км", "fast"]
    ten_fast = m.loc["10 км", "fast"]
    half_skew = m.loc["21.1 км", "skew"]
    ten_skew = m.loc["10 км", "skew"]
    support = (half_fast > ten_fast) and (half_skew < ten_skew)
    verdict = (
        "Гипотеза скорее подтверждается:\nполумарафон плотнее у быстрых, хвост медленных легче."
        if support
        else "Гипотеза не выглядит однозначной:\nсмотрите метрики — картина смешанная."
    )
    fig.suptitle("Проверка гипотезы о «спортивности» полумарафона", fontsize=15)
    fig.text(0.5, -0.02, verdict, ha="center", fontsize=11, color="#1f3d2f")
    add_help_box(
        axes[2],
        "Справка\n"
        "Ожидание гипотезы:\n"
        "у 21.1 км выше доля «быстрых»\n"
        "и меньше/иная асимметрия, чем\n"
        "у 10 км. Это упрощённый тест;\n"
        "полная картина — в остальных\n"
        "графиках.",
        "upper right",
    )
    save(fig, "16_hypothesis_check.png")


def plot_17_gini_cv(df: pd.DataFrame) -> None:
    fin = finishers(df)
    fin = fin[fin["race_group"].isin(ORDER_GROUPS)]

    def gini(x):
        x = np.sort(np.asarray(x, dtype=float))
        n = len(x)
        if n == 0 or x.sum() == 0:
            return np.nan
        return (2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n

    rows = []
    for grp in ORDER_GROUPS:
        t = fin.loc[fin["race_group"] == grp, "finish_s"].astype(float)
        rows.append({"Дистанция": grp, "Gini": gini(t), "CV": t.std() / t.mean()})
    tab = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(tab))
    w = 0.35
    ax.bar(x - w / 2, tab["Gini"], w, color="#2c4a7c", label="Gini")
    ax.bar(x + w / 2, tab["CV"], w, color="#c45c26", label="CV (σ/μ)")
    ax.set_xticks(x, tab["Дистанция"])
    ax.set_title("Неравенство результатов: Gini и коэффициент вариации")
    ax.legend(frameon=False)
    add_help_box(
        ax,
        "Справка\n"
        "Gini и CV растут, когда поле\n"
        "сильнее размазано по уровню.\n"
        "Массовая дистанция часто даёт\n"
        "большее «неравенство» времён,\n"
        "чем более селективная.",
        "upper left",
    )
    save(fig, "17_inequality_gini_cv.png")
    tab.to_csv(DATA / "inequality.csv", index=False)


def main() -> None:
    setup_style()
    df = pd.read_csv(DATA / "results.csv")
    # numeric
    for col in [
        "finish_s",
        "pace_s_per_km",
        "norm_minmax",
        "norm_p1p99",
        "norm_vs_winner",
        "norm_vs_median",
        "percentile",
        "riegel_10k_s",
        "distance_km",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    plot_01_minmax_hist(df)
    plot_02_p1p99_kde(df)
    plot_03_cdf(df)
    plot_04_share_bins(df)
    plot_05_vs_winner(df)
    plot_06_pace_violin(df)
    plot_07_riegel(df)
    plot_08_metrics_table(df)
    plot_09_gender(df)
    plot_10_categories(df)
    plot_11_clubs(df)
    plot_12_cities(df)
    plot_13_elite_depth(df)
    plot_14_qq(df)
    plot_15_ks_stats(df)
    plot_16_hypothesis_verdict(df)
    plot_17_gini_cv(df)
    print("Готово:", FIG)


if __name__ == "__main__":
    main()
