#!/usr/bin/env python3
"""Преобразование сырых строк startimer в tidy CSV."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def parse_time_to_seconds(value) -> float | None:
    """Время с startimer: миллисекунды (int) или строка hh:mm:ss."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value != value or value <= 0:
            return None
        # API отдаёт финиш в миллисекундах (например 4459153 ≈ 1:14:19)
        if value >= 1000:
            return float(value) / 1000.0
        return float(value)
    s = str(value).strip()
    if not s or s in {"-", "—", "DNS", "DNF", "DSQ", "n/a", "None"}:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", s):
        num = float(s)
        return num / 1000.0 if num >= 1000 else num
    parts = s.split(":")
    try:
        parts_f = [float(p) for p in parts]
    except ValueError:
        return None
    if len(parts_f) == 3:
        h, m, sec = parts_f
        return h * 3600 + m * 60 + sec
    if len(parts_f) == 2:
        m, sec = parts_f
        return m * 60 + sec
    if len(parts_f) == 1:
        return parts_f[0]
    return None


def race_group(name: str) -> str:
    if name.startswith("21.1"):
        return "21.1 км"
    return name


def build_frame(raw: dict, meta: dict) -> pd.DataFrame:
    rows = []
    for race, table in raw.items():
        cols = meta[race]["columns"]
        distance_m = meta[race]["distance_m"]
        name_to_idx = {c["name"]: c["field"] for c in cols if c.get("name")}
        finish_idx = name_to_idx.get("finish")
        for arr in table:
            status = arr[name_to_idx["status"]] if "status" in name_to_idx else None
            finish_raw = (
                arr[finish_idx] if finish_idx is not None and finish_idx < len(arr) else None
            )
            finish_s = parse_time_to_seconds(finish_raw)
            gender = arr[name_to_idx["gender"]] if "gender" in name_to_idx else None
            if race == "21.1 км Ж":
                gender = "Ж"
            elif race == "21.1 км М":
                gender = "М"
            row = {
                "race": race,
                "race_group": race_group(race),
                "distance_m": distance_m,
                "distance_km": distance_m / 1000.0 if distance_m else None,
                "bib": arr[name_to_idx["bib"]] if "bib" in name_to_idx else None,
                "last_name": arr[name_to_idx["last_name"]] if "last_name" in name_to_idx else None,
                "first_name": arr[name_to_idx["first_name"]] if "first_name" in name_to_idx else None,
                "gender": gender,
                "category": arr[name_to_idx["cat"]] if "cat" in name_to_idx else None,
                "city": arr[name_to_idx["city"]] if "city" in name_to_idx else None,
                "club": arr[name_to_idx["club"]] if "club" in name_to_idx else None,
                "status": status,
                "finish_raw": finish_raw,
                "finish_s": finish_s,
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    before = len(df)
    df = df.drop_duplicates(subset=["race", "bib"], keep="first").reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Удалено дубликатов race+bib: {dropped}")
    df["finished"] = df["finish_s"].notna() & (df["finish_s"] > 0)
    df.loc[df["finished"], "pace_s_per_km"] = (
        df.loc[df["finished"], "finish_s"] / df.loc[df["finished"], "distance_km"]
    )
    return df


def add_normalizations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in (
        "norm_minmax",
        "norm_p1p99",
        "norm_vs_winner",
        "norm_vs_median",
        "percentile",
        "riegel_10k_s",
    ):
        out[col] = pd.NA

    for race, g in out.groupby("race"):
        mask = out["race"] == race
        fin = g["finished"]
        times = g.loc[fin, "finish_s"]
        if times.empty:
            continue
        tmin, tmax = times.min(), times.max()
        p1, p99 = times.quantile(0.01), times.quantile(0.99)
        med = times.median()
        winner = tmin
        ranks = times.rank(method="average", ascending=True)
        pct = (ranks - 1) / max(len(times) - 1, 1)

        out.loc[mask & out["finished"], "norm_minmax"] = (times - tmin) / (
            tmax - tmin if tmax > tmin else 1
        )
        denom = (p99 - p1) if p99 > p1 else 1
        out.loc[mask & out["finished"], "norm_p1p99"] = ((times - p1) / denom).clip(0, 1)
        out.loc[mask & out["finished"], "norm_vs_winner"] = times / winner
        out.loc[mask & out["finished"], "norm_vs_median"] = times / med
        out.loc[mask & out["finished"], "percentile"] = pct.values

        dist = g["distance_km"].iloc[0]
        if dist and dist > 0:
            out.loc[mask & out["finished"], "riegel_10k_s"] = times * (10.0 / dist) ** 1.06

    return out


def main() -> None:
    raw = json.loads((DATA / "raw_rows.json").read_text(encoding="utf-8"))
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    df = add_normalizations(build_frame(raw, meta))
    df.to_csv(DATA / "results.csv", index=False)
    try:
        df.to_parquet(DATA / "results.parquet", index=False)
    except Exception as exc:  # noqa: BLE001
        print("parquet пропущен:", exc)

    finished = df[df["finished"]].copy()
    summary = (
        finished.groupby(["race", "race_group", "distance_km"], dropna=False)
        .agg(
            n=("finish_s", "size"),
            median_s=("finish_s", "median"),
            mean_s=("finish_s", "mean"),
            min_s=("finish_s", "min"),
            max_s=("finish_s", "max"),
            skew=("finish_s", "skew"),
        )
        .reset_index()
    )
    summary.to_csv(DATA / "summary_by_race.csv", index=False)
    print(summary.to_string(index=False))
    print("Строк:", len(df), "финишёров:", int(df["finished"].sum()))


if __name__ == "__main__":
    main()
