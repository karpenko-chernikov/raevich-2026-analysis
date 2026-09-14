#!/usr/bin/env python3
"""Скачивание результатов с startimer.online через WebSocket /stream-up/."""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
from pathlib import Path

import websockets

DEFAULT_EVENT = "20260912-novosibirsk"
URI = "wss://startimer.online/stream-up/"
PAGE = 50


async def recv_cmd(ws, timeout: float = 60):
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))


async def connect():
    sslctx = ssl.create_default_context()
    return await websockets.connect(
        URI,
        ssl=sslctx,
        max_size=50_000_000,
        open_timeout=30,
        ping_interval=None,
        close_timeout=5,
        additional_headers={
            "Origin": "https://startimer.online",
            "User-Agent": "Mozilla/5.0",
        },
    )


async def load_settings(ws, event: str) -> tuple[list[str], dict]:
    await ws.send(
        json.dumps(
            {
                "command": "update_race_settings",
                "race_name": "",
                "lang": "ru",
                "event": event,
            }
        )
    )
    available: list[str] = []
    settings: dict = {}
    while True:
        msg = await recv_cmd(ws)
        if msg.get("command") != "update_race_settings":
            continue
        if not available:
            available = msg["data"]["availableRaces"]
        for k, v in msg["data"].get("races", {}).items():
            settings[k] = v
        missing = [r for r in available if r not in settings]
        if not missing:
            return available, settings
        await ws.send(
            json.dumps(
                {
                    "command": "update_race_settings",
                    "race_name": missing[0],
                    "lang": "ru",
                    "event": event,
                }
            )
        )


async def select_race(ws, event: str, race: str) -> dict:
    await ws.send(
        json.dumps(
            {
                "command": "update_race_settings",
                "race_name": race,
                "lang": "ru",
                "event": event,
            }
        )
    )
    while True:
        msg = await recv_cmd(ws)
        if msg.get("command") == "update_race_settings":
            return list(msg["data"]["races"].values())[0]


async def fetch_race_rows(ws, event: str, race: str) -> list:
    # Сначала явно выбираем дистанцию — иначе rowsMax «липнет» к предыдущей.
    await select_race(ws, event, race)
    rows: list = []
    page = 1
    rows_max = None
    empty_streak = 0
    while True:
        await ws.send(
            json.dumps(
                {
                    "event": event,
                    "command": "subscribe_on_update_data",
                    "params": {
                        "filters": [],
                        "page": page,
                        "rows_count": PAGE,
                        "race_name": race,
                    },
                }
            )
        )
        while True:
            msg = await recv_cmd(ws)
            if msg.get("status") == "ERROR":
                raise RuntimeError(f"{race} p{page}: {msg}")
            if msg.get("command") == "update_data":
                chunk = msg["data"]["rows"]
                rows_max = msg["data"]["rowsMax"]
                rows.extend(chunk)
                print(f"  {race}: p{page} +{len(chunk)} => {len(rows)}/{rows_max}")
                break
        if rows_max is None:
            break
        if not chunk:
            empty_streak += 1
            if empty_streak >= 2:
                break
        else:
            empty_streak = 0
        if len(rows) >= rows_max:
            break
        page += 1
    return rows[:rows_max] if rows_max is not None else rows


async def fetch_all(event: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ws = await connect()
    try:
        available, settings = await load_settings(ws, event)
        print("Дистанции:", available)
        all_rows: dict[str, list] = {}
        for race in available:
            for attempt in range(1, 4):
                try:
                    settings[race] = await select_race(ws, event, race)
                    all_rows[race] = await fetch_race_rows(ws, event, race)
                    break
                except (
                    websockets.exceptions.ConnectionClosed,
                    asyncio.TimeoutError,
                    RuntimeError,
                ) as exc:
                    print(f"  повтор {attempt}/3 для {race}: {exc}")
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    await asyncio.sleep(1.5)
                    ws = await connect()
                    available2, settings2 = await load_settings(ws, event)
                    available = available2
                    settings.update(settings2)
            else:
                raise RuntimeError(f"Не удалось скачать: {race}")
    finally:
        try:
            await ws.close()
        except Exception:
            pass

    meta = {
        race: {
            "distance_m": settings[race]["raceSettings"].get("distance"),
            "n": len(all_rows[race]),
            "columns": settings[race]["columnsSettings"],
        }
        for race in available
    }
    (out_dir / "raw_rows.json").write_text(
        json.dumps(all_rows, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Сохранено:", {k: len(v) for k, v in all_rows.items()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", default=DEFAULT_EVENT)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    asyncio.run(fetch_all(args.event, args.out))


if __name__ == "__main__":
    main()
