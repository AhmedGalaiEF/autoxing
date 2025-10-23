#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# examples/tasks.py

from __future__ import annotations
import os
import re
from typing import List, Optional
import sys, os
sys.path.append(os.path.join(os.getcwd(),"lib"))
# Your SDK
from api_lib import (
    Robot,
    Task,
    create_task,
    get_ef_robots,
    get_robots,    # <-- add this
    pd,
)


# -----------------------------------------------------------------------------
# Config toggles (safe defaults)
# -----------------------------------------------------------------------------
RUN_MOVES = False   # set True to actually issue navigation tasks (goto / etc.)
RUN_FACTORY_TASK = False  # set True to actually POST the factory task

# -----------------------------------------------------------------------------
# Init
# -----------------------------------------------------------------------------
def choose_robot() -> Robot:
    """
    Pick one EF robot per your rule:
    ef_robots[ef_robots.isOnLine == False].iloc[0].robotId
    (If none exist, fallback to any EF robot row.)
    """
    ef = get_ef_robots()
    if isinstance(ef, str):
        raise RuntimeError(f"get_ef_robots() returned text: {ef}")

    if ef is None or ef.empty:
        raise RuntimeError("No EF robots found.")

    cand = ef[ef.isOnLine == False]
    if cand.empty:
        row = ef.iloc[0]
    else:
        row = cand.iloc[0]

    rid = str(row["robotId"])
    print(f"[init] Selected robotId: {rid} (online={bool(row.get('isOnLine', False))})")
    robot = Robot(rid)

    # For your README style:
    # rid = '...'  # <-- keep this line commented
    return robot


# -----------------------------------------------------------------------------
# Basic examples (commented one-liners)
# -----------------------------------------------------------------------------
def basic_examples(robot: Robot) -> None:
    """
    Uncomment to try simple actions.
    """
    print("[basic] Robot is ready. Uncomment lines below to execute.")

    # robot.go_charge()
    # robot.go_back()

    # # go to a POI by name:
    # poi_name = "Warten"
    # robot.go_to_poi(poi_name)

    # # wait at POI for N seconds:
    # robot.wait_at("Warten", wait_seconds=10)

    # # lift up/down at a shelf POI (type 34):
    # robot.pickup_at("Sicht 1", area_delivery=False)   # lift up at shelf
    # robot.dropdown_at("Sicht 1", area_delivery=True)  # lift down using area delivery

    # # go to an explicit pose (x, y, yaw_deg):
    # robot.go_to_pose((0.0, 0.0, 0.0))


# -----------------------------------------------------------------------------
# Path-planning over Evac.* POIs, pick shortest, optionally go
# -----------------------------------------------------------------------------
def _poi_candidates(robot: Robot, pattern=r"(?i)^Evac.*") -> list[str]:
    df = robot.get_pois()
    if isinstance(df, str) or df is None or df.empty or "name" not in df.columns:
        return []
    return sorted(df["name"].astype(str)[df["name"].astype(str).str.match(pattern)].unique())

def _is_spot_occupied_by_business_peer(target_name: str, *, current_robot: Robot) -> bool:
    """
    Returns True if ANY other robot in the same businessId is currently 'isAt'
    the POI named target_name (case-insensitive), else False.
    """
    try:
        biz_id = str(current_robot.df.businessId)
        me_id  = str(current_robot.df.robotId)

        robots_df = get_robots()
        if isinstance(robots_df, str) or robots_df is None or robots_df.empty:
            return False  # can't check → assume free

        peers = robots_df[(robots_df["businessId"].astype(str) == biz_id) &
                          (robots_df["robotId"].astype(str) != me_id)]

        for _, row in peers.iterrows():
            rid = str(row["robotId"])
            try:
                peer = Robot(rid)
                st = peer.get_state()  # has st["isAt"] as a DataFrame
                at = st.get("isAt", None)
                if at is None or not hasattr(at, "empty") or at.empty:
                    continue
                names = at.get("name")
                kinds = at.get("kind")
                if names is None:
                    continue
                # check POIs (and areas just in case) by name, case-insensitive
                name_eq = names.astype(str).str.strip().str.lower() == target_name.strip().lower()
                if name_eq.any():
                    # If you want to restrict only to POIs, uncomment:
                    # if kinds is not None:
                    #     name_eq &= (kinds.astype(str).str.lower() == "poi")
                    if name_eq.any():
                        return True
            except Exception:
                # ignore a failing peer; keep checking others
                continue
        return False
    except Exception:
        return False

def plan_best_evac_avoiding_occupied(robot: Robot, *, move_if_free: bool = False) -> str | None:
    """
    - Find POIs matching ^Evac.*
    - Plan path length for each
    - Sort by shortest path
    - For each candidate in order, skip if ANY peer robot (same businessId) is 'isAt' that POI
    - Pick the first free spot; optionally move the robot there
    """
    cands = _poi_candidates(robot, r"(?i)^Evac.*")
    if not cands:
        print("[evac] No POIs matching ^Evac.*")
        return None

    # Plan all candidates first (to rank by length_m)
    scored: list[tuple[float, str]] = []
    for name in cands:
        try:
            plan = robot.plan_path(name)
            L = float(plan.get("length_m") or 0.0)
            if L > 0:
                scored.append((L, name))
                print(f"[evac] {name}: length ≈ {L:.2f} m")
            else:
                print(f"[evac] {name}: no valid path (length=0).")
        except Exception as e:
            print(f"[evac] plan_path failed for '{name}': {e}")

    if not scored:
        print("[evac] No valid paths to any Evac.* POI.")
        return None

    scored.sort(key=lambda t: t[0])  # shortest first
    print(f"[evac] Ordered by length: {scored}")

    # Occupancy check by business peers
    for L, name in scored:
        occ = _is_spot_occupied_by_business_peer(name, current_robot=robot)
        print(f"[evac] Check occupied? {name}: {occ}")
        if not occ:
            print(f"[evac] Selected free evacuation spot: '{name}' (≈ {L:.2f} m).")
            if move_if_free:
                robot.go_to_poi(name)
            return name

    print("[evac] All candidate evacuation spots are currently occupied by business peers.")
    return None



# -----------------------------------------------------------------------------
# Area / factory task example (lift up then down), then back to 'Warten'
# -----------------------------------------------------------------------------
def area_factory_example(robot: Robot) -> None:
    """
    Build a Task with two shelf points (pickup then dropdown), then back to 'Warten'.
    This mirrors your snippet:

      Task(temprob, "area", taskType="factory", runType="lift")
          .pickup(ptts[0]['ext']['name'], lift_up=True)
          .pickup(ptts[1]['ext']['name'], lift_down=True)
          .back("Warten")

    We’ll pick two shelf-looking POIs (`Sicht` or `Euro`) heuristically.
    """
    pois = robot.get_pois()
    if isinstance(pois, str) or pois is None or pois.empty:
        print("[factory] POI table unavailable or empty.")
        return

    names = pois["name"].astype(str)

    # Heuristics to grab two shelves:
    # 1) the highest-numbered 'Sicht N'
    # 2) the highest-numbered 'Euro N'
    def highest_number(rx: str) -> Optional[str]:
        pool = names[names.str.match(rx, case=False)]
        if pool.empty:
            return None
        def tail_num(s: str) -> int:
            try: return int(re.findall(r"(\d+)$", s.strip())[0])
            except Exception: return -1
        return max((str(s) for s in pool), key=tail_num, default=None)

    pick_a = highest_number(r"^Sicht\s*\d+$")
    pick_b = highest_number(r"^Euro\s*\d+$")

    # Fallbacks
    if not pick_a and len(names) >= 1: pick_a = names.iloc[0]
    if not pick_b and len(names) >= 2: pick_b = names.iloc[1] if names.iloc[1] != pick_a else names.iloc[0]

    print(f"[factory] pickup A: {pick_a!r}")
    print(f"[factory] dropdown B: {pick_b!r}")

    # Build the task against the same robot
    temprob = robot  # same instance is fine
    task = (
        Task(temprob, "area", taskType="factory", runType="lift")
            .pickup(pick_a, lift_up=True)         # lift up at first shelf
            .pickup(pick_b, lift_down=True)       # lift down at second shelf (API uses .pickup with lift_down=True)
            .back("Warten")                       # back to waiting
    )

    print("[factory] task_dict:")
    print(task.task_dict)

    if RUN_FACTORY_TASK:
        print("[factory] RUN_FACTORY_TASK=True → POSTing create_task")
        resp = create_task(**task.task_dict)
        print(f"[factory] create_task response: {resp!r}")
    else:
        print("[factory] RUN_FACTORY_TASK=False → not creating task (dry run).")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    robot = choose_robot()

    # -- basic (commented calls):
    basic_examples(robot)

    # -- path planning over Evac.* and optional move:
    chosen = plan_best_evac_avoiding_occupied(robot, move_if_free=False)
    print(f"[evac] Final choice: {chosen!r}")

    # -- area/factory lift example:
    area_factory_example(robot)


if __name__ == "__main__":
    main()
