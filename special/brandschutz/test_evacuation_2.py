#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, math
# from detect_alarm import callback

#sys.path.append(r"C:\Users\Ahmed Galai\Desktop\dev\sdk\AX_PY_SDK")
from datetime import datetime
from AX_PY_SDK_5 import (
    get_business_robots, Robot,
    get_robot_status, cancel_task, create_task,
)

def log(msg: str):
    print(f"{datetime.now().isoformat()} {msg}", flush=True)

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
def cancel_if_active(robot: Robot):
    try:
        status_df = get_robot_status(robot.df)
        if status_df.empty:
            log(f"[{robot.SN}] no status info.")
            return

        task_obj = None

        if "data" in status_df.columns:
            cell = status_df["data"].iloc[0]
            if isinstance(cell, dict):
                task_obj = cell.get("taskObj")
            elif isinstance(cell, str):
                # try parse stringified JSON
                try:
                    task_obj = pd.json.loads(cell).get("taskObj")
                except Exception:
                    log(f"[{robot.SN}] data cell is plain string: {cell}")
            else:
                log(f"[{robot.SN}] Unexpected data type in 'data': {type(cell)}")

        elif "taskObj" in status_df.columns:
            cell = status_df["taskObj"].iloc[0]
            if isinstance(cell, dict):
                task_obj = cell
            else:
                log(f"[{robot.SN}] taskObj cell is not a dict: {cell}")

        if task_obj and isinstance(task_obj, dict) and task_obj.get("taskId"):
            tid = task_obj["taskId"]
            log(f"[{robot.SN}] Canceling active task {tid}...")
            cancel_task(tid)
            log(f"[{robot.SN}] Task {tid} canceled.")
        else:
            log(f"[{robot.SN}] No active task.")
    except Exception as e:
        log(f"[{robot.SN}] Error canceling task: {e}")

        
def find_standby_evac_pois(robot: Robot):
    pois = robot.get_pois()
    if pois.empty:
        return []
    
    print(pois.name)

    mask = pois["name"].astype(str).str.match(r"^Evac\d+$", na=False)

    # if "type" in pois.columns:
    #     mask &= pois["type"].astype(str).str.lower().eq("standby")

    return [row.to_dict() for _, row in pois[mask].iterrows()]


def assign_and_dispatch(robots):
    ref_robot = robots[0]
    evac_pts = find_standby_evac_pois(ref_robot)
    if not evac_pts:
        log("No standby evac points found.")
        return

    used = set()
    for robot in robots:
        cancel_if_active(robot)
        pos = robot.get_curr_pos()

        candidates = sorted(
            [pt for pt in evac_pts if pt["id"] not in used],
            key=lambda pt: euclid(pos, (pt["coordinate"][0], pt["coordinate"][1])),
        )
        if not candidates:
            log(f"[{robot.SN}] No free evac point.")
            continue

        pt = candidates[0]
        used.add(pt["id"])

        task_name = f"evacuate_{robot.SN}_{int(datetime.now().timestamp())}"
        log(f"[{robot.SN}] Dispatching to {pt['name']} (area {pt['areaId']})")

        try:
            resp = create_task(
                task_name=task_name,
                robot=robot.df,
                runType=22,        # direct
                sourceType=6,      # SDK
                taskPts=[{
                    "x": pt["coordinate"][0],
                    "y": pt["coordinate"][1],
                    "yaw": pt.get("yaw", 0),
                    "stopRadius": 1,
                    "areaId": pt["areaId"],
                    "type": pt.get("type"),
                    "ext": {"id": pt["id"], "name": pt["name"]},
                }],
                runNum=1,
                taskType=4,        # delivery
                routeMode=2,       # shortest
                runMode=1,         # flex avoid
                speed=1.0,
                detourRadius=1.0,
                ignorePublicSite=True,
            )
            log(f"[{robot.SN}] Task created: {resp.get('taskId')}")
        except Exception as e:
            log(f"[{robot.SN}] Task creation failed: {e}")

def main():
    import time
    cooldown = 20
    df = get_business_robots("Assa Abloy")
    online = df.query("isOnLine == True")
    if online.empty:
        log("No online Assa Abloy robots.")
        sys.exit(1)

    robots = [Robot(rid) for rid in online["robotId"]]
    # robots = [Robot("2383308702220zO")]
    log(f"Testing evacuation dispatch for Assa Abloy robots: {', '.join([r.SN for r in robots])}")
    while True:
        # if callback():
        assign_and_dispatch(robots)
        time.sleep(cooldown)

if __name__ == "__main__":
    main()

