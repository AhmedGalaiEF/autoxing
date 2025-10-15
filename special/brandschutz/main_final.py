#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, math, time, json
# RPi + SDK imports go here (you said don't worry about imports)
import RPi.GPIO as GPIO
from AX_PY_SDK_5 import get_business_robots, Robot, get_robot_status, cancel_task, create_task
from datetime import datetime

ALARM_GPIO = 17  # BCM

# --- GPIO setup ---
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(ALARM_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def log(msg: str):
    print(f"{datetime.now().isoformat()} {msg}", flush=True)

def is_alarm_active() -> bool:
    """NC contact opens on alarm -> input falls LOW via pull-down."""
    state = GPIO.input(ALARM_GPIO)
    return state == GPIO.LOW
    # return False  # <- replace with the two lines above when running on Pi

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def cancel_if_active(robot):
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
                try:
                    task_obj = json.loads(cell).get("taskObj")
                except Exception:
                    log(f"[{robot.SN}] data cell is plain string: {cell}")
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

def find_standby_evac_pois(robot):
    pois = robot.get_pois()
    if pois.empty:
        return []
    mask = pois["name"].astype(str).str.match(r"^Evac\d+$", na=False)
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
        pos = robot.get_curr_pos()  # (x, y)
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
                runType=22, sourceType=6,
                taskPts=[{
                    "x": pt["coordinate"][0], "y": pt["coordinate"][1],
                    "yaw": pt.get("yaw", 0), "stopRadius": 1,
                    "areaId": pt["areaId"], "type": pt.get("type"),
                    "ext": {"id": pt["id"], "name": pt["name"]},
                }],
                runNum=1, taskType=4, routeMode=2, runMode=1,
                speed=1.0, detourRadius=1.0, ignorePublicSite=True,
            )
            log(f"[{robot.SN}] Task created: {resp.get('taskId')}")
        except Exception as e:
            log(f"[{robot.SN}] Task creation failed: {e}")

def main():
    df = get_business_robots("Assa Abloy")
    online = df.query("isOnLine == True")
    if online.empty:
        log("No online Assa Abloy robots.")
        sys.exit(1)

    robots = [Robot(rid) for rid in online["robotId"]]
    log(f"Evacuation dispatch for: {', '.join([r.SN for r in robots])}")

    cooldown = 20
    latched = False
    log("Monitoring GPIO17 (NC contact opens on alarm, wiring: 3.3V→contact→GPIO)")

    try:
        while True:
            a = is_alarm_active()
            if a and not latched:
                log("ALARM detected → dispatch once")
                assign_and_dispatch(robots)
                latched = True
            elif not a and latched:
                log("Alarm cleared → unlatch")
                latched = False
            time.sleep(cooldown)
    finally:
        GPIO.cleanup()   # uncomment on the Pi
        pass

if __name__ == "__main__":
    main()
