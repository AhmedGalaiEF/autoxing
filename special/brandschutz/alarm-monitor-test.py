#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, math, time, json, signal, threading
from datetime import datetime
import RPi.GPIO as GPIO

from api_lib import (
    Robot,
    get_business_robots,
    create_task,
    cancel_task,
    pd
)

BUSINESS_NAME_PREFIX = "Assa Abloy"
ALARM_GPIO = 17

# --- timings ---
POLL_INTERVAL_SEC   = 0.2    # fast alarm polling
ASSIGN_REFRESH_SEC  = 10     # preplan refresh while idle
RESEND_COOLDOWN_SEC = 60     # ONLY throttles re-cancel/re-dispatch

EVAC_REGEX = r"(?i)^Evac.*"

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(ALARM_GPIO, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def is_alarm_active() -> bool:
    # NC contact opens on alarm -> input LOW via pull-down
    # return GPIO.input(ALARM_GPIO) == GPIO.LOW
    return True

def log(msg: str): print(f"{datetime.now().isoformat()} {msg}", flush=True)

def euclid(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])

def safe_get_pose_xy(robot: Robot):
    try:
        st = robot.get_state()
        pose = st.get("pose")
        if isinstance(pose, (tuple, list)) and len(pose) >= 2:
            return float(pose[0]), float(pose[1])
    except Exception:
        pass
    try:
        x, y = robot.get_curr_pos()
        return float(x), float(y)
    except Exception:
        return float("nan"), float("nan")

def cancel_if_active(robot: Robot):
    try:
        # st = robot.get_state()
        task_obj = robot.get_task()
        if isinstance(task_obj, dict) and task_obj.get("taskId"):
            tid = task_obj["taskId"]
            log(f"[{robot.SN}] Canceling active task {tid}...")
            # cancel_task(tid)
            log(f"[{robot.SN}] Task {tid} canceled.")
        else:
            log(f"[{robot.SN}] No active task.")
    except Exception as e:
        log(f"[{robot.SN}] Error canceling task: {e}")

def collect_evac_points(robots: list[Robot]):
    seen_ids, points = set(), []
    for r in robots:
        try:
            pois = r.get_pois()
            if isinstance(pois, pd.DataFrame) and not pois.empty:
                mask = pois["name"].astype(str).str.match(EVAC_REGEX, na=False)
                for _, row in pois[mask].iterrows():
                    pid = str(row.get("id"))
                    if pid and pid not in seen_ids:
                        coord = row.get("coordinate") or [None, None]
                        points.append({
                            "id": pid,
                            "name": str(row.get("name", "")),
                            "x": float(coord[0]),
                            "y": float(coord[1]),
                            "yaw": row.get("yaw", 0) or 0,
                            "areaId": row.get("areaId"),
                            "type": row.get("type")
                        })
                        seen_ids.add(pid)
        except Exception as e:
            log(f"[{r.SN}] POI fetch failed: {e}")
    return points

def plan_length(robot: Robot, poi) -> float:
    # Prefer planned path length; fallback Euclidean
    try:
        res = robot.plan_path(poi["name"])
        L = float(res.get("length_m") or 0.0)
        if L > 0: return L
    except Exception:
        pass
    rx, ry = safe_get_pose_xy(robot)
    if not (math.isfinite(rx) and math.isfinite(ry)): return float("inf")
    return euclid((rx, ry), (poi["x"], poi["y"]))

def compute_best_assignment(robots: list[Robot], evac_pts: list[dict]):
    if not robots or not evac_pts: return {}
    pairs = []
    for r in robots:
        for p in evac_pts:
            L = plan_length(r, p)
            pairs.append((L, r.SN, p["id"]))
    pairs.sort(key=lambda t: t[0])

    assigned, used_pts = {}, set()
    by_id = {p["id"]: p for p in evac_pts}
    for L, rid, pid in pairs:
        if rid in assigned or pid in used_pts: continue
        if not math.isfinite(L): continue
        assigned[rid] = by_id[pid] | {"planned_length_m": L}
        used_pts.add(pid)
        if len(assigned) >= len(robots): break
    return assigned

def dispatch_to_cached_targets(robots: list[Robot], cache: dict[str, dict]):
    used = set()
    for r in robots:
        poi = cache.get(r.SN)
        if not poi:
            log(f"[{r.SN}] No cached target. Skipping.")
            continue
        if poi["id"] in used:
            log(f"[{r.SN}] Target already used this wave. Skipping.")
            continue
        used.add(poi["id"])
        cancel_if_active(r)
        try:
            task_name = f"evacuate_{r.SN}_{int(time.time())}"
            log(f"[{r.SN}] Dispatching to {poi['name']} (area {poi.get('areaId')})")
            # resp = create_task(
            #     task_name=task_name,
            #     robot=r.df,
            #     runType=22, sourceType=6,
            #     taskPts=[{
            #         "x": poi["x"], "y": poi["y"],
            #         "yaw": poi.get("yaw", 0), "stopRadius": 1,
            #         "areaId": poi.get("areaId"), "type": poi.get("type"),
            #         "ext": {"id": poi["id"], "name": poi["name"]},
            #     }],
            #     runNum=1, taskType=4, routeMode=2, runMode=1,
            #     speed=1.0, detourRadius=1.0, ignorePublicSite=True,
            # )
            # robot.go_to_poi(poi['name'])
            tid = (resp or {}).get("taskId")
            log(f"[{r.SN}] Task created: {tid}")
        except Exception as e:
            log(f"[{r.SN}] Task creation failed: {e}")

class EvacPlanner:
    def __init__(self):
        self._stop = False
        self._lock = threading.Lock()
        self._cached_targets: dict[str, dict] = {}
        self._robots: list[Robot] = []
        self._evac_pts: list[dict] = []
        self._last_assign_ts = 0.0
        self._last_dispatch_ts = 0.0  # gates re-send while alarm stays active

    def stop(self):
        with self._lock: self._stop = True

    def load_robots(self):
        df = get_business_robots(BUSINESS_NAME_PREFIX)
        if isinstance(df, str) or df is None or df.empty:
            log(f"No robots for '{BUSINESS_NAME_PREFIX}'."); return []
        online = df[df["isOnLine"] == True]
        if online.empty:
            log(f"No online robots for '{BUSINESS_NAME_PREFIX}'."); return []
        robots = [Robot(rid) for rid in online["robotId"]]
        log(f"Tracking robots: {', '.join(r.SN for r in robots)}")
        return robots

    def refresh_inputs(self):
        self._robots = self.load_robots()
        self._evac_pts = collect_evac_points(self._robots) if self._robots else []
        if self._evac_pts:
            log(f"Found {len(self._evac_pts)} Evac POIs.")
        else:
            log("No Evac.* POIs found.")

    def recompute_assignments_if_due(self, force=False):
        now = time.time()
        if not force and (now - self._last_assign_ts) < ASSIGN_REFRESH_SEC:
            return
        if not self._robots or not self._evac_pts:
            return
        log("Preplanning optimal assignments…")
        new_map = compute_best_assignment(self._robots, self._evac_pts)
        with self._lock:
            self._cached_targets = new_map
            self._last_assign_ts = now
        if new_map:
            brief = ", ".join(f"{rid}→{v['name']} (~{v['planned_length_m']:.1f} m)"
                              for rid, v in new_map.items())
            log(f"Planned: {brief}")
        else:
            log("No feasible assignments.")

    def run(self):
        self.refresh_inputs()
        self.recompute_assignments_if_due(force=True)

        last_alarm = False
        log(f"Monitoring GPIO{ALARM_GPIO} (NC opens on alarm). "
            f"Poll {POLL_INTERVAL_SEC}s. Resend cooldown {RESEND_COOLDOWN_SEC}s.")

        try:
            while True:
                with self._lock:
                    if self._stop: break

                alarm = is_alarm_active()
                now = time.time()

                if alarm:
                    # rising edge → immediate dispatch
                    if not last_alarm:
                        log("ALARM detected → dispatch now (first wave)")
                        if not self._robots:
                            self.refresh_inputs()
                        if not self._cached_targets:
                            self.recompute_assignments_if_due(force=True)
                        dispatch_to_cached_targets(self._robots, self._cached_targets)
                        self._last_dispatch_ts = now

                    # while alarm remains active → re-dispatch only on cooldown
                    elif (now - self._last_dispatch_ts) >= RESEND_COOLDOWN_SEC:
                        log("ALARM still active → re-dispatch after cooldown")
                        # optional: refresh targets before re-dispatch
                        if not self._cached_targets:
                            self.refresh_inputs()
                            self.recompute_assignments_if_due(force=True)
                        dispatch_to_cached_targets(self._robots, self._cached_targets)
                        self._last_dispatch_ts = now

                else:
                    # alarm cleared → reset resend gate and keep planning in background
                    if last_alarm:
                        log("Alarm cleared → reset resend gate")
                    # light background maintenance
                    if (now - self._last_assign_ts) >= ASSIGN_REFRESH_SEC:
                        self.refresh_inputs()
                        self.recompute_assignments_if_due(force=True)

                last_alarm = alarm
                time.sleep(POLL_INTERVAL_SEC)
        finally:
            GPIO.cleanup()

# ---- entrypoint ----
def main():
    planner = EvacPlanner()
    def _stop(*_): planner.stop()
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    planner.run()

if __name__ == "__main__":
    import signal
    main()
