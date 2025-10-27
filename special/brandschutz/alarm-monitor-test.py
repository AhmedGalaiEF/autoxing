#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, math, time, json, signal, threading, re
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
HEARTBEAT_SEC = 5
# --- timings ---
POLL_INTERVAL_SEC   = 0.2    # fast alarm polling
ASSIGN_REFRESH_SEC  = 120     # preplan refresh while idle
RESEND_COOLDOWN_SEC = 60     # ONLY throttles re-cancel/re-dispatch
VERBOSE_REPLAN_LOGS = True  # set False if too chatty

EVAC_REGEX = r"(?i)^Evac.*"
BRANDSCHUTZ_REGEX = r"(?i)^Brandschutz$"  # exact name match, case-insensitive
BRANDSCHUTZ_RADIUS_M = 20.0               # << requirement

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

# ---- FIXED get_state (no pandas dtype warnings) ----
# If api_lib.Robot already defines get_state, you can monkey-patch it like this,
# or copy these lines into your class implementation directly.

def cancel_if_active(robot: Robot):
    try:
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
    """
    Global set of all Evac.* POIs across robots, with coordinates and metadata.
    """
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

def brandschutz_ok(is_at_df: pd.DataFrame) -> bool:
    """
    True if there is at least one Brandschutz *area* in isAt.
    (isAt already applies min_dist_area threshold from get_state)
    """
    if not isinstance(is_at_df, pd.DataFrame) or is_at_df.empty:
        return False
    area_rows = is_at_df[is_at_df["kind"].astype(str).str.lower().eq("area")]
    if area_rows.empty:
        return False
    name_series = area_rows["name"].astype(str)
    return name_series.str.match(BRANDSCHUTZ_REGEX, na=False).any()

def get_robot_local_evac_candidates(robot: Robot, global_evacs_by_id: dict) -> list[dict]:
    """
    For a single robot:
    - require proximity to a Brandschutz area within 20m (by calling get_state with min_dist_area=20)
    - return only Evac.* POIs present in st['isAt'] for this robot, mapped to full POI dicts from global list.
    """
    try:
        log(f"getting robot state... [{robot.SN}]")
        st = robot.get_state(min_dist_area=BRANDSCHUTZ_RADIUS_M)
        is_at = st.get("isAt")
        if not brandschutz_ok(is_at):
            log(f"[{robot.SN}] Not within {BRANDSCHUTZ_RADIUS_M:.0f} m of a Brandschutz area → skip.")
            return []
        # POIs in isAt that match EVAC_REGEX
        poi_rows = is_at[is_at["kind"].astype(str).str.lower().eq("poi")]
        if poi_rows.empty:
            log(f"[{robot.SN}] No Evac.* POIs in isAt.")
            return []
        poi_rows = poi_rows[poi_rows["name"].astype(str).str.match(EVAC_REGEX, na=False)]
        if poi_rows.empty:
            log(f"[{robot.SN}] No Evac.* POIs in isAt after filter.")
            return []
        # Map to global POI dicts (ensure we use consistent x/y/yaw, areaId, etc.)
        out = []
        for _, row in poi_rows.iterrows():
            pid = str(row.get("id"))
            if not pid:
                continue
            g = global_evacs_by_id.get(pid)
            if g:
                out.append(g)
        if not out:
            log(f"[{robot.SN}] Evac.* POIs in isAt but no global match by id (ids may differ).")
        return out
    except Exception as e:
        log(f"[{robot.SN}] get_state/isAt parse failed: {e}")
        return []

def compute_best_assignment(robots: list[Robot], global_evac_pts: list[dict], should_stop=lambda: False):
    """
    Build per-robot candidate POIs from isAt (only if near Brandschutz within 20m),
    then greedy assign shortest planned path, ensuring unique POIs.
    """
    if should_stop():
        return {}
    if not robots or not global_evac_pts:
        return {}

    evacs_by_id = {p["id"]: p for p in global_evac_pts}

    # Gather candidates per robot
    candidates = {}
    for r in robots:
        if should_stop():
            return {}
        cands = get_robot_local_evac_candidates(r, evacs_by_id)
        if cands:
            candidates[r.SN] = cands
        else:
            # Explicit: skip this robot
            pass

    if not candidates:
        log("No eligible robots (within Brandschutz radius) with local Evac.* candidates.")
        return {}

    # Build all (length, rid, pid) pairs only from allowed candidates
    pairs = []
    for r in robots:
        rid = r.SN
        if rid not in candidates:
            continue
        for p in candidates[rid]:
            if should_stop():
                return {}
            L = plan_length(r, p)
            pairs.append((L, rid, p["id"]))

    if not pairs:
        log("No feasible (robot, Evac POI) pairs.")
        return {}

    pairs.sort(key=lambda t: t[0])

    assigned, used_pts = {}, set()
    for L, rid, pid in pairs:
        if should_stop():
            return assigned
        try:
            if rid in assigned or pid in used_pts:
                continue
            if not math.isfinite(L):
                continue
            assigned[rid] = evacs_by_id[pid] | {"planned_length_m": L}
            used_pts.add(pid)
            if len(assigned) >= len(candidates):
                break
        except KeyboardInterrupt:
            raise
        except Exception:
            continue

    # Log summary
    if assigned:
        brief = ", ".join(f"{rid}→{v['name']} (~{v['planned_length_m']:.1f} m)"
                          for rid, v in assigned.items())
        log(f"Planned (Brandschutz≤{BRANDSCHUTZ_RADIUS_M:.0f} m): {brief}")
    else:
        log("No assignments found under Brandschutz proximity rule.")
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
            resp = None
            tid = (resp or {}).get("taskId")
            log(f"[{r.SN}] Task created: {tid}")
        except KeyboardInterrupt:
            raise
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
        self._last_heartbeat = 0.0

    def stop(self):
        with self._lock:
            self._stop = True

    def should_stop(self):
        with self._lock:
            return self._stop

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
        due = (now - self._last_assign_ts) >= ASSIGN_REFRESH_SEC
        if not force and not due:
            if VERBOSE_REPLAN_LOGS:
                remaining = ASSIGN_REFRESH_SEC - (now - self._last_assign_ts)
                log(f"[replan] skipped (cooldown {remaining:.1f}s)")
            return
        if not self._robots or not self._evac_pts:
            if VERBOSE_REPLAN_LOGS:
                log("[replan] skipped (no robots or no Evac.* POIs)")
            return

        log(f"Preplanning optimal assignments (Brandschutz≤{BRANDSCHUTZ_RADIUS_M:.0f} m)…")
        new_map = compute_best_assignment(self._robots, self._evac_pts, self.should_stop)

        with self._lock:
            old = self._cached_targets
            self._cached_targets = new_map
            self._last_assign_ts = now

        if not new_map:
            log("[replan] no feasible assignments.")
        else:
            brief = ", ".join(f"{rid}→{v['name']} (~{v['planned_length_m']:.1f} m)"
                              for rid, v in new_map.items())
            if old != new_map:
                log(f"[replan] updated: {brief}")
            elif VERBOSE_REPLAN_LOGS:
                log(f"[replan] unchanged: {brief}")


    def run(self):
        self.refresh_inputs()
        self.recompute_assignments_if_due(force=True)

        last_alarm = False
        log(f"Monitoring GPIO{ALARM_GPIO} (NC opens on alarm). "
            f"Poll {POLL_INTERVAL_SEC}s. Resend cooldown {RESEND_COOLDOWN_SEC}s.")

        try:
            while True:
                if self.should_stop():
                    break

                alarm = is_alarm_active()
                now = time.time()

                if alarm:

                    if not last_alarm:
                        log("ALARM detected → dispatch now (first wave)")
                        if not self._robots:
                            self.refresh_inputs()
                        if not self._cached_targets:
                            self.recompute_assignments_if_due(force=True)
                        dispatch_to_cached_targets(self._robots, self._cached_targets)
                        self._last_dispatch_ts = now

                    elif (now - self._last_dispatch_ts) >= RESEND_COOLDOWN_SEC:
                        log("ALARM still active → re-dispatch after cooldown")
                        # (optional) refresh inputs infrequently
                        if (now - self._last_assign_ts) >= ASSIGN_REFRESH_SEC:
                            self.refresh_inputs()
                            self.recompute_assignments_if_due(force=True)
                        dispatch_to_cached_targets(self._robots, self._cached_targets)
                        self._last_dispatch_ts = now
                    # Try to keep assignments fresh even during alarm
                    self.recompute_assignments_if_due(force=False)

                else:
                    # Idle background maintenance
                    if (now - self._last_assign_ts) >= ASSIGN_REFRESH_SEC:
                        self.refresh_inputs()
                    self.recompute_assignments_if_due(force=False)

                last_alarm = alarm
                time.sleep(POLL_INTERVAL_SEC)

                if now - getattr(self, "_last_heartbeat", 0.0) >= HEARTBEAT_SEC:
                    log(f"[hb] alarm={alarm} cached_targets={len(self._cached_targets)} last_replan={int(now - self._last_assign_ts)}s ago")
                    self._last_heartbeat = now


        except KeyboardInterrupt:
            log("Loop stopped by user (Ctrl+C).")
            self.stop()
        finally:
            GPIO.cleanup()

# ---- entrypoint ----
def main():
    planner = EvacPlanner()

    # Let Ctrl+C (SIGINT) and SIGTERM raise KeyboardInterrupt normally.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    try:
        planner.run()
    except KeyboardInterrupt:
        log("Interrupted by user.")
    finally:
        try:
            GPIO.cleanup()
        except Exception:
            pass

if __name__ == "__main__":
    main()