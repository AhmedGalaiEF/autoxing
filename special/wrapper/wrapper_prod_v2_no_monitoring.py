# -*- coding: utf-8 -*-

import os, re, time, threading, json, glob
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, no_update
import dash_bootstrap_components as dbc

# ---- Force default timeout for all requests ----
import requests as _requests
if not getattr(_requests.sessions.Session.request, "_wrapped_with_timeout", False):
    _orig_request = _requests.sessions.Session.request
    def _request_with_default_timeout(self, method, url, **kwargs):
        if "timeout" not in kwargs or kwargs["timeout"] is None:
            kwargs["timeout"] = 5.0
        return _orig_request(self, method, url, **kwargs)
    _request_with_default_timeout._wrapped_with_timeout = True
    _requests.sessions.Session.request = _request_with_default_timeout
# -----------------------------------------------

# ===== SDK =====
from api_lib import Robot, create_task

# ===== App / Robot config =====
DEFAULT_ROBOT_ID = "FS52505505633sR"
WAITING_POI_NAME = "Warten"

# Motion / actions
RUN_TYPE_LIFT = 29
TASK_TYPE_LIFT = 5
SOURCE_SDK     = 6
ROUTE_SEQ      = 1
RUNMODE_FLEX   = 1
ACTION         = {"lift_up": 47, "lift_down": 48}

# Simulation pause action
PAUSE_SECONDS = 5
def act_pause(seconds: int = PAUSE_SECONDS) -> Dict[str, Any]:
    # type=18/pauseTime per your backend
    return {"type": 18, "data": {"pauseTime": int(seconds)}}

# Timing / gating - Now configurable via settings
POLL_SEC           = 0.55
ARRIVE_DIST_M      = 0.25
ROW_GATE_DWELL_SEC = 10.0
PRE_PULSE_DWELL_S  = 10.0
POST_PULSE_DWELL_S = 180.0
ROUNDTRIP_TIMEOUT_S = 600

# ===== GPIO (with Dummy fallback) =====
RELAY_PIN   = 23
ACTIVE_HIGH = True
PULSE_SEC   = 1.0
try:
    import RPi.GPIO as GPIO  # type: ignore
except Exception:
    class DummyGPIO:
        BCM="BCM"; OUT="OUT"; HIGH=1; LOW=0
        def setmode(self,*_a,**_k): pass
        def setup(self,p,_m):       pass
        def output(self,p,v):       pass
        def cleanup(self):          pass
    GPIO = DummyGPIO()  # type: ignore

def _gpio_init():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_PIN, GPIO.OUT)
    GPIO.output(RELAY_PIN, GPIO.LOW if ACTIVE_HIGH else GPIO.HIGH)

def _pulse_gpio():
    _log(f"[GPIO] PULSE pin {RELAY_PIN} ({'active-high' if ACTIVE_HIGH else 'active-low'}) {PULSE_SEC}s")
    GPIO.output(RELAY_PIN, GPIO.HIGH if ACTIVE_HIGH else GPIO.LOW)
    time.sleep(PULSE_SEC)
    GPIO.output(RELAY_PIN, GPIO.LOW if ACTIVE_HIGH else GPIO.HIGH)

# ===== POI helpers =====
RX_PICKUP = re.compile(r"^Abhol\s*\d+$", re.IGNORECASE)
RX_SICHT  = re.compile(r"^Sicht\s*\d+$", re.IGNORECASE)
RX_EURO   = re.compile(r"^Euro\s*\d+$",  re.IGNORECASE)
RX_DIV    = re.compile(r"^Div\s*\d+$",   re.IGNORECASE)

# --- Single robot instance (no multiple initializations) ---
_ROBOT_LOCK = threading.Lock()
_ROBOT_ID: str = DEFAULT_ROBOT_ID
_ROBOT: Optional[Robot] = None

def _pickups(robot_id: str) -> List[str]:
    df = _poi_df()
    if df.empty:
        return []
    names = [str(n) for n in df["name"].astype(str).tolist()]
    picks = {n for n in names if RX_PICKUP.fullmatch(n or "")}
    if any(n.lower() == "wrapper" for n in names):
        picks.add("Wrapper")
    def key(n):
        if n.lower() == "wrapper": return (0, 0)
        try: return (1, int(n.split()[-1]))
        except: return (1, 999999)
    return sorted(picks, key=key)

def _drops(robot_id: str) -> List[str]:
    df = _poi_df()
    if df.empty: return []
    names = [str(n) for n in df["name"].astype(str).tolist()]
    pool = set()
    for n in names:
        if RX_SICHT.fullmatch(n) or RX_EURO.fullmatch(n) or RX_DIV.fullmatch(n):
            pool.add(n)
    if any(n.lower() == "wrapper" for n in names):
        pool.add("Wrapper")
    def fam(n: str) -> int:
        nl = n.lower()
        if nl.startswith("Sicht"): return 0
        if nl.startswith("Euro"):  return 1
        if nl.startswith("Div"):   return 2
        if nl == "Wrapper":        return 3
        return 9
    def num(n: str) -> int:
        try: return int(n.split()[-1])
        except: return 0
    return sorted(pool, key=lambda n: (fam(n), num(n), n.lower()))

def get_robot() -> Robot:
    global _ROBOT
    with _ROBOT_LOCK:
        if _ROBOT is None:
            _ROBOT = Robot(_ROBOT_ID)
        return _ROBOT

def set_robot(robot_id: str):
    global _ROBOT_ID, _ROBOT
    with _ROBOT_LOCK:
        if robot_id != _ROBOT_ID or _ROBOT is None:
            _ROBOT_ID = robot_id
            _ROBOT = Robot(robot_id)

def _poi_df() -> pd.DataFrame:
    try:
        r = get_robot()
        df = r.get_pois()
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _norm_poi(row: pd.Series | Dict[str, Any]) -> Dict[str, Any]:
    # accepts pd.Series OR dict from get_poi_details
    g = row.get if hasattr(row, "get") else (lambda k, d=None: row[k] if k in row else d)  # type: ignore
    c = g("coordinate", None)
    x = float(c[0]) if isinstance(c, (list, tuple)) and len(c) >= 2 else None
    y = float(c[1]) if isinstance(c, (list, tuple)) and len(c) >= 2 else None
    return {
        "name": str(g("name","")),
        "x": x, "y": y,
        "yaw": float(g("yaw", 0.0) or 0.0),
        "areaId": str(g("areaId","") or ""),
        "id": g("id"),
    }

def _first_match_name(df: pd.DataFrame, query: str, *, regex: bool = False) -> Optional[str]:
    if df.empty or "name" not in df.columns:
        return None
    s = df["name"].astype(str)
    m = s.str.contains(query, case=False, regex=regex) if regex else (s.str.lower() == query.lower())
    if not m.any():
        return None
    return str(s[m].iloc[0])

def _poi_details_safe(name_query: str, *, regex: bool = False) -> Optional[Dict[str, Any]]:
    df = _poi_df()
    nm = _first_match_name(df, name_query, regex=regex)
    if not nm:
        return None
    try:
        det = get_robot().get_poi_details(nm)
        return _norm_poi(det)
    except Exception:
        row = df[df["name"].astype(str) == nm]
        return _norm_poi(row.iloc[0]) if not row.empty else None

def _all_poi_names() -> List[str]:
    df = _poi_df()
    if df.empty or "name" not in df.columns:
        return []
    return sorted({str(n).strip() for n in df["name"].astype(str).tolist() if str(n).strip()})

def _find_poi(name: str) -> Optional[Dict[str, Any]]:
    return _poi_details_safe(name, regex=False)

def _find_waiting() -> Optional[Dict[str, Any]]:
    p = _poi_details_safe(WAITING_POI_NAME, regex=False)
    if p: return p
    p = _poi_details_safe(r"warten", regex=True)
    if p: return p
    df = _poi_df()
    return _norm_poi(df.iloc[0]) if not df.empty else None

def _find_wrapper() -> Optional[Dict[str, Any]]:
    p = _poi_details_safe(r"wrapper", regex=True)
    if p: return p
    df = _poi_df()
    m = df[df["name"].astype(str).str.match(RX_PICKUP)]
    if not m.empty:
        return _poi_details_safe(str(m.iloc[0]["name"]), regex=False)
    return None

def _as_xy(obj):
    if isinstance(obj, dict):
        if "x" in obj and "y" in obj:
            return float(obj["x"]), float(obj["y"])
        v = obj.get("coordinate") or obj.get("pos")
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        raise TypeError(f"Unsupported dict shape for position: {obj}")
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return float(obj[0]), float(obj[1])
    raise TypeError(f"Unsupported position type: {type(obj)} -> {obj!r}")

def _distance(curr, target: Dict[str, Any]) -> float:
    cx, cy = _as_xy(curr)
    tx, ty = float(target["x"]), float(target["y"])
    return ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5

def act_lift_up()   -> Dict[str, Any]: return {"type": ACTION["lift_up"],   "data": {}}
def act_lift_down() -> Dict[str, Any]: return {"type": ACTION["lift_down"], "data": {}}

def pt(p: Dict[str, Any], acts=None, stopRadius=1.0) -> Dict[str, Any]:
    d = {"x": p["x"], "y": p["y"], "yaw": p["yaw"], "areaId": p["areaId"], "stopRadius": float(stopRadius)}
    if acts: d["stepActs"] = acts
    d["ext"] = {"id": p.get("id"), "name": p.get("name")}
    return d

def back_pt(p: Dict[str, Any]) -> Dict[str, Any]:
    return {"x": p["x"], "y": p["y"], "yaw": p["yaw"], "areaId": p["areaId"], "stopRadius": 1.0, "ext": {"id": p.get("id"), "name": p.get("name")}}

# ---- Region targeting (drop-down) ----
REGION_OPTIONS = ["Euroboxen", "Sichtlager", "Divers-Links", "Divers-Rechts"]

def _highest_numbered_name(df: pd.DataFrame, rx: re.Pattern) -> Optional[str]:
    if df.empty or "name" not in df.columns:
        return None
    names = df["name"].astype(str)
    cand = names[names.str.match(rx)]
    if cand.empty:
        return None
    def tail_num(s: str) -> int:
        try: return int(s.split()[-1])
        except Exception: return -1
    best = max(cand, key=lambda s: tail_num(str(s)))
    return str(best)

def _resolve_region_to_poi_name(region: str) -> Optional[str]:
    df = _poi_df()
    if region == "Euroboxen":
        return _highest_numbered_name(df, RX_EURO)
    if region == "Sichtlager":
        return _highest_numbered_name(df, RX_SICHT)
    if region == "Divers-Links":
        s = _first_match_name(df, "Div 8", regex=False)
        if s: return s
        cand = df["name"].astype(str)
        m = cand[cand.str.match(r"(?i)^Div\s*8$")]
        return str(m.iloc[0]) if not m.empty else None
    if region == "Divers-Rechts":
        s = _first_match_name(df, "Div 4", regex=False)
        if s: return s
        cand = df["name"].astype(str)
        m = cand[cand.str.match(r"(?i)^Div\s*4$")]
        return str(m.iloc[0]) if not m.empty else None
    return None

# ===== Settings Management =====
SETTINGS_DIR = "./settings"

def _ensure_settings_dir():
    if not os.path.exists(SETTINGS_DIR):
        os.makedirs(SETTINGS_DIR)

def _get_settings_files() -> List[str]:
    _ensure_settings_dir()
    files = glob.glob(os.path.join(SETTINGS_DIR, "*.json"))
    return sorted([os.path.basename(f) for f in files])

def _load_settings_from_file(filename: str) -> Optional[Dict[str, Any]]:
    try:
        path = os.path.join(SETTINGS_DIR, filename)
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        _log(f"[Settings] Error loading {filename}: {e}")
        return None

def _save_settings_to_file(filename: str, settings: Dict[str, Any]) -> bool:
    try:
        _ensure_settings_dir()
        path = os.path.join(SETTINGS_DIR, filename)
        with open(path, 'w') as f:
            json.dump(settings, f, indent=2)
        _log(f"[Settings] Saved to {filename}")
        return True
    except Exception as e:
        _log(f"[Settings] Error saving {filename}: {e}")
        return False

def _get_current_settings() -> Dict[str, Any]:
    return {
        "poll_sec": POLL_SEC,
        "arrive_dist_m": ARRIVE_DIST_M,
        "row_gate_dwell_sec": ROW_GATE_DWELL_SEC,
        "pre_pulse_dwell_s": PRE_PULSE_DWELL_S,
        "post_pulse_dwell_s": POST_PULSE_DWELL_S,
        "roundtrip_timeout_s": ROUNDTRIP_TIMEOUT_S,
        "pulse_sec": PULSE_SEC,
    }

def _apply_settings(settings: Dict[str, Any]):
    global POLL_SEC, ARRIVE_DIST_M, ROW_GATE_DWELL_SEC, PRE_PULSE_DWELL_S, POST_PULSE_DWELL_S, ROUNDTRIP_TIMEOUT_S, PULSE_SEC
    POLL_SEC = float(settings.get("poll_sec", POLL_SEC))
    ARRIVE_DIST_M = float(settings.get("arrive_dist_m", ARRIVE_DIST_M))
    ROW_GATE_DWELL_SEC = float(settings.get("row_gate_dwell_sec", ROW_GATE_DWELL_SEC))
    PRE_PULSE_DWELL_S = float(settings.get("pre_pulse_dwell_s", PRE_PULSE_DWELL_S))
    POST_PULSE_DWELL_S = float(settings.get("post_pulse_dwell_s", POST_PULSE_DWELL_S))
    ROUNDTRIP_TIMEOUT_S = float(settings.get("roundtrip_timeout_s", ROUNDTRIP_TIMEOUT_S))
    PULSE_SEC = float(settings.get("pulse_sec", PULSE_SEC))
    _log(f"[Settings] Applied: {settings}")

# ===== Logging buffer =====
from collections import deque, defaultdict
LOG_BUF = deque(maxlen=1500)
LOG_LOCK = threading.Lock()

RESET_ROWS = defaultdict(int)
RESET_LOCK = threading.Lock()

def _log(msg: str):
    print(msg, flush=True)
    with LOG_LOCK:
        LOG_BUF.append(f"{datetime.now().isoformat(timespec='seconds')} | {msg}")

def _consume_logs() -> str:
    with LOG_LOCK:
        return "\n".join(LOG_BUF)

def _clear_logs():
    with LOG_LOCK:
        LOG_BUF.clear()
    _log("[UI] Logs cleared")

def _set_reset_row(i: int, pulses: int = 1):
    with RESET_LOCK:
        RESET_ROWS[int(i)] = max(RESET_ROWS.get(int(i), 0), pulses)

def _take_reset_rows() -> set:
    to_reset = set()
    with RESET_LOCK:
        for idx, cnt in list(RESET_ROWS.items()):
            if cnt > 0:
                to_reset.add(idx)
                RESET_ROWS[idx] = cnt - 1
            if RESET_ROWS[idx] <= 0:
                del RESET_ROWS[idx]
    return to_reset

# ===== Dwell utilities =====
def dwell_until(robot: Robot, target: Dict[str, Any], radius_m: float, dwell_s: float, stop_event: Optional[threading.Event] = None) -> bool:
    start = None
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        if stop_event and stop_event.is_set():
            _log("[Dwell] stopped.")
            return False
        try:
            curr = robot.get_curr_pos()
        except Exception:
            time.sleep(POLL_SEC); continue
        d = _distance(curr, target)
        ts = datetime.now().isoformat(timespec="seconds")
        cx, cy = _as_xy(curr)
        if d <= radius_m:
            if start is None:
                start = time.monotonic()
                _log(f"[Dwell] [{ts}] Enter zone d={d:.3f}m at ({cx:.3f},{cy:.3f})")
            elapsed = time.monotonic() - start
            _log(f"[Dwell] [{ts}] IN zone d={d:.3f}m | dwell={elapsed:.1f}s / {dwell_s:.1f}s")
            if elapsed >= dwell_s:
                _log(f"[Dwell] [{ts}] dwell satisfied ({dwell_s:.1f}s).")
                return True
        else:
            if start is not None:
                _log(f"[Dwell] [{ts}] left zone d={d:.3f}m — reset dwell timer.")
            start = None
        time.sleep(POLL_SEC)
    _log("[Dwell] timeout.")
    return False

def depart_then_dwell(robot: Robot, target: Dict[str, Any], radius_m: float, dwell_s: float, stop_event: Optional[threading.Event] = None) -> bool:
    while True:
        if stop_event and stop_event.is_set():
            _log("[Dwell] stopped during depart.")
            return False
        try:
            curr = robot.get_curr_pos()
        except Exception:
            time.sleep(POLL_SEC); continue
        if _distance(curr, target) > radius_m:
            break
        time.sleep(POLL_SEC)
    return dwell_until(robot, target, radius_m, dwell_s, stop_event)

# ===== FSM Runner =====
from enum import Enum, auto

class FSMState(Enum):
    IDLE = auto()
    ROW_START = auto()
    SUBMIT_A = auto()
    PREPARE_PULSE = auto()
    PULSE = auto()
    POST_PULSE_WAIT = auto()
    SUBMIT_B = auto()
    ROW_DONE = auto()
    FINISHED = auto()
    ABORTED = auto()

# ===== Round-trip gating =====

def _is_at(robot: Robot, target: Dict[str, Any], radius_m: float) -> bool:
    try:
        curr = robot.get_curr_pos()
        return _distance(curr, target) <= radius_m
    except Exception:
        return False

def _wait_until_depart(robot: Robot, point: Dict[str, Any], radius_m: float, timeout_s: float, stop_event: Optional[threading.Event] = None) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if stop_event and stop_event.is_set():
            _log("[Gate] cancel during depart wait")
            return False
        curr = robot.get_curr_pos()
        if not _is_at(robot, point, radius_m):
            _log(f"[Gate] Departed '{point.get('name','?')}'")
            return True
        time.sleep(POLL_SEC)
        _log(f"waiting to depart {point.get('name','?')}")
        _log(f"distance from {point.get('name','?')}:")
        _log(_distance(curr, point))
    _log(f"[Gate] depart timeout from '{point.get('name','?')}'")
    return False

def _wait_until_arrive(robot: Robot, point: Dict[str, Any], radius_m: float, timeout_s: float, stop_event: Optional[threading.Event] = None) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if stop_event and stop_event.is_set():
            _log("[Gate] cancel during arrive wait")
            return False
        curr = robot.get_curr_pos()
        if _is_at(robot, point, radius_m):
            _log(f"[Gate] Arrived at '{point.get('name','?')}'")
            return True
        time.sleep(POLL_SEC)
        _log(f"waiting to arrive at {point.get('name','?')}")
        _log(f"distance from {point.get('name','?')}:")
        _log(_distance(curr, point))
    _log(f"[Gate] arrive timeout to '{point.get('name','?')}'")
    return False

def wrapper_roundtrip_gate(robot: Robot,
                           waiting: Dict[str, Any],
                           wrapper: Dict[str, Any],
                           radius_m: float = ARRIVE_DIST_M,
                           timeout_s: float = ROUNDTRIP_TIMEOUT_S,
                           stop_event: Optional[threading.Event] = None) -> bool:
    """
    Require: leave WAITING -> (optionally hit WRAPPER) -> return to WAITING (each within radius).
    Returns True when the sequence completes; False on timeout or cancel.
    """
    _log("[Gate] Round-trip: waiting → wrapper → waiting")

    # 1) Leave waiting (if already away, this passes immediately)
    if not _wait_until_depart(robot, waiting, radius_m, timeout_s, stop_event):
        return False

    # 2) Hitting wrapper was previously optional/commented. Keep same behavior.
    # if not _wait_until_arrive(robot, wrapper, radius_m, timeout_s, stop_event):
    #     return False

    # 3) Return to waiting
    if not _wait_until_arrive(robot, waiting, radius_m, timeout_s, stop_event):
        return False

    _log("[Gate] Round-trip satisfied.")
    return True

class RowSpec:
    __slots__ = ("ui_idx","pickup","drop","wrapper","use_wrapper","post_pulse_dwell")
    def __init__(self, ui_idx:int, pickup:Dict[str,Any], drop:Dict[str,Any], wrapper:Optional[Dict[str,Any]], use_wrapper:bool, post_pulse_dwell:float=180.0):
        self.ui_idx = ui_idx
        self.pickup = pickup
        self.drop = drop
        self.wrapper = wrapper
        self.use_wrapper = use_wrapper
        self.post_pulse_dwell = post_pulse_dwell

class FSMRunner(threading.Thread):
    def __init__(self, robot_id: str, waiting_poi: Dict[str,Any], rows: List[RowSpec], on_exit=None):
        super().__init__(daemon=True)
        self.robot_id = robot_id
        self.waiting = waiting_poi
        self.rows = rows
        self.row_idx = 0
        self.state = FSMState.IDLE
        self._stop = threading.Event()
        self._on_exit = on_exit  # callback to clear global RUNNER

    def _create(self, rob: Robot, name: str, points: List[Dict[str,Any]]):
        body = {
            "task_name": name, "robot": rob.df, "runType": RUN_TYPE_LIFT, "sourceType": SOURCE_SDK,
            "taskPts": points, "runNum": 1, "taskType": TASK_TYPE_LIFT,
            "routeMode": ROUTE_SEQ, "runMode": RUNMODE_FLEX, "speed": 1.0,
            "detourRadius": 1.0, "ignorePublicSite": False, "backPt": back_pt(self.waiting)
        }
        _log(f"[FSM] create_task {name}")
        resp = create_task(**body)
        _log(f"[FSM] submitted: {name} → {resp}")
        return resp

    def stop(self): self._stop.set()

    def _cleanup(self):
        if callable(self._on_exit):
            try: self._on_exit()  # type: ignore
            except Exception: pass

    def run(self):
        try:
            _gpio_init()
            rob = get_robot()
            self.state = FSMState.ROW_START

            while not self._stop.is_set():
                if self.row_idx >= len(self.rows):
                    self.state = FSMState.FINISHED
                    _log("[FSM] Plan finished.")
                    return
                row = self.rows[self.row_idx]
                row_no = row.ui_idx + 1
                _log(f"[FSM] Row UI#{row_no} state={self.state.name}")

                if self.state == FSMState.ROW_START:
                    if self.row_idx > 0:
                        _log(f"[FSM] Row {self.row_idx+1} GATE: dwell {ROW_GATE_DWELL_SEC:.0f}s at '{self.waiting['name']}'")
                        if not dwell_until(rob, self.waiting, ARRIVE_DIST_M, ROW_GATE_DWELL_SEC, self._stop):
                            _log("[FSM] gate dwell failed → ABORTED"); self.state = FSMState.ABORTED; continue
                    self.state = FSMState.SUBMIT_A

                elif self.state == FSMState.SUBMIT_A:
                    try:
                        if row.use_wrapper:
                            from api_lib_v1 import Robot_v2, Task
                            temprob = Robot_v2(rob.SN)
                            name = f"r{self.row_idx+1}_A_{int(time.time())}"
                            ptts = [
                                pt(row.pickup, acts=[act_lift_up()]),
                                pt(row.wrapper, acts=[act_lift_down()]),
                                pt(self.waiting),
                            ]
                            task = Task(temprob, "area", taskType="factory",runType="lift").pickup(ptts[0]['ext']['name'], lift_up=True).pickup(ptts[1]['ext']['name'], lift_down=True).back("Warten")
                            print(task.task_dict)
                            resp = create_task(**task.task_dict)
                            self.state = FSMState.PREPARE_PULSE
                        else:
                            name = f"r{self.row_idx+1}_{int(time.time())}"
                            ptts = [
                                pt(row.pickup, acts=[act_lift_up()]),
                                pt(row.drop,   acts=[act_lift_down()]),
                            ]
                            from api_lib_v1 import Robot_v2, Task
                            temprob = Robot_v2(rob.SN)
                            task = Task(temprob, "area", taskType="factory",runType="lift").pickup(ptts[0]['ext']['name'], lift_up=True, areaDelivery=False).pickup(ptts[1]['ext']['name'], lift_down=True, areaDelivery=True).back("Warten")
                            resp = create_task(**task.task_dict)
                            _log(f"[FSM] Row {self.row_idx+1}: confirmation dwell {ROW_GATE_DWELL_SEC:.0f}s at waiting")
                            if not depart_then_dwell(rob, self.waiting, ARRIVE_DIST_M, ROW_GATE_DWELL_SEC, self._stop):
                                _log("[FSM] confirmation dwell failed → ABORTED"); self.state = FSMState.ABORTED; continue
                            self.state = FSMState.ROW_DONE
                    except Exception as e:
                        _log(f"[FSM] SUBMIT_A failed: {e}"); self.state = FSMState.ABORTED

                elif self.state == FSMState.PREPARE_PULSE:
                    row = self.rows[self.row_idx]
                    if not row.wrapper:
                        _log("[FSM] PREPARE_PULSE: wrapper POI missing → ABORTED")
                        self.state = FSMState.ABORTED
                        continue

                    _log(f"[FSM] Gate before pre-pulse: require waiting→wrapper→waiting (radius {ARRIVE_DIST_M} m)")
                    if not wrapper_roundtrip_gate(rob, self.waiting, row.wrapper, ARRIVE_DIST_M, ROUNDTRIP_TIMEOUT_S, self._stop):
                        _log("[FSM] Gate failed → ABORTED")
                        self.state = FSMState.ABORTED
                        continue

                    _log(f"[FSM] Pre-pulse dwell {PRE_PULSE_DWELL_S:.0f}s at waiting (gate passed)")
                    if dwell_until(rob, self.waiting, ARRIVE_DIST_M, PRE_PULSE_DWELL_S, self._stop):
                        self.state = FSMState.PULSE
                    else:
                        _log("[FSM] pre-pulse dwell failed → ABORTED")
                        self.state = FSMState.ABORTED

                elif self.state == FSMState.PULSE:
                    _pulse_gpio()
                    self.state = FSMState.POST_PULSE_WAIT

                elif self.state == FSMState.POST_PULSE_WAIT:
                    row = self.rows[self.row_idx]
                    post_pulse_s = row.post_pulse_dwell if hasattr(row, 'post_pulse_dwell') else POST_PULSE_DWELL_S
                    _log(f"[FSM] Post-pulse dwell {post_pulse_s:.0f}s at waiting")
                    if dwell_until(rob, self.waiting, ARRIVE_DIST_M, post_pulse_s, self._stop):
                        self.state = FSMState.SUBMIT_B
                    else:
                        _log("[FSM] post-pulse dwell failed → ABORTED"); self.state = FSMState.ABORTED

                elif self.state == FSMState.SUBMIT_B:
                    try:
                        name = f"r{self.row_idx+1}_B_{int(time.time())}"
                        ptts = [
                            pt(row.wrapper, acts=[act_lift_up()]),
                            pt(row.drop,    acts=[act_lift_down()]),
                        ]
                        from api_lib_v1 import Robot_v2, Task
                        temprob = Robot_v2(rob.SN)
                        task = Task(temprob, "area", taskType="factory",runType="lift").pickup(ptts[0]['ext']['name'], lift_up=True, areaDelivery=True).pickup(ptts[1]['ext']['name'], lift_down=True, areaDelivery=True).back("Warten")
                        print(task.task_dict)
                        resp = create_task(**task.task_dict)
                        _log(f"[FSM] Row {self.row_idx+1}: confirmation dwell {ROW_GATE_DWELL_SEC:.0f}s at waiting")
                        if not depart_then_dwell(rob, self.waiting, ARRIVE_DIST_M, ROW_GATE_DWELL_SEC, self._stop):
                            _log("[FSM] B completion dwell failed → ABORTED"); self.state = FSMState.ABORTED; continue
                        self.state = FSMState.ROW_DONE
                    except Exception as e:
                        _log(f"[FSM] SUBMIT_B failed: {e}"); self.state = FSMState.ABORTED

                elif self.state == FSMState.ROW_DONE:
                    _log(f"[FSM] Row {self.row_idx+1} done → request UI reset")
                    _set_reset_row(row.ui_idx)
                    self.row_idx += 1
                    if self.row_idx >= len(self.rows):
                        self.state = FSMState.FINISHED
                        _set_reset_row(row.ui_idx - 1)
                        _log("[FSM] Plan finished.")
                        return
                    self.state = FSMState.ROW_START

                elif self.state in (FSMState.ABORTED, FSMState.FINISHED):
                    return

                time.sleep(POLL_SEC)
        finally:
            self._cleanup()

# ===== Dash UI =====
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], suppress_callback_exceptions=True)
app.title = "Wrapper Control — Settings"

# Theme stylesheets
dbc_themes = {
    "dark": dbc.themes.DARKLY,
    "light": dbc.themes.FLATLY
}
app.index_string = """
<!DOCTYPE html>
<html>
<head>
  {%metas%}
  <title>{%title%}</title>
  {%favicon%}
  {%css%}
  <style id="theme-style">
    /* Default: Dark theme */
    body {
      background: #0e1218;
      color: #f8f9fb;
      transition: background 0.3s, color 0.3s;
    }

    .card, .alert, .btn, .form-control {
      border-radius: 8px;
    }

    /* Dropdown base (dcc.Dropdown -> react-select v1) */
    .dash-dropdown,
    .dash-dropdown .Select,
    .dash-dropdown .Select-control,
    .dash-dropdown .Select-menu-outer {
      background: #0f1724;
      color: #e9eef7;
      border-color: #2b3750;
      transition: background 0.3s, color 0.3s, border-color 0.3s;
    }

    .dash-dropdown .Select-option.is-focused,
    .dash-dropdown .Select-option:hover {
      background: #1b2842;
      color: #ffffff;
    }

    .dash-dropdown > div > div {
      border-color: #2b3750 !important;
    }

    /* Keep selected value text bright */
    .dash-dropdown .Select.has-value.Select--single > .Select-control .Select-value .Select-value-label {
      color: #e9eef7 !important;
    }

    /* Keep input text visible while typing */
    .dash-dropdown .Select-input > input {
      color: #e9eef7 !important;
    }

    /* Keep selected option visible in the open menu */
    .dash-dropdown .Select-option.is-selected {
      background: #1b2842;
      color: #ffffff !important;
    }

    /* Multi-select */
    .dash-dropdown .Select--multi .Select-value {
      background: #1b2842;
      border-color: #2b3750;
    }

    .dash-dropdown .Select--multi .Select-value-label {
      color: #e9eef7 !important;
    }

    /* Buttons */
    .btn-primary {
      background: #3b82f6;
      border-color: #3b82f6;
    }

    .btn-primary:hover {
      background: #2563eb;
      border-color: #2563eb;
    }

    /* Log window */
    pre {
      background: #0f1724;
      color: #dbe7ff;
      padding: 12px;
      border-radius: 8px;
      border: 1px solid #2b3750;
      transition: background 0.3s, color 0.3s, border-color 0.3s;
    }

    /* Light theme overrides */
    body.light-theme {
      background: #f8f9fa;
      color: #212529;
    }

    body.light-theme .dash-dropdown,
    body.light-theme .dash-dropdown .Select,
    body.light-theme .dash-dropdown .Select-control,
    body.light-theme .dash-dropdown .Select-menu-outer {
      background: #ffffff;
      color: #212529;
      border-color: #ced4da;
    }

    body.light-theme .dash-dropdown .Select-option.is-focused,
    body.light-theme .dash-dropdown .Select-option:hover {
      background: #e9ecef;
      color: #212529;
    }

    body.light-theme .dash-dropdown > div > div {
      border-color: #ced4da !important;
    }

    body.light-theme .dash-dropdown .Select.has-value.Select--single > .Select-control .Select-value .Select-value-label {
      color: #212529 !important;
    }

    body.light-theme .dash-dropdown .Select-input > input {
      color: #212529 !important;
    }

    body.light-theme .dash-dropdown .Select-option.is-selected {
      background: #dee2e6;
      color: #212529 !important;
    }

    body.light-theme .dash-dropdown .Select--multi .Select-value {
      background: #e9ecef;
      border-color: #ced4da;
    }

    body.light-theme .dash-dropdown .Select--multi .Select-value-label {
      color: #212529 !important;
    }

    body.light-theme pre {
      background: #ffffff;
      color: #212529;
      border: 1px solid #dee2e6;
    }

    /* Tab styling - Dark theme */
    .nav-tabs .nav-link {
      color: #e9eef7;
      background: #0f1724;
      border-color: #2b3750;
      transition: background 0.3s, color 0.3s, border-color 0.3s;
    }

    .nav-tabs .nav-link.active {
      color: #ffffff;
      background: #1b2842;
      border-color: #3b82f6;
    }

    .nav-tabs .nav-link:hover {
      color: #ffffff;
      background: #1b2842;
    }

    /* Tab styling - Light theme */
    body.light-theme .nav-tabs .nav-link {
      color: #495057;
      background: #f8f9fa;
      border-color: #dee2e6;
    }

    body.light-theme .nav-tabs .nav-link.active {
      color: #212529;
      background: #ffffff;
      border-color: #3b82f6;
    }

    body.light-theme .nav-tabs .nav-link:hover {
      color: #212529;
      background: #e9ecef;
    }
  </style>
</head>
<body>
  {%app_entry%}
  <footer>
    {%config%}
    {%scripts%}
    {%renderer%}
  </footer>
</body>
</html>
"""

def _opts(lst: List[str]) -> List[Dict[str,str]]:
    return [{"label": s, "value": s} for s in lst]

def row_ui(i: int) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dbc.Checklist(
                    id={"type":"include-ck","index":i},
                    options=[{"label":" Include","value":"on"}],
                    value=[], switch=True
                ), width=2),
                dbc.Col(dcc.Dropdown(
                    id={"type":"pickup-dd","index":i},
                    options=[], placeholder="Pickup (any POI)",
                    searchable=False, clearable=False, className="dash-dropdown"
                ), width=3),
                dbc.Col(dcc.Dropdown(
                    id={"type":"drop-dd","index":i},
                    options=[], placeholder="Drop (any Region)",
                    searchable=False, clearable=False, className="dash-dropdown"
                ), width=3),
                dbc.Col(dbc.Checklist(
                    id={"type":"wrapper-ck","index":i},
                    options=[{"label":" Wrapper","value":"on"}],
                    value=[], switch=True
                ), width=2),
                dbc.Col([
                    html.Label("Post-Pulse (s)", style={"fontSize": "0.85rem", "marginBottom": "2px"}),
                    dcc.Slider(
                        id={"type":"post-pulse-slider","index":i},
                        min=60, max=180, step=10, value=180,
                        marks={60: "60", 90: "90", 120: "120", 150: "150", 180: "180"},
                        disabled=True,
                        tooltip={"placement": "bottom", "always_visible": False}
                    )
                ], width=2),
            ], align="center")
        ]), className="mb-2"
    )

# Settings UI
def settings_ui() -> html.Div:
    current = _get_current_settings()
    return html.Div([
        dbc.Card(dbc.CardBody([
            html.H5("Threshold Settings", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    html.Label("Poll Interval (sec)"),
                    dbc.Input(id="setting-poll-sec", type="number", value=current["poll_sec"], step=0.01, min=0.01),
                ], width=6, className="mb-3"),
                dbc.Col([
                    html.Label("Arrive Distance (m)"),
                    dbc.Input(id="setting-arrive-dist", type="number", value=current["arrive_dist_m"], step=0.01, min=0.01),
                ], width=6, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Row Gate Dwell (sec)"),
                    dbc.Input(id="setting-row-gate-dwell", type="number", value=current["row_gate_dwell_sec"], step=0.5, min=0),
                ], width=6, className="mb-3"),
                dbc.Col([
                    html.Label("Pre-Pulse Dwell (sec)"),
                    dbc.Input(id="setting-pre-pulse-dwell", type="number", value=current["pre_pulse_dwell_s"], step=0.5, min=0),
                ], width=6, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Post-Pulse Dwell (sec)"),
                    dbc.Input(id="setting-post-pulse-dwell", type="number", value=current["post_pulse_dwell_s"], step=1, min=0),
                ], width=6, className="mb-3"),
                dbc.Col([
                    html.Label("Roundtrip Timeout (sec)"),
                    dbc.Input(id="setting-roundtrip-timeout", type="number", value=current["roundtrip_timeout_s"], step=10, min=0),
                ], width=6, className="mb-3"),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Pulse Duration (sec)"),
                    dbc.Input(id="setting-pulse-sec", type="number", value=current["pulse_sec"], step=0.1, min=0.1),
                ], width=6, className="mb-3"),
            ]),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    dbc.Button("Apply Settings", id="btn-apply-settings", color="success", className="me-2"),
                    dbc.Button("Save Settings", id="btn-save-settings", color="primary"),
                ], width="auto"),
            ]),
            html.Hr(),
            html.H5("Load Settings", className="mt-3 mb-3"),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id="settings-file-dropdown",
                        options=_opts(_get_settings_files()),
                        placeholder="Select settings file",
                        className="dash-dropdown"
                    ),
                ], width=8),
                dbc.Col([
                    dbc.Button("Load", id="btn-load-settings", color="secondary"),
                ], width="auto"),
            ]),
        ])),
        dbc.Alert(id="settings-alert", is_open=False, duration=4000, className="mt-3"),

        # Modals for save confirmation
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Save Settings")),
            dbc.ModalBody([
                html.Label("Enter filename (without .json):"),
                dbc.Input(id="save-filename-input", type="text", placeholder="my_settings"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Save", id="btn-confirm-save", color="primary", className="me-2"),
                dbc.Button("Cancel", id="btn-cancel-save", color="secondary"),
            ]),
        ], id="modal-save-filename", is_open=False),

        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("File Exists")),
            dbc.ModalBody(html.Div(id="overwrite-message")),
            dbc.ModalFooter([
                dbc.Button("Overwrite", id="btn-confirm-overwrite", color="danger", className="me-2"),
                dbc.Button("Cancel", id="btn-cancel-overwrite", color="secondary"),
            ]),
        ], id="modal-overwrite-confirm", is_open=False),

        dcc.Store(id="pending-save-filename"),
        dcc.Store(id="pending-save-data"),
    ])

app.layout = dbc.Container([
    dbc.Card(dbc.CardBody([
        dbc.Row([
           dbc.Col(html.Div("Robot ID:"), width="auto"),
           dbc.Col(dbc.Input(id="robot-id", value=DEFAULT_ROBOT_ID, placeholder="robotId", n_submit=0, type="text", disabled=True),
                   width=4),
           dbc.Col(dbc.Button("Load robot", id="btn-load", color="secondary", disabled=True), width="auto"),
           dbc.Col(html.Div(id="robot-hint", className="ms-3"), width=True),
           dbc.Col(dbc.Button("🌙 Toggle Theme", id="btn-toggle-theme", color="info", size="sm"), width="auto"),
        ], align="center"),
        html.Div(id="waiting-fixed", style={"textAlign":"right", "marginTop":"6px"}),

        dcc.Store(id="robot-state"),  # {"robot_id":..., "waiting": {...}}
        dcc.Store(id="theme-store", storage_type='session', data='dark'),  # Store theme in session
    ]), className="mb-3"),

    dbc.Tabs([
        dbc.Tab(label="Control", children=[
            html.Div([row_ui(i) for i in range(4)], className="mt-3"),
            dbc.Button("Start", id="btn-start", color="primary", className="my-3 me-2"),
            dbc.Button("Cancel", id="btn-back", color="danger", className="my-3"),
            dbc.Alert(id="result", color="info", is_open=False, duration=8000, className="mt-2"),
        ]),
        dbc.Tab(label="Settings", children=[
            html.Div(settings_ui(), className="mt-3")
        ]),
    ]),

    dcc.Interval(id="tick", interval=1000, n_intervals=0),

    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col(html.H5("Logs"), width="auto"),
            dbc.Col([
                dbc.Button("Clear Logs", id="btn-clear-logs", color="warning", size="sm", className="me-2"),
                dbc.Button("Toggle Auto-Scroll", id="btn-toggle-autoscroll", color="info", size="sm"),
                html.Span(id="autoscroll-status", className="ms-2", style={"color": "#4ade80"}),
            ], width="auto", className="ms-auto"),
        ], align="center", className="mb-2"),
        html.Div(
            html.Pre(id="log", style={"whiteSpace":"pre-wrap","height":"70vh","overflowY":"auto"}),
            id="log-container"
        )
    ]), className="mt-3"),

    dcc.Store(id="autoscroll-state", data=True)
], fluid=True)

# Keep a single runner at a time
RUNNER: Optional["FSMRunner"] = None
RUNNER_LOCK = threading.Lock()

def _runner_clear_global():
    global RUNNER
    with RUNNER_LOCK:
        RUNNER = None
    _log("[FSM] Runner cleared (exited).")

# ===== Callbacks =====

# Load robot → compute picks/drops/waiting and store in robot-state
@app.callback(
   Output("robot-state", "data"),
   Output("robot-hint", "children"),
   Output("waiting-fixed", "children"),
   *[Output({"type":"pickup-dd","index":i}, "options") for i in range(4)],
   *[Output({"type":"drop-dd","index":i}, "options") for i in range(4)],
   Input("btn-load", "n_clicks"),
   State("robot-id", "value"),
   prevent_initial_call=False
)
def on_load_robot(n, robot_id):
    rid = (robot_id or "").strip() or DEFAULT_ROBOT_ID
    _log(f"[UI] Load robot → '{rid}' (clicks={n})")
    try:
        set_robot(rid)
        picks = _pickups(rid)                   # pickup list = POIs
        drops = REGION_OPTIONS                  # drop list = region labels
        wait  = _find_waiting()
    except Exception as e:
        _log(f"[UI] load robot error: {e}")
        picks, drops, wait = [], [], None

    hint = f"Active robot: {rid}"
    waiting_text = f"Waiting point: {wait['name'] if wait else '(missing)'}"
    state = {"robot_id": rid, "waiting": wait}

    return state, hint, waiting_text, *([_opts(picks)]*4), *([_opts(drops)]*4)

from dash import ctx

@app.callback(
    Output("result", "children"),
    Output("result", "is_open"),
    Input("btn-start", "n_clicks"),
    Input("btn-back", "n_clicks"),
    State("robot-state", "data"),
    *[State({"type":"include-ck","index":i}, "value") for i in range(4)],
    *[State({"type":"pickup-dd","index":i}, "value") for i in range(4)],
    *[State({"type":"drop-dd","index":i}, "value") for i in range(4)],
    *[State({"type":"wrapper-ck","index":i}, "value") for i in range(4)],
    *[State({"type":"post-pulse-slider","index":i}, "value") for i in range(4)],
    prevent_initial_call=True
)
def handle_actions(n_start, n_back, rstate, *state):
    global RUNNER
    trigger = getattr(ctx, "triggered_id", None)
    if trigger is None:
        return no_update, False

    rid = (rstate or {}).get("robot_id") or DEFAULT_ROBOT_ID
    wait = (rstate or {}).get("waiting") or _find_waiting()
    set_robot(rid)  # ensure singleton matches

    if trigger == "btn-start":
        _log(f"[UI] Start Plan clicked for robot '{rid}'")
        if not wait:
            return "Waiting point not found.", True

        includes = state[0:4]
        pickups  = state[4:8]
        drops    = state[8:12]
        wchecks  = state[12:16]
        sliders  = state[16:20]
        wrapper_poi = _find_wrapper()

        rows: List[RowSpec] = []
        for i in range(4):
            if "on" not in (includes[i] or []):
                continue
            pick = _find_poi(pickups[i]) if pickups[i] else None
            resolved_drop_name = _resolve_region_to_poi_name(drops[i]) if drops[i] else None
            _log(f"resolved poi name: {resolved_drop_name}")
            drop = _find_poi(resolved_drop_name) if resolved_drop_name else None
            if drops[i] and not drop:
                _log(f"[UI] Drop region '{drops[i]}' could not be resolved to a POI")

            if not pick or not drop:
                continue
            use_wrapper = ("on" in (wchecks[i] or []))
            if use_wrapper and not wrapper_poi:
                continue
            post_pulse_dwell = float(sliders[i]) if sliders[i] is not None else POST_PULSE_DWELL_S
            rows.append(RowSpec(i, pick, drop, wrapper_poi, use_wrapper, post_pulse_dwell))

        if not rows:
            return "Nothing to run.", True

        with RUNNER_LOCK:
            if RUNNER and RUNNER.is_alive():
                _log("[UI] Runner already active.")
                return "Runner already active. Cancel or wait.", True
            RUNNER = FSMRunner(rid, wait, rows, on_exit=_runner_clear_global)
            _log(f"[UI] Runner starting with {len(rows)} row(s).")
            RUNNER.start()

        return "Plan started. Watch server logs.", True

    if trigger == "btn-back":
        _log(f"[UI] Cancel clicked for robot '{rid}'")
        # stop FSM thread if any
        with RUNNER_LOCK:
            if RUNNER and hasattr(RUNNER, "stop"):
                try:
                    RUNNER.stop()
                    rob = Robot(rid)
                    rob.cancel_task()
                    rob.back()
                    _log("[UI] Cancel: sent 'go Standby' task.")
                except Exception:
                    _log(f"[UI] Cancel: standby task failed: {e}")
                    pass

        # actively send the robot to Standby with a tiny task
        # try:
        #     if not wait:
        #         wait = _find_waiting()
        #     if wait:
        #         name = f"cancel_to_waiting_{int(time.time())}"
        #         body = [
        #             pt(wait, acts=[act_pause(1)]),  # tiny pause to be visible in logs
        #         ]
        #         create_task(
        #             task_name=name, robot=get_robot().df,
        #             runType=RUN_TYPE_LIFT, sourceType=SOURCE_SDK,
        #             taskPts=body, runNum=1, taskType=TASK_TYPE_LIFT,
        #             routeMode=ROUTE_SEQ, runMode=RUNMODE_FLEX, speed=1.0,
        #             detourRadius=1.0, ignorePublicSite=False, backPt=back_pt(wait)
        #         )
        #         _log("[UI] Cancel: sent 'go Standby' task.")
        # except Exception as e:
        #     _log(f"[UI] Cancel: standby task failed: {e}")

        return "Canceled current plan and sent robot to Standby.", True

    return no_update, False

# Settings callbacks
@app.callback(
    Output("settings-alert", "children"),
    Output("settings-alert", "is_open"),
    Output("settings-alert", "color"),
    Input("btn-apply-settings", "n_clicks"),
    State("setting-poll-sec", "value"),
    State("setting-arrive-dist", "value"),
    State("setting-row-gate-dwell", "value"),
    State("setting-pre-pulse-dwell", "value"),
    State("setting-post-pulse-dwell", "value"),
    State("setting-roundtrip-timeout", "value"),
    State("setting-pulse-sec", "value"),
    prevent_initial_call=True
)
def apply_settings(n, poll_sec, arrive_dist, row_gate, pre_pulse, post_pulse, roundtrip, pulse_sec):
    if n is None:
        return no_update, False, "info"

    settings = {
        "poll_sec": float(poll_sec),
        "arrive_dist_m": float(arrive_dist),
        "row_gate_dwell_sec": float(row_gate),
        "pre_pulse_dwell_s": float(pre_pulse),
        "post_pulse_dwell_s": float(post_pulse),
        "roundtrip_timeout_s": float(roundtrip),
        "pulse_sec": float(pulse_sec),
    }
    _apply_settings(settings)
    return "Settings applied successfully!", True, "success"

# Open save modal
@app.callback(
    Output("modal-save-filename", "is_open"),
    Output("pending-save-data", "data"),
    Input("btn-save-settings", "n_clicks"),
    Input("btn-cancel-save", "n_clicks"),
    State("setting-poll-sec", "value"),
    State("setting-arrive-dist", "value"),
    State("setting-row-gate-dwell", "value"),
    State("setting-pre-pulse-dwell", "value"),
    State("setting-post-pulse-dwell", "value"),
    State("setting-roundtrip-timeout", "value"),
    State("setting-pulse-sec", "value"),
    prevent_initial_call=True
)
def open_save_modal(n_save, n_cancel, poll_sec, arrive_dist, row_gate, pre_pulse, post_pulse, roundtrip, pulse_sec):
    trigger = getattr(ctx, "triggered_id", None)

    if trigger == "btn-save-settings":
        settings = {
            "poll_sec": float(poll_sec),
            "arrive_dist_m": float(arrive_dist),
            "row_gate_dwell_sec": float(row_gate),
            "pre_pulse_dwell_s": float(pre_pulse),
            "post_pulse_dwell_s": float(post_pulse),
            "roundtrip_timeout_s": float(roundtrip),
            "pulse_sec": float(pulse_sec),
        }
        return True, settings

    return False, no_update

# Handle save confirmation and overwrite check
@app.callback(
    Output("modal-overwrite-confirm", "is_open"),
    Output("overwrite-message", "children"),
    Output("pending-save-filename", "data"),
    Output("settings-alert", "children", allow_duplicate=True),
    Output("settings-alert", "is_open", allow_duplicate=True),
    Output("settings-alert", "color", allow_duplicate=True),
    Output("modal-save-filename", "is_open", allow_duplicate=True),
    Input("btn-confirm-save", "n_clicks"),
    Input("btn-cancel-overwrite", "n_clicks"),
    State("save-filename-input", "value"),
    State("pending-save-data", "data"),
    prevent_initial_call=True
)
def handle_save_confirmation(n_confirm, n_cancel_overwrite, filename, settings_data):
    trigger = getattr(ctx, "triggered_id", None)

    if trigger == "btn-cancel-overwrite":
        return False, no_update, no_update, no_update, False, "info", True

    if trigger == "btn-confirm-save" and filename:
        filename = filename.strip()
        if not filename:
            return False, no_update, no_update, "Please enter a valid filename", True, "warning", True

        if not filename.endswith(".json"):
            filename += ".json"

        # Check if file exists
        filepath = os.path.join(SETTINGS_DIR, filename)
        if os.path.exists(filepath):
            return True, f"File '{filename}' already exists. Do you want to overwrite it?", filename, no_update, False, "info", False

        # Save directly
        if _save_settings_to_file(filename, settings_data):
            return False, no_update, None, f"Settings saved to {filename}", True, "success", False
        else:
            return False, no_update, None, f"Failed to save settings to {filename}", True, "danger", False

    return False, no_update, no_update, no_update, False, "info", no_update

# Handle overwrite confirmation
@app.callback(
    Output("settings-alert", "children", allow_duplicate=True),
    Output("settings-alert", "is_open", allow_duplicate=True),
    Output("settings-alert", "color", allow_duplicate=True),
    Output("modal-overwrite-confirm", "is_open", allow_duplicate=True),
    Output("settings-file-dropdown", "options"),
    Input("btn-confirm-overwrite", "n_clicks"),
    State("pending-save-filename", "data"),
    State("pending-save-data", "data"),
    prevent_initial_call=True
)
def handle_overwrite(n_clicks, filename, settings_data):
    if n_clicks and filename and settings_data:
        if _save_settings_to_file(filename, settings_data):
            return f"Settings saved to {filename}", True, "success", False, _opts(_get_settings_files())
        else:
            return f"Failed to save settings to {filename}", True, "danger", False, no_update
    return no_update, False, "info", False, no_update

# Load settings
@app.callback(
    Output("setting-poll-sec", "value"),
    Output("setting-arrive-dist", "value"),
    Output("setting-row-gate-dwell", "value"),
    Output("setting-pre-pulse-dwell", "value"),
    Output("setting-post-pulse-dwell", "value"),
    Output("setting-roundtrip-timeout", "value"),
    Output("setting-pulse-sec", "value"),
    Output("settings-alert", "children", allow_duplicate=True),
    Output("settings-alert", "is_open", allow_duplicate=True),
    Output("settings-alert", "color", allow_duplicate=True),
    Input("btn-load-settings", "n_clicks"),
    State("settings-file-dropdown", "value"),
    prevent_initial_call=True
)
def load_settings(n_clicks, filename):
    if n_clicks and filename:
        settings = _load_settings_from_file(filename)
        if settings:
            return (
                settings.get("poll_sec", POLL_SEC),
                settings.get("arrive_dist_m", ARRIVE_DIST_M),
                settings.get("row_gate_dwell_sec", ROW_GATE_DWELL_SEC),
                settings.get("pre_pulse_dwell_s", PRE_PULSE_DWELL_S),
                settings.get("post_pulse_dwell_s", POST_PULSE_DWELL_S),
                settings.get("roundtrip_timeout_s", ROUNDTRIP_TIMEOUT_S),
                settings.get("pulse_sec", PULSE_SEC),
                f"Settings loaded from {filename}",
                True,
                "success"
            )
        else:
            return (no_update,) * 7 + (f"Failed to load settings from {filename}", True, "danger")
    return (no_update,) * 7 + (no_update, False, "info")

# Periodic log pump + row reset
@app.callback(
    Output("log", "children"),
    # Row 1
    Output({"type":"include-ck","index":0}, "value"),
    Output({"type":"pickup-dd","index":0}, "value"),
    Output({"type":"drop-dd","index":0}, "value"),
    Output({"type":"wrapper-ck","index":0}, "value"),
    # Row 2
    Output({"type":"include-ck","index":1}, "value"),
    Output({"type":"pickup-dd","index":1}, "value"),
    Output({"type":"drop-dd","index":1}, "value"),
    Output({"type":"wrapper-ck","index":1}, "value"),
    # Row 3
    Output({"type":"include-ck","index":2}, "value"),
    Output({"type":"pickup-dd","index":2}, "value"),
    Output({"type":"drop-dd","index":2}, "value"),
    Output({"type":"wrapper-ck","index":2}, "value"),
    # Row 4
    Output({"type":"include-ck","index":3}, "value"),
    Output({"type":"pickup-dd","index":3}, "value"),
    Output({"type":"drop-dd","index":3}, "value"),
    Output({"type":"wrapper-ck","index":3}, "value"),
    Input("tick", "n_intervals"),
    prevent_initial_call=False
)
def tick(_n):
    log_text = _consume_logs()
    resets = _take_reset_rows()

    outs: List[Any] = []
    def cleared():
        return ([], None, None, [])

    for i in range(4):
        if i in resets:
            outs.extend(cleared())
        else:
            outs.extend([no_update, no_update, no_update, no_update])

    return (log_text, *outs)

# Refresh settings file dropdown periodically
@app.callback(
    Output("settings-file-dropdown", "options", allow_duplicate=True),
    Input("tick", "n_intervals"),
    prevent_initial_call=True
)
def refresh_settings_dropdown(_n):
    return _opts(_get_settings_files())

# Enable/disable post-pulse sliders based on wrapper checkbox
@app.callback(
    Output({"type":"post-pulse-slider","index":0}, "disabled"),
    Output({"type":"post-pulse-slider","index":1}, "disabled"),
    Output({"type":"post-pulse-slider","index":2}, "disabled"),
    Output({"type":"post-pulse-slider","index":3}, "disabled"),
    Input({"type":"wrapper-ck","index":0}, "value"),
    Input({"type":"wrapper-ck","index":1}, "value"),
    Input({"type":"wrapper-ck","index":2}, "value"),
    Input({"type":"wrapper-ck","index":3}, "value"),
    prevent_initial_call=False
)
def toggle_post_pulse_sliders(w0, w1, w2, w3):
    # Enable slider (disabled=False) when wrapper is checked ("on" in list)
    # Disable slider (disabled=True) when wrapper is not checked
    return (
        "on" not in (w0 or []),  # row 0: disabled if wrapper NOT checked
        "on" not in (w1 or []),  # row 1: disabled if wrapper NOT checked
        "on" not in (w2 or []),  # row 2: disabled if wrapper NOT checked
        "on" not in (w3 or []),  # row 3: disabled if wrapper NOT checked
    )

# Clear logs callback
@app.callback(
    Output("log", "children", allow_duplicate=True),
    Input("btn-clear-logs", "n_clicks"),
    prevent_initial_call=True
)
def clear_logs(n_clicks):
    if n_clicks:
        _clear_logs()
        return ""
    return no_update

# Toggle autoscroll callback and update status
@app.callback(
    Output("autoscroll-state", "data"),
    Output("autoscroll-status", "children"),
    Input("btn-toggle-autoscroll", "n_clicks"),
    Input("autoscroll-state", "data"),
    prevent_initial_call=False
)
def toggle_autoscroll(n_clicks, current_state):
    trigger = getattr(ctx, "triggered_id", None)

    # If triggered by button click
    if trigger == "btn-toggle-autoscroll" and n_clicks:
        new_state = not current_state
        status_text = "✓ ON" if new_state else "✗ OFF"
        return new_state, status_text

    # Initial load or state change
    status_text = "✓ ON" if current_state else "✗ OFF"
    return current_state, status_text

# Autoscroll logs to bottom (conditionally based on state)
app.clientside_callback(
    """
    function(log_content, autoscroll_enabled) {
        if (autoscroll_enabled) {
            setTimeout(function() {
                var logElement = document.getElementById('log');
                if (logElement) {
                    logElement.scrollTop = logElement.scrollHeight;
                }
            }, 50);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("log-container", "style"),
    Input("log", "children"),
    State("autoscroll-state", "data")
)

# Theme toggle callback
app.clientside_callback(
    """
    function(n_clicks, current_theme) {
        if (n_clicks) {
            // Toggle theme
            var new_theme = current_theme === 'dark' ? 'light' : 'dark';

            // Update stylesheet
            var links = document.getElementsByTagName('link');
            for (var i = 0; i < links.length; i++) {
                if (links[i].href.includes('bootstrap')) {
                    if (new_theme === 'dark') {
                        links[i].href = 'https://cdn.jsdelivr.net/npm/bootswatch@5/dist/darkly/bootstrap.min.css';
                    } else {
                        links[i].href = 'https://cdn.jsdelivr.net/npm/bootswatch@5/dist/flatly/bootstrap.min.css';
                    }
                }
            }

            // Toggle body class for custom styles
            if (new_theme === 'light') {
                document.body.classList.add('light-theme');
            } else {
                document.body.classList.remove('light-theme');
            }

            return new_theme;
        }
        return current_theme || 'dark';
    }
    """,
    Output("theme-store", "data"),
    Input("btn-toggle-theme", "n_clicks"),
    State("theme-store", "data"),
    prevent_initial_call=False
)

# ===== Run =====
if __name__ == "__main__":
    set_robot(DEFAULT_ROBOT_ID)
    app.run(
        host="0.0.0.0",
        port=8050,
        debug=False,
        threaded=True,
        use_reloader=False
    )
