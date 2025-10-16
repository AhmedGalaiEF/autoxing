#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, time, threading
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
from api_lib_v1 import *   # <-- uses Robot_v2 for "Return"

# ===== App / Robot config =====
DEFAULT_ROBOT_ID = "FS52505505633sR"   # <-- hardcoded
WAITING_POI_NAME = "Wartepunkt"

# Motion / actions (unchanged opcodes, keep your backend mapping)
RUN_TYPE_LIFT   = 29
TASK_TYPE_LIFT  = 5
SOURCE_SDK      = 6
ROUTE_SEQ       = 1
RUNMODE_FLEX    = 1
ACTION          = {"lift_up": 47, "lift_down": 48}

def act_pause(seconds: int = 5) -> Dict[str, Any]:
    return {"type": 18, "data": {"pauseTime": int(seconds)}}

# ===== Defaults (now overridable via Settings tab) =====
# Per your request:
# - input reset (15s -> 5s)      -> ROW_GATE_DWELL_SEC
# - pre GPIO pulse (15s -> 5s)   -> PRE_PULSE_DWELL_S
# - post GPIO pulse (10s -> 180s)-> POST_PULSE_DWELL_S
DEFAULTS = {
    "POLL_SEC": 0.5,
    "ARRIVE_DIST_M": 0.15,       # lowered threshold; prefer isAt when available
    "ROW_GATE_DWELL_SEC": 5.0,   # "input reset"
    "PRE_PULSE_DWELL_S": 5.0,    # "pre GPIO pulse"
    "POST_PULSE_DWELL_S": 180.0  # "post GPIO pulse"
}

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
# Regex: include "Sichtlager"
RX_PICKUP = re.compile(r"^Abhol\s*\d+$", re.IGNORECASE)
RX_SICHT  = re.compile(r"^(Sicht|Sichtlager)\s*\d+$", re.IGNORECASE)   # <-- adjusted
RX_EURO   = re.compile(r"^Euro\s*\d+$",  re.IGNORECASE)
RX_DIV    = re.compile(r"^Div\s*\d+$",   re.IGNORECASE)

# --- Single robot instance (no multiple initializations) ---
_ROBOT_LOCK = threading.Lock()
_ROBOT_ID: str = DEFAULT_ROBOT_ID
_ROBOT: Optional[Robot_v2] = None

def get_robot() -> Robot_v2:
    global _ROBOT
    with _ROBOT_LOCK:
        if _ROBOT is None:
            _ROBOT = Robot_v2(_ROBOT_ID)
        return _ROBOT

def set_robot(robot_id: str):
    global _ROBOT_ID, _ROBOT
    with _ROBOT_LOCK:
        if robot_id != _ROBOT_ID or _ROBOT is None:
            _ROBOT_ID = robot_id
            _ROBOT = Robot_v2(robot_id)

# --- Preload POIs once and keep cached in UI store ---
def _poi_df() -> pd.DataFrame:
    try:
        r = get_robot()
        df = r.get_pois()
        print(df)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def _norm_poi(row: pd.Series | Dict[str, Any]) -> Dict[str, Any]:
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

# ===== Dwell utilities (prefer isAt when available) =====
def _is_at(robot: Robot_v2, target: Dict[str, Any], radius_m: float) -> bool:
    # Prefer robot.get_state().get('isAt') if provided by SDK
    try:
        st = robot.get_state()
        if isinstance(st, dict) and "isAt" in st:
            # If SDK exposes isAt with point id/name, try to match quickly
            is_at = bool(st["isAt"])
            if is_at:
                # if ext id matches, accept; otherwise fall back to distance
                return True
    except Exception:
        pass
    try:
        curr = robot.get_curr_pos()
        return _distance(curr, target) <= radius_m
    except Exception:
        return False

def dwell_until(robot: Robot_v2, target: Dict[str, Any], radius_m: float, dwell_s: float, poll: float) -> bool:
    start = None
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        if _is_at(robot, target, radius_m):
            if start is None:
                start = time.monotonic()
                _log(f"[Dwell] Enter zone at '{target.get('name')}'")
            elapsed = time.monotonic() - start
            _log(f"[Dwell] IN zone | dwell={elapsed:.1f}s / {dwell_s:.1f}s")
            if elapsed >= dwell_s:
                _log(f"[Dwell] dwell satisfied ({dwell_s:.1f}s).")
                return True
        else:
            if start is not None:
                _log("[Dwell] left zone — reset dwell timer.")
            start = None
        time.sleep(poll)
    _log("[Dwell] timeout.")
    return False

def depart_then_dwell(robot: Robot_v2, target: Dict[str, Any], radius_m: float, dwell_s: float, poll: float) -> bool:
    # depart
    while True:
        try:
            curr = robot.get_curr_pos()
            if _distance(curr, target) > radius_m:
                break
        except Exception:
            pass
        time.sleep(poll)
    # dwell after re-enter
    return dwell_until(robot, target, radius_m, dwell_s, poll)

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

class RowSpec:
    __slots__ = ("ui_idx","pickup","drop","wrapper","use_wrapper")
    def __init__(self, ui_idx:int, pickup:Dict[str,Any], drop:Dict[str,Any], wrapper:Optional[Dict[str,Any]], use_wrapper:bool):
        self.ui_idx = ui_idx
        self.pickup = pickup
        self.drop = drop
        self.wrapper = wrapper
        self.use_wrapper = use_wrapper

class FSMRunner(threading.Thread):
    def __init__(self, robot_id: str, waiting_poi: Dict[str,Any], rows: List[RowSpec], settings: Dict[str, float], on_exit=None):
        super().__init__(daemon=True)
        self.robot_id = robot_id
        self.waiting = waiting_poi
        self.rows = rows
        self.row_idx = 0
        self.state = FSMState.IDLE
        self._stop = threading.Event()
        self._on_exit = on_exit  # callback to clear global RUNNER
        # settings
        self.poll = float(settings["POLL_SEC"])
        self.arrive = float(settings["ARRIVE_DIST_M"])
        self.gate_dwell = float(settings["ROW_GATE_DWELL_SEC"])
        self.pre_pulse = float(settings["PRE_PULSE_DWELL_S"])
        self.post_pulse = float(settings["POST_PULSE_DWELL_S"])

    def _create(self, rob: Robot_v2, name: str, points: List[Dict[str,Any]]):
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
            try: self._on_exit()
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
                        _log(f"[FSM] Row {self.row_idx+1} GATE: dwell {self.gate_dwell:.0f}s at '{self.waiting['name']}'")
                        if not dwell_until(rob, self.waiting, self.arrive, self.gate_dwell, self.poll):
                            _log("[FSM] gate dwell failed → ABORTED"); self.state = FSMState.ABORTED; continue
                    self.state = FSMState.SUBMIT_A

                elif self.state == FSMState.SUBMIT_A:
                    try:
                        if row.use_wrapper:
                            name = f"r{self.row_idx+1}_A_{int(time.time())}"
                            self._create(rob, name, [
                                pt(row.pickup, acts=[act_lift_up()]),
                                pt(row.wrapper, acts=[act_lift_down()]),
                                pt(self.waiting),
                            ])
                            self.state = FSMState.PREPARE_PULSE
                        else:
                            name = f"r{self.row_idx+1}_{int(time.time())}"
                            self._create(rob, name, [
                                pt(row.pickup, acts=[act_lift_up()]),
                                pt(row.drop,   acts=[act_lift_down()]),
                            ])
                            _log(f"[FSM] Row {self.row_idx+1}: confirmation dwell {self.gate_dwell:.0f}s at waiting")
                            if not depart_then_dwell(rob, self.waiting, self.arrive, self.gate_dwell, self.poll):
                                _log("[FSM] confirmation dwell failed → ABORTED"); self.state = FSMState.ABORTED; continue
                            self.state = FSMState.ROW_DONE
                    except Exception as e:
                        _log(f"[FSM] SUBMIT_A failed: {e}"); self.state = FSMState.ABORTED

                elif self.state == FSMState.PREPARE_PULSE:
                    _log(f"[FSM] Pre-pulse dwell {self.pre_pulse:.0f}s at waiting")
                    if dwell_until(rob, self.waiting, self.arrive, self.pre_pulse, self.poll):
                        self.state = FSMState.PULSE
                    else:
                        _log("[FSM] pre-pulse dwell failed → ABORTED"); self.state = FSMState.ABORTED

                elif self.state == FSMState.PULSE:
                    _pulse_gpio()
                    self.state = FSMState.POST_PULSE_WAIT

                elif self.state == FSMState.POST_PULSE_WAIT:
                    _log(f"[FSM] Post-pulse dwell {self.post_pulse:.0f}s at waiting")
                    if dwell_until(rob, self.waiting, self.arrive, self.post_pulse, self.poll):
                        self.state = FSMState.SUBMIT_B
                    else:
                        _log("[FSM] post-pulse dwell failed → ABORTED"); self.state = FSMState.ABORTED

                elif self.state == FSMState.SUBMIT_B:
                    try:
                        name = f"r{self.row_idx+1}_B_{int(time.time())}"
                        self._create(rob, name, [
                            pt(row.wrapper, acts=[act_lift_up()]),
                            pt(row.drop,    acts=[act_lift_down()]),
                        ])
                        _log(f"[FSM] Row {self.row_idx+1}: confirmation dwell {self.gate_dwell:.0f}s at waiting")
                        if not depart_then_dwell(rob, self.waiting, self.arrive, self.gate_dwell, self.poll):
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

                time.sleep(self.poll)
        finally:
            self._cleanup()

# ===== Dash UI =====
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Wrapper Control — High Contrast"
app.index_string = """
<!DOCTYPE html>
<html>
<head>
  {%metas%}
  <title>{%title%}</title>
  {%favicon%}
  {%css%}
  <style>
    body { background:#0e1218; color:#f8f9fb; }
    .card, .alert, .btn, .form-control { border-radius: 8px; }
    .Select, .Select-control, .Select-menu-outer { background:#0f1724; color:#e9eef7; border-color:#2b3750; }
    .Select-option.is-focused, .Select-option:hover { background:#1b2842; color:#ffffff; }
    .dash-dropdown>div>div { border-color:#2b3750 !important; }
    .btn-primary { background:#3b82f6; border-color:#3b82f6; }
    .btn-primary:hover { background:#2563eb; border-color:#2563eb; }
    pre { background:#0f1724; color:#dbe7ff; padding:12px; border-radius:8px; border:1px solid #2b3750; }
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
                    options=[], placeholder="Drop (any POI)",
                    searchable=False, clearable=False, className="dash-dropdown"
                ), width=5),
                dbc.Col(dbc.Checklist(
                    id={"type":"wrapper-ck","index":i},
                    options=[{"label":" Wrapper","value":"on"}],
                    value=[], switch=True
                ), width=2),
            ])
        ]), className="mb-2"
    )

# === Settings UI ===
def settings_ui() -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H5("Settings"),
        dbc.Row([
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Arrive distance (m)"),
                dbc.Input(id="set-arrive", type="number", step=0.01, value=DEFAULTS["ARRIVE_DIST_M"])
            ]), md=6),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Poll (s)"),
                dbc.Input(id="set-poll", type="number", step=0.1, value=DEFAULTS["POLL_SEC"])
            ]), md=6),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Input reset dwell (s)"),
                dbc.Input(id="set-gate", type="number", step=1, value=DEFAULTS["ROW_GATE_DWELL_SEC"])
            ]), md=4),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Pre GPIO dwell (s)"),
                dbc.Input(id="set-pre", type="number", step=1, value=DEFAULTS["PRE_PULSE_DWELL_S"])
            ]), md=4),
            dbc.Col(dbc.InputGroup([
                dbc.InputGroupText("Post GPIO dwell (s)"),
                dbc.Input(id="set-post", type="number", step=1, value=DEFAULTS["POST_PULSE_DWELL_S"])
            ]), md=4),
        ], className="mb-2"),
        dbc.Button("Apply Settings", id="btn-apply", color="secondary"),
        html.Div(id="settings-hint", className="mt-2", style={"opacity":0.85})
    ]))

app.layout = dbc.Container([
    dcc.Store(id="robot-state"),     # {"robot_id":..., "waiting": {...}}
    dcc.Store(id="poi-cache"),       # {"picks": [...], "drops":[...]}
    dcc.Store(id="settings", data=DEFAULTS.copy()),

    dbc.Tabs([
        dbc.Tab(label="Control", tab_id="tab-control", children=[
            html.Div([row_ui(i) for i in range(4)]),
            dbc.Button("Start Plan", id="btn-start", color="primary", className="my-3 me-2"),
            dbc.Button("Cancel Task", id="btn-cancel", color="warning", className="my-3 me-2"),
            dbc.Button("Return to Standby", id="btn-return", color="danger", className="my-3"),
            dbc.Alert(id="result", color="info", is_open=False, duration=8000, className="mt-2"),
            dcc.Interval(id="tick", interval=1000, n_intervals=0),
            dbc.Card(dbc.CardBody([html.Pre(id="log", style={"whiteSpace":"pre-wrap","maxHeight":"40vh","overflowY":"auto"})]), className="mt-3")
        ]),
        dbc.Tab(label="Settings", tab_id="tab-settings", children=[settings_ui()])
    ], active_tab="tab-control"),
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
from dash import ctx

# Preload robot + POIs at startup
@app.callback(
    Output("robot-state", "data"),
    Output("poi-cache", "data"),
    Input("tick", "n_intervals"),
    State("robot-state", "data"),
    prevent_initial_call=False
)
def preload_robot(_n, rstate):
    if rstate:  # already loaded
        return no_update, no_update
    set_robot(DEFAULT_ROBOT_ID)
    _log(f"[Init] Active robot: {DEFAULT_ROBOT_ID}")
    picks = _all_poi_names()            # full list for pickup
    # drops filtered by Sicht/Sichtlager/Euro/Div plus Wrapper if exists
    df = _poi_df()
    names = [str(n) for n in df["name"].astype(str).tolist()] if not df.empty else []
    drops_pool = set()
    for n in names:
        if RX_SICHT.fullmatch(n) or RX_EURO.fullmatch(n) or RX_DIV.fullmatch(n):
            drops_pool.add(n)
    if any(n.lower() == "wrapper" for n in names):
        drops_pool.add("Wrapper")
    drops = sorted(drops_pool)
    wait  = _find_waiting()
    return {"robot_id": DEFAULT_ROBOT_ID, "waiting": wait}, {"picks": picks, "drops": drops}

# Fill dropdown options when POIs cached
@app.callback(
    *[Output({"type":"pickup-dd","index":i}, "options") for i in range(4)],
    *[Output({"type":"drop-dd","index":i}, "options") for i in range(4)],
    Input("poi-cache", "data"),
    prevent_initial_call=True
)
def fill_dd(cache):
    if not cache:
        return *([[]]*4), *([[]]*4)
    return *([_opts(cache.get("picks", []))]*4), *([_opts(cache.get("drops", []))]*4)

# Apply settings
@app.callback(
    Output("settings", "data"),
    Output("settings-hint", "children"),
    Input("btn-apply", "n_clicks"),
    State("settings", "data"),
    State("set-arrive", "value"),
    State("set-poll", "value"),
    State("set-gate", "value"),
    State("set-pre", "value"),
    State("set-post", "value"),
    prevent_initial_call=True
)
def on_apply_settings(_n, curr, arrive, poll, gate, pre, post):
    s = curr.copy() if curr else DEFAULTS.copy()
    try:
        if arrive is not None: s["ARRIVE_DIST_M"] = float(arrive)
        if poll   is not None: s["POLL_SEC"] = float(poll)
        if gate   is not None: s["ROW_GATE_DWELL_SEC"] = float(gate)
        if pre    is not None: s["PRE_PULSE_DWELL_S"] = float(pre)
        if post   is not None: s["POST_PULSE_DWELL_S"] = float(post)
        return s, "Settings applied."
    except Exception as e:
        return no_update, f"Failed to apply: {e}"

# Handle Start / Cancel / Return
@app.callback(
    Output("result", "children"),
    Output("result", "is_open"),
    Input("btn-start", "n_clicks"),
    Input("btn-cancel", "n_clicks"),
    Input("btn-return", "n_clicks"),
    State("robot-state", "data"),
    State("settings", "data"),
    *[State({"type":"include-ck","index":i}, "value") for i in range(4)],
    *[State({"type":"pickup-dd","index":i}, "value") for i in range(4)],
    *[State({"type":"drop-dd","index":i}, "value") for i in range(4)],
    *[State({"type":"wrapper-ck","index":i}, "value") for i in range(4)],
    prevent_initial_call=True
)
def handle_actions(n_start, n_cancel, n_return, rstate, settings, *state):
    global RUNNER
    trigger = getattr(ctx, "triggered_id", None)
    if trigger is None:
        return no_update, False

    rid = (rstate or {}).get("robot_id") or DEFAULT_ROBOT_ID
    wait = (rstate or {}).get("waiting") or _find_waiting()
    set_robot(rid)  # ensure singleton matches

    # START PLAN
    if trigger == "btn-start":
        _log(f"[UI] Start Plan for robot '{rid}'")
        if not wait:
            return "Waiting point not found.", True

        includes = state[0:4]
        pickups  = state[4:8]
        drops    = state[8:12]
        wchecks  = state[12:16]
        wrapper_poi = _find_wrapper()

        rows: List[RowSpec] = []
        for i in range(4):
            if "on" not in (includes[i] or []):
                continue
            pick = _find_poi(pickups[i]) if pickups[i] else None
            drop = _find_poi(drops[i])   if drops[i]   else None
            if not pick or not drop:
                continue
            use_wrapper = ("on" in (wchecks[i] or []))
            if use_wrapper and not wrapper_poi:
                continue
            rows.append(RowSpec(i, pick, drop, wrapper_poi, use_wrapper))

        if not rows:
            return "Nothing to run.", True

        with RUNNER_LOCK:
            if RUNNER and RUNNER.is_alive():
                _log("[UI] Runner already active.")
                return "Runner already active. Cancel first.", True
            RUNNER = FSMRunner(rid, wait, rows, settings or DEFAULTS, on_exit=_runner_clear_global)
            _log(f"[UI] Runner starting with {len(rows)} row(s).")
            RUNNER.start()

        return "Plan started. Watch server logs.", True

    # CANCEL TASK (separate)
    if trigger == "btn-cancel":
        _log(f"[UI] Cancel Task for robot '{rid}'")
        # stop FSM thread if any
        with RUNNER_LOCK:
            if RUNNER and hasattr(RUNNER, "stop"):
                try: RUNNER.stop()
                except Exception: pass
        # cancel current robot task via SDK
        try:
            get_robot().cancel_task()
            _log("[UI] cancel_task() called.")
            return "Current task canceled; runner stopped.", True
        except Exception as e:
            _log(f"[UI] cancel_task failed: {e}")
            return f"Cancel failed: {e}", True

    # RETURN TO STANDBY (separate)
    if trigger == "btn-return":
        _log(f"[UI] Return to Standby for robot '{rid}'")
        # don't implicitly cancel; this is a direct return command using Robot_v2
        try:
            r2 = Robot_v2(rid)
            r2.go_back()
            _log("[UI] Robot_v2.go_back() called.")
            return "Return to Standby triggered.", True
        except Exception as e:
            _log(f"[UI] go_back failed: {e}")
            # fallback: tiny task to the waiting point
            try:
                wait = wait or _find_waiting()
                if wait:
                    name = f"return_to_waiting_{int(time.time())}"
                    body = [pt(wait, acts=[act_pause(1)])]
                    create_task(
                        task_name=name, robot=get_robot().df,
                        runType=RUN_TYPE_LIFT, sourceType=SOURCE_SDK,
                        taskPts=body, runNum=1, taskType=TASK_TYPE_LIFT,
                        routeMode=ROUTE_SEQ, runMode=RUNMODE_FLEX, speed=1.0,
                        detourRadius=1.0, ignorePublicSite=False, backPt=back_pt(wait)
                    )
                    _log("[UI] Fallback: created task to Waiting.")
                    return "Return fallback: sent task to Waiting.", True
            except Exception as e2:
                _log(f"[UI] Return fallback failed: {e2}")
                return f"Return failed: {e}", True

    return no_update, False

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

# ===== Run =====
def main():
    set_robot(DEFAULT_ROBOT_ID)
    app.run(
        host="0.0.0.0",
        port=8050,
        debug=False,
        threaded=True,
        use_reloader=False
    )
if __name__ == "__main__":
    # preload default robot once
    main()
