#!/usr/bin/env python3
import sys
import math
import uuid
import json
import logging

import numpy as np
import pandas as pd
import requests

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os, sys
sys.path.append(os.path.join(os.getcwd(),"lib"))

# --------------------------------------------------------------------
# Import your API lib (the huge file you pasted, with Robot = Robot_v2)
# --------------------------------------------------------------------
from api_lib import (
    Robot,
    get_online_robots,
    get_map_meta,
    normalize_map_meta,
    world_to_pixel,
    pixel_to_world,
    _astar_on_cost,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# --------------------------------------------------------------------
# Ollama config
# --------------------------------------------------------------------
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "command-r7b"

# --------------------------------------------------------------------
# Base cost map / meta
# --------------------------------------------------------------------
def get_base_cost_map(robot: Robot):
    """
    Wrap Robot_v2.get_env to produce a base costmap (float32 0..1) and map meta.
    """
    cost = robot.get_env(return_uint8=False)  # float32 [0,1], HxW

    area_id = robot.df.areaId
    meta_raw = get_map_meta(area_id=area_id, robot_sn=robot.SN)
    meta, _ = normalize_map_meta(meta_raw)

    if "res_m_per_px" not in meta:
        raise RuntimeError("map meta missing res_m_per_px")
    if "origin_x_m" not in meta or "origin_y_m" not in meta:
        raise RuntimeError("map meta missing origin_x_m/origin_y_m")

    H, W = cost.shape
    meta.setdefault("pixel_w", W)
    meta.setdefault("pixel_h", H)

    return cost.astype(np.float32), meta


# --------------------------------------------------------------------
# Dummy sandbox robot
# --------------------------------------------------------------------
class DummyRobot:
    """
    Sandbox robot that:
      - lives on the same costmap grid as the real robot's map
      - uses the same A* (_astar_on_cost) for planning
      - mimics a small subset of Robot_v2 API (go_to_poi, pickup, dropdown, etc.)
    """

    def __init__(self, cost: np.ndarray, meta: dict, real_robot: Robot):
        self.cost = cost.astype(np.float32)
        self.meta = dict(meta)
        self.real_robot = real_robot

        self.H, self.W = self.cost.shape
        # start roughly at center of free space
        self.col = self.W // 2
        self.row = self.H // 2
        self.theta = 0.0
        self.status = "IDLE"
        self.path_rc = []  # list[(row, col)]
        self.carrying = False

    # ---- coordinate transforms ----
    def _world_to_rc(self, x_m: float, y_m: float):
        H = self.H
        ox, oy = self.meta["origin_x_m"], self.meta["origin_y_m"]
        res = self.meta["res_m_per_px"]
        rot = float(self.meta.get("rotation_deg", 0.0))
        px, py = world_to_pixel(
            x_m,
            y_m,
            origin_x_m=ox,
            origin_y_m=oy,
            res_m_per_px=res,
            img_h_px=H,
            rotation_deg=rot,
        )
        c = int(round(px))
        r = int(round(py))
        c = max(0, min(self.W - 1, c))
        r = max(0, min(self.H - 1, r))
        return r, c

    def _rc_to_world(self, r: int, c: int):
        H = self.H
        ox, oy = self.meta["origin_x_m"], self.meta["origin_y_m"]
        res = self.meta["res_m_per_px"]
        rot = float(self.meta.get("rotation_deg", 0.0))
        x_m, y_m = pixel_to_world(
            float(c),
            float(r),
            origin_x_m=ox,
            origin_y_m=oy,
            res_m_per_px=res,
            img_h_px=H,
            rotation_deg=rot,
        )
        return x_m, y_m

    # ---- planning ----
    def _plan_to_rc(self, goal_rc):
        start_rc = (self.row, self.col)
        path_rc = _astar_on_cost(self.cost, start_rc, goal_rc, block_threshold=0.99)
        self.path_rc = path_rc or []
        if path_rc:
            self.row, self.col = path_rc[-1]
            self.status = f"MOVED({goal_rc[1]}, {goal_rc[0]})"
        else:
            self.status = "NO_PATH"

    def _plan_to_world(self, x_m, y_m):
        goal_rc = self._world_to_rc(x_m, y_m)
        self._plan_to_rc(goal_rc)

    # ---- public API ----
    def get_state(self):
        x_m, y_m = self._rc_to_world(self.row, self.col)
        return {
            "pose_world": (x_m, y_m, self.theta),
            "pose_grid": (self.row, self.col),
            "status": self.status,
            "path_rc": list(self.path_rc),
            "carrying": self.carrying,
        }

    def go_to_poi(self, poi_name: str):
        pois = self.real_robot.get_pois()
        if isinstance(pois, pd.DataFrame) and not pois.empty:
            target = pois[pois["name"].astype(str).str.strip() == str(poi_name).strip()]
            if target.empty:
                self.status = f"POI_NOT_FOUND({poi_name})"
                return
            coord = target.iloc[0]["coordinate"]
            x_m, y_m = float(coord[0]), float(coord[1])
            self.status = f"GO_TO_POI({poi_name})"
            self._plan_to_world(x_m, y_m)
        else:
            self.status = "NO_POIS"

    def go_charge(self):
        pois = self.real_robot.get_pois()
        if not isinstance(pois, pd.DataFrame) or pois.empty:
            self.status = "NO_POIS"
            return
        chargers = pois[pois["type"] == 9]
        if chargers.empty:
            self.status = "NO_CHARGER_POI"
            return
        name = chargers.iloc[0]["name"]
        self.go_to_poi(name)

    def go_back(self):
        pois = self.real_robot.get_pois()
        if not isinstance(pois, pd.DataFrame) or pois.empty:
            self.status = "NO_POIS"
            return
        standby = pois[pois["type"] == 10]
        if standby.empty:
            self.status = "NO_STANDBY_POI"
            return
        name = standby.iloc[0]["name"]
        self.go_to_poi(name)

    def pickup_at_poi(self, poi_name: str):
        self.go_to_poi(poi_name)
        if "NO_" in self.status or "POI_NOT_FOUND" in self.status:
            return
        self.carrying = True
        self.status = "PICKUP_DONE"

    def dropdown_at_poi(self, poi_name: str):
        if not self.carrying:
            self.status = "NO_CARRYING"
            return
        self.go_to_poi(poi_name)
        if "NO_" in self.status or "POI_NOT_FOUND" in self.status:
            return
        self.carrying = False
        self.status = "DROPDOWN_DONE"

    def stop(self):
        self.path_rc = []
        self.status = "STOPPED"


# --------------------------------------------------------------------
# Task + Ollama
# --------------------------------------------------------------------
TASK_ACTIONS = {"GO_TO_POI", "PICKUP", "DROPDOWN", "GO_CHARGE", "GO_BACK", "STOP"}
TASK_TRIGGERS = {"immediate", "on_pickup_done", "on_dropdown_done"}


def call_ollama_controller(history):
    """
    history: list[{"role":"user"|"assistant","content":str}]
    returns dict: {"reply": str, "tasks":[{id, action, target, trigger}]}
    """
    schema = {
        "type": "object",
        "properties": {
            "reply": {"type": "string"},
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "action": {"type": "string"},
                        "target": {"type": "string"},
                        "trigger": {"type": "string"},
                    },
                    "required": ["action"],
                },
            },
        },
        "required": ["reply", "tasks"],
    }

    system_prompt = (
        "You control a simulated warehouse robot (DummyRobot) via an asynchronous task queue.\n\n"
        "You must ALWAYS respond with pure JSON (no markdown) matching this schema:\n"
        "{\n"
        '  \"reply\": \"natural language message to the user\",\n'
        '  \"tasks\": [\n'
        "    {\n"
        '      \"id\": \"optional unique string id\",\n'
        '      \"action\": \"GO_TO_POI | PICKUP | DROPDOWN | GO_CHARGE | GO_BACK | STOP\",\n'
        '      \"target\": \"optional POI name\",\n'
        '      \"trigger\": \"immediate | on_pickup_done | on_dropdown_done\"\n'
        "    }, ...\n"
        "  ]\n"
        "}\n\n"
        "Semantics:\n"
        "- GO_TO_POI: move to a named POI (target is POI name).\n"
        "- PICKUP: go to POI and mark carrying=TRUE, status=PICKUP_DONE.\n"
        "- DROPDOWN: go to POI and mark carrying=FALSE, status=DROPDOWN_DONE (only if currently carrying).\n"
        "- GO_CHARGE: go to a charging POI.\n"
        "- GO_BACK: go to standby/waiting POI.\n"
        "- STOP: stop immediately (clear path).\n\n"
        "Triggers:\n"
        "- immediate: run as soon as possible.\n"
        "- on_pickup_done: execute when a pickup has completed (status=PICKUP_DONE).\n"
        "- on_dropdown_done: execute when a dropdown has completed (status=DROPDOWN_DONE).\n\n"
        "You are allowed to push multiple tasks; e.g., first a PICKUP with immediate, and a DROPDOWN with trigger=on_pickup_done.\n"
        "If user just chats, you can send an empty tasks list.\n"
        "Never output anything other than JSON."
    )

    messages = [{"role": "system", "content": system_prompt}]
    for m in history[-12:]:
        role = m.get("role", "user")
        content = str(m.get("content", ""))
        if not content:
            continue
        messages.append({"role": role, "content": content})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "format": schema,
    }

    try:
        # shorter timeout so UI isn't frozen forever if Ollama is down
        resp = requests.post(OLLAMA_URL, json=payload, timeout=1000)
        resp.raise_for_status()
        data = resp.json()
        content = data["message"]["content"]
        if isinstance(content, dict):
            parsed = content
        else:
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = {"reply": "I could not produce valid JSON.", "tasks": []}
        if "reply" not in parsed:
            parsed["reply"] = ""
        if "tasks" not in parsed:
            parsed["tasks"] = []
        return parsed
    except Exception as e:
        log.exception("Ollama call failed")
        return {"reply": f"(model error: {e})", "tasks": []}


def run_task_scheduler(queue, dummy_state_before, dummy: DummyRobot):
    """
    queue: list of task dicts
    dummy_state_before: last dummy.get_state()
    Applies at most one non-STOP task per tick (but STOP can interrupt).
    Returns (new_queue, new_dummy_state)
    """
    queue = list(queue or [])

    # STOP wins immediately
    stop_idx = next((i for i, t in enumerate(queue) if t["action"] == "STOP"), None)
    if stop_idx is not None:
        t = queue.pop(stop_idx)
        dummy.stop()
        log.info("Executed STOP from %s", t.get("source"))
        return queue, dummy.get_state()

    if not queue:
        return queue, dummy_state_before

    current_status = dummy_state_before.get("status", "")
    for i, t in enumerate(queue):
        action = t["action"]
        trigger = t.get("trigger", "immediate")
        target = t.get("target")

        if trigger == "immediate":
            pass
        elif trigger == "on_pickup_done":
            if current_status != "PICKUP_DONE":
                continue
        elif trigger == "on_dropdown_done":
            if current_status != "DROPDOWN_DONE":
                continue
        else:
            continue

        log.info(
            "Executing queued task: %s %s (trigger=%s, source=%s)",
            action,
            target,
            trigger,
            t.get("source"),
        )

        if action == "GO_TO_POI":
            if target:
                dummy.go_to_poi(target)
        elif action == "GO_CHARGE":
            dummy.go_charge()
        elif action == "GO_BACK":
            dummy.go_back()
        elif action == "PICKUP":
            if target:
                dummy.pickup_at_poi(target)
        elif action == "DROPDOWN":
            if target:
                dummy.dropdown_at_poi(target)
        else:
            pass

        queue.pop(i)
        return queue, dummy.get_state()

    return queue, dummy_state_before


# --------------------------------------------------------------------
# Matplotlib canvas (3D, decimated)
# --------------------------------------------------------------------
class CostmapCanvas(FigureCanvas):
    def __init__(self, cost: np.ndarray, dummy_state: dict, real_state: dict | None):
        fig = Figure()
        super().__init__(fig)
        self.ax = fig.add_subplot(111, projection="3d")
        self._cost = cost
        self.update_scene(cost, dummy_state, real_state)

    def update_scene(self, cost: np.ndarray, dummy_state: dict, real_state: dict | None):
        self.ax.clear()
        self._cost = cost

        H, W = cost.shape

        # decimate for plotting to avoid freezing on large maps
        step_y = max(H // 80, 1)
        step_x = max(W // 80, 1)
        ys = np.arange(0, H, step_y)
        xs = np.arange(0, W, step_x)
        X, Y = np.meshgrid(xs, ys)
        Z = cost[::step_y, ::step_x]

        self.ax.plot_surface(
            X, Y, Z,
            rstride=1,
            cstride=1,
            cmap="viridis",
            linewidth=0,
            antialiased=False,
        )

        # dummy path
        path_rc = dummy_state.get("path_rc") or []
        if len(path_rc) >= 2:
            px = [c for (r, c) in path_rc]
            py = [r for (r, c) in path_rc]
            pz = [cost[r, c] + 0.05 for (r, c) in path_rc]
            self.ax.plot3D(px, py, pz, color="blue", linewidth=2)

        # dummy marker
        try:
            row, col = dummy_state["pose_grid"]
            self.ax.scatter(
                [col], [row], [cost[row, col] + 0.1],
                color="blue", s=30, label="Dummy robot"
            )
        except Exception:
            pass

        self.ax.set_xlabel("Grid X")
        self.ax.set_ylabel("Grid Y")
        self.ax.set_zlabel("Cost")
        self.ax.set_zlim(0, max(1.0, float(cost.max()) + 0.1))
        self.ax.view_init(elev=45, azim=-135)
        self.draw_idle()


# --------------------------------------------------------------------
# Login dialog
# --------------------------------------------------------------------
class LoginDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Login")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        self.user_edit = QtWidgets.QLineEdit()
        self.pass_edit = QtWidgets.QLineEdit()
        self.pass_edit.setEchoMode(QtWidgets.QLineEdit.Password)
        self.error_label = QtWidgets.QLabel("")
        self.error_label.setStyleSheet("color: red;")

        btn = QtWidgets.QPushButton("Login")
        btn.clicked.connect(self.on_login)

        layout.addWidget(QtWidgets.QLabel("Username"))
        layout.addWidget(self.user_edit)
        layout.addWidget(QtWidgets.QLabel("Password"))
        layout.addWidget(self.pass_edit)
        layout.addWidget(self.error_label)
        layout.addWidget(btn)

        self.resize(300, 150)

    def on_login(self):
        u = self.user_edit.text().strip()
        p = self.pass_edit.text().strip()
        if u == "admin" and p == "admin":
            self.accept()
        else:
            self.error_label.setText("Invalid credentials")


# --------------------------------------------------------------------
# Background worker for real robot polling
# --------------------------------------------------------------------
class RealRobotWorker(QtCore.QThread):
    stateUpdated = QtCore.pyqtSignal(dict)

    def __init__(self, rob: Robot, parent=None):
        super().__init__(parent)
        self.rob = rob
        self._stop_flag = False

    def run(self):
        while not self._stop_flag:
            try:
                st = self.rob.get_state()
                if not isinstance(st, dict):
                    st = {"info": str(st)}
                self.stateUpdated.emit(st)
            except Exception as e:
                log.exception("rob.get_state() failed in worker")
                self.stateUpdated.emit({"error": str(e)})
            # poll every 3 seconds
            self.msleep(3000)

    def stop(self):
        self._stop_flag = True


# --------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, rob: Robot, dummy: DummyRobot, base_cost: np.ndarray, base_meta: dict):
        super().__init__()
        self.setWindowTitle("Robot Sandbox (PyQt)")

        self.rob = rob
        self.dummy = dummy
        self.base_cost = base_cost
        self.base_meta = base_meta

        # state
        self.queue = []
        self.history = []
        self.last_real_state: dict | None = None

        # central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)

        # canvas
        self.canvas = CostmapCanvas(base_cost, dummy.get_state(), None)
        hbox.addWidget(self.canvas, stretch=3)

        # sidebar
        side = QtWidgets.QWidget()
        side_layout = QtWidgets.QVBoxLayout(side)
        hbox.addWidget(side, stretch=1)

        monofont = QtGui.QFont("Consolas", 9)

        self.real_label = QtWidgets.QLabel()
        self.real_label.setFont(monofont)
        self.real_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.real_label.setText("Real robot: (pending)")
        self.real_label.setMinimumHeight(80)

        self.dummy_label = QtWidgets.QLabel()
        self.dummy_label.setFont(monofont)
        self.dummy_label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self.dummy_label.setMinimumHeight(80)

        side_layout.addWidget(QtWidgets.QLabel("<b>Real robot state</b>"))
        side_layout.addWidget(self.real_label)
        side_layout.addWidget(QtWidgets.QLabel("<b>Dummy robot state</b>"))
        side_layout.addWidget(self.dummy_label)

        side_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))

        # POI dropdown + buttons
        pois_df = self.rob.get_pois()
        self.poi_combo = QtWidgets.QComboBox()
        if isinstance(pois_df, pd.DataFrame) and not pois_df.empty:
            for _, row in pois_df.iterrows():
                label = f"{row['name']} (type {row['type']})"
                self.poi_combo.addItem(label, userData=row["name"])
        side_layout.addWidget(QtWidgets.QLabel("<b>Sandbox controls</b>"))
        side_layout.addWidget(self.poi_combo)

        self.btn_go_poi = QtWidgets.QPushButton("Go to POI (sandbox)")
        self.btn_go_charge = QtWidgets.QPushButton("Go charge (sandbox)")
        self.btn_go_back = QtWidgets.QPushButton("Go back / standby (sandbox)")

        side_layout.addWidget(self.btn_go_poi)
        side_layout.addWidget(self.btn_go_charge)
        side_layout.addWidget(self.btn_go_back)

        side_layout.addWidget(QtWidgets.QLabel(
            "Buttons enqueue immediate tasks for DummyRobot."
        ))

        side_layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))

        # Chat UI
        side_layout.addWidget(QtWidgets.QLabel("<b>Ollama Chatbot</b>"))

        self.chat_log = QtWidgets.QTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_log.setFont(monofont)
        self.chat_log.setMinimumHeight(150)

        self.chat_input = QtWidgets.QPlainTextEdit()
        self.chat_input.setFixedHeight(60)

        self.btn_chat_send = QtWidgets.QPushButton("Send to Ollama")

        side_layout.addWidget(self.chat_log)
        side_layout.addWidget(self.chat_input)
        side_layout.addWidget(self.btn_chat_send)
        side_layout.addWidget(QtWidgets.QLabel(
            "Model can enqueue tasks with triggers (immediate / on_*_done)."
        ))

        side_layout.addStretch(1)

        # signals
        self.btn_go_poi.clicked.connect(self.on_go_poi)
        self.btn_go_charge.clicked.connect(self.on_go_charge)
        self.btn_go_back.clicked.connect(self.on_go_back)
        self.btn_chat_send.clicked.connect(self.on_chat_send)

        # real-robot worker thread
        self.worker = RealRobotWorker(self.rob, self)
        self.worker.stateUpdated.connect(self.on_real_state_updated)
        self.worker.start()

        # timer for dummy scheduler + repaint
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)  # ms
        self.timer.timeout.connect(self.on_timer_tick)
        self.timer.start()

        self.on_timer_tick()

    # --------- manual buttons ----------
    def _new_task_id(self):
        return str(uuid.uuid4())

    def on_go_poi(self):
        idx = self.poi_combo.currentIndex()
        if idx < 0:
            return
        poi_name = self.poi_combo.itemData(idx)
        self.queue.append({
            "id": self._new_task_id(),
            "action": "GO_TO_POI",
            "target": poi_name,
            "trigger": "immediate",
            "source": "manual",
        })

    def on_go_charge(self):
        self.queue.append({
            "id": self._new_task_id(),
            "action": "GO_CHARGE",
            "target": None,
            "trigger": "immediate",
            "source": "manual",
        })

    def on_go_back(self):
        self.queue.append({
            "id": self._new_task_id(),
            "action": "GO_BACK",
            "target": None,
            "trigger": "immediate",
            "source": "manual",
        })

    # --------- chat ----------
    def on_chat_send(self):
        text = self.chat_input.toPlainText().strip()
        if not text:
            return

        self.history.append({"role": "user", "content": text})

        result = call_ollama_controller(self.history)
        reply = str(result.get("reply", "") or "")
        tasks_raw = result.get("tasks") or []

        self.history.append({"role": "assistant", "content": reply})

        for t in tasks_raw:
            try:
                action = str(t.get("action", "")).upper()
            except Exception:
                continue
            if action not in TASK_ACTIONS:
                continue
            trigger = str(t.get("trigger", "immediate")).lower()
            if trigger not in TASK_TRIGGERS:
                trigger = "immediate"
            target = t.get("target")
            self.queue.append({
                "id": t.get("id") or self._new_task_id(),
                "action": action,
                "target": target,
                "trigger": trigger,
                "source": "llm",
            })

        self.chat_input.clear()
        self._update_chat_log()

    def _update_chat_log(self):
        lines = []
        for m in self.history[-40:]:
            role = m.get("role", "user")
            prefix = "User" if role == "user" else "Assistant"
            lines.append(f"{prefix}: {m.get('content', '')}")
        self.chat_log.setPlainText("\n".join(lines))
        self.chat_log.verticalScrollBar().setValue(
            self.chat_log.verticalScrollBar().maximum()
        )

    # --------- real robot worker signal ----------
    @QtCore.pyqtSlot(dict)
    def on_real_state_updated(self, state: dict):
        self.last_real_state = state
        self._update_real_label(state)

    # --------- timer tick ----------
    def on_timer_tick(self):
        dummy_state_before = self.dummy.get_state()
        self.queue, dummy_state = run_task_scheduler(self.queue, dummy_state_before, self.dummy)

        # use last_real_state from worker thread (non-blocking)
        real_state = self.last_real_state or {"info": "Polling..."}

        # update canvas
        self.canvas.update_scene(self.base_cost, dummy_state, real_state)

        # update dummy label
        self._update_dummy_label(dummy_state)

        # keep chat log synced
        self._update_chat_log()

    def _update_real_label(self, real_state: dict):
        if isinstance(real_state, dict):
            lines = []
            if "error" in real_state:
                lines.append(f"ERROR: {real_state['error']}")
            else:
                lines.append(f"isOnLine   : {real_state.get('isOnLine')}")
                lines.append(f"isCharging : {real_state.get('isCharging')}")
                if "pose" in real_state:
                    try:
                        x, y, yaw = real_state["pose"]
                        lines.append(
                            f"pose       : x={float(x):.3f}, y={float(y):.3f}, yaw={float(yaw):.3f}"
                        )
                    except Exception:
                        lines.append(f"pose       : {real_state['pose']}")
                if "battery" in real_state:
                    lines.append(f"battery    : {real_state['battery']}")
                if real_state.get("task") is not None:
                    lines.append(f"task       : {real_state['task']}")
            self.real_label.setText("\n".join(lines))
        else:
            self.real_label.setText(str(real_state))

    def _update_dummy_label(self, dummy_state: dict):
        try:
            row, col = dummy_state["pose_grid"]
            xw, yw, _ = dummy_state["pose_world"]
            path_len = len(dummy_state.get("path_rc") or [])
            carrying = dummy_state.get("carrying")
        except Exception:
            row = col = 0
            xw = yw = 0.0
            path_len = 0
            carrying = False

        txt = (
            f"Dummy status: {dummy_state.get('status')}\n"
            f"grid pose   : row={row}, col={col}\n"
            f"world pose  : x={xw:.3f}, y={yw:.3f}\n"
            f"carrying    : {carrying}\n"
            f"path length : {path_len} cells\n"
            f"queue size  : {len(self.queue)}"
        )
        self.dummy_label.setText(txt)

    def closeEvent(self, event):
        # stop worker thread cleanly
        try:
            self.worker.stop()
            self.worker.wait(2000)
        except Exception:
            pass
        super().closeEvent(event)


# --------------------------------------------------------------------
# main
# --------------------------------------------------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)

    # login
    dlg = LoginDialog()
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        sys.exit(0)

    # real robot
    online = get_online_robots()
    if online.empty:
        QtWidgets.QMessageBox.critical(None, "Error", "No online robots found.")
        sys.exit(1)

    rob = Robot(online.iloc[0].robotId)
    log.info("Using robot: %s", rob)

    base_cost, base_meta = get_base_cost_map(rob)
    dummy = DummyRobot(base_cost, base_meta, rob)

    win = MainWindow(rob, dummy, base_cost, base_meta)
    win.resize(1400, 800)
    win.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

