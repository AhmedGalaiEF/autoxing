# ws_ext.py
# -*- coding: utf-8 -*-
"""
AutoXing WebSocket extension covering the Oversee stream and common doc entries.

- Endpoint (overseas): wss://serviceglobal.autoxing.com/robot-control/oversee/<robotId>
- Auth: SHORT token 'key' as WebSocket subprotocol (Sec-WebSocket-Protocol).
- Heartbeat: {"reqType":"onHeartBeat"} every 5 seconds (app-level).
- Subscriptions: robotState, taskState, issue (extensible).
- Queries: getRobotState, getTaskState, getTaskList (helpers provided).
- Control/Tasks: helper methods craft frames (commented in tests by default).

Notes:
- Proxies are disabled around the socket by default to avoid "scheme http is invalid".
- If your node is different (China/private), change DEFAULT_WS_BASE accordingly.
"""

from __future__ import annotations
import json, logging, os, ssl, threading, time, traceback
from typing import Any, Callable, Dict, Optional, List
from urllib.parse import urlparse
from importlib import import_module

try:
    import websocket  # pip install websocket-client
except ImportError as e:
    raise RuntimeError("Missing dependency: pip install websocket-client") from e

log = logging.getLogger(__name__)
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# --- Configuration ---
DEFAULT_WS_BASE = "wss://serviceglobal.autoxing.com"
OVERSEE_PATH = "/robot-control/oversee/{robotId}"  # confirmed working on your side
HEARTBEAT_SECONDS = 5

# Try to auto-discover get_token_key() from the host project
_DISCOVERED: Dict[str, Any] = {}
for name in ("api_lib", "__main__"):
    try:
        m = import_module(name)
    except Exception:
        continue
    if m:
        _DISCOVERED["get_token_key"] = getattr(m, "get_token_key", None)
        break

def _default_token_key_provider() -> str:
    gk = _DISCOVERED.get("get_token_key")
    if callable(gk):
        return gk()
    raise RuntimeError(
        "Provide token_key_provider=... returning the SHORT token 'key' from the auth API."
    )

def _normalize_ws_base(url_ws_base: str) -> str:
    u = urlparse(url_ws_base)
    if u.scheme in ("ws", "wss"):
        return url_ws_base.rstrip("/")
    if u.scheme == "http":
        return ("ws://" + url_ws_base[len("http://"):]).rstrip("/")
    if u.scheme == "https":
        return ("wss://" + url_ws_base[len("https://"):]).rstrip("/")
    if not u.scheme:
        return "wss://" + url_ws_base.rstrip("/").lstrip("/")
    raise ValueError(f"Unsupported URL scheme: {u.scheme}")

class _Heartbeat(threading.Thread):
    daemon = True
    def __init__(self, wsapp_getter: Callable[[], Optional[websocket.WebSocketApp]], stop_event: threading.Event):
        super().__init__(name="WS-Heartbeat")
        self._get = wsapp_getter
        self._stop = stop_event
    def run(self):
        payload = json.dumps({"reqType": "onHeartBeat"})
        while not self._stop.is_set():
            wsapp = self._get()
            try:
                if wsapp and wsapp.sock and wsapp.sock.connected:
                    wsapp.send(payload)
            except Exception:
                pass
            self._stop.wait(HEARTBEAT_SECONDS)

class _WSWorker(threading.Thread):
    daemon = True
    def __init__(
        self,
        *,
        robot_id: str,
        token_key_provider: Callable[[], str],
        url_ws_base: str = DEFAULT_WS_BASE,
        use_env_proxies: bool = False,
        # auto-subscribe topics (override in subscribe_oversee):
        subscribe_types: Optional[List[str]] = None,
    ):
        super().__init__(name=f"WS-oversee-{robot_id[-6:]}")
        self.robot_id = robot_id
        self.token_key_provider = token_key_provider
        self.url_ws_base = _normalize_ws_base(url_ws_base)
        self.use_env_proxies = use_env_proxies
        self._stop = threading.Event()
        self._wsapp: Optional[websocket.WebSocketApp] = None
        self._hb: Optional[_Heartbeat] = None
        self._subscribe_types = subscribe_types or ["robotState", "taskState", "issue"]

        # user callbacks
        self.on_message: Optional[Callable[[dict], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_open: Optional[Callable[[], None]] = None
        self.on_close: Optional[Callable[[Optional[int], Optional[str]], None]] = None

    # ------------- basics -------------
    def _build_url(self) -> str:
        return f"{self.url_ws_base}{OVERSEE_PATH.format(robotId=self.robot_id)}"

    def stop(self):
        self._stop.set()
        try:
            if self._wsapp:
                self._wsapp.close()
        except Exception:
            pass

    def send(self, obj: dict):
        """Generic send (dict -> JSON)."""
        try:
            if self._wsapp and self._wsapp.sock and self._wsapp.sock.connected:
                self._wsapp.send(json.dumps(obj))
        except Exception:
            logging.exception("WS send failed")

    # ------------- helpers for common doc entries -------------
    # Subscriptions
    def subscribe(self, types: List[str]):
        self.send({"reqType": "onSubscribe", "data": {"robotId": self.robot_id, "types": types}})

    # Queries
    def get_robot_state(self):
        self.send({"reqType": "getRobotState", "data": {"robotId": self.robot_id}})

    def get_task_state(self):
        self.send({"reqType": "getTaskState", "data": {"robotId": self.robot_id}})

    def get_task_list(self, page: int = 1, pageSize: int = 20):
        self.send({"reqType": "getTaskList", "data": {"robotId": self.robot_id, "page": page, "pageSize": pageSize}})

    # Control (commented in tests; here as helpers)
    def control_pause(self):
        self.send({"reqType": "onControl", "data": {"robotId": self.robot_id, "action": "pause"}})

    def control_resume(self):
        self.send({"reqType": "onControl", "data": {"robotId": self.robot_id, "action": "resume"}})

    def control_stop(self):
        self.send({"reqType": "onControl", "data": {"robotId": self.robot_id, "action": "stop"}})

    def control_go_home(self):
        self.send({"reqType": "onControl", "data": {"robotId": self.robot_id, "action": "goHome"}})

    def control_nav_to(self, x: float = None, y: float = None, poiId: str = None, speed: float = None):
        """
        Navigate either by XY or by poiId. Supply whichever your server expects.
        """
        payload = {"robotId": self.robot_id}
        if poiId is not None:
            payload["poiId"] = poiId
        if x is not None and y is not None:
            payload["x"] = x; payload["y"] = y
        if speed is not None:
            payload["speed"] = speed
        self.send({"reqType": "onNavigate", "data": payload})

    # Task operations (commented in tests; here as helpers)
    def task_create(self, name: str, runType: int, taskPts: list, **kwargs):
        """
        Create/execute a task (shape varies per deployment).
        Example kwargs you might add: taskType, routeMode, runMode, speed, detourRadius, backPt.
        """
        data = {"robotId": self.robot_id, "name": name, "runType": runType, "taskPts": taskPts}
        data.update(kwargs)
        self.send({"reqType": "onTaskCreate", "data": data})

    def task_cancel(self, taskId: str):
        self.send({"reqType": "onTaskCancel", "data": {"robotId": self.robot_id, "taskId": taskId}})

    def task_execute(self, taskId: str):
        self.send({"reqType": "onTaskExecute", "data": {"robotId": self.robot_id, "taskId": taskId}})

    # ------------- WS callbacks -------------
    def _cb_message(self, ws, message: str):
        try:
            data = json.loads(message)
        except Exception:
            data = {"_raw": message}
        if self.on_message:
            try: self.on_message(data)
            except Exception: logging.exception("on_message handler crashed")

    def _cb_error(self, ws, err):
        if self.on_error:
            try: self.on_error(err)
            except Exception: logging.exception("on_error handler crashed")
        else:
            logging.error("[oversee] WS error: %s", err)

    def _cb_open(self, ws):
        if self.on_open:
            try: self.on_open()
            except Exception: logging.exception("on_open handler crashed")
        logging.info("[oversee] WS open")

        # start heartbeat thread
        if self._hb is None or not self._hb.is_alive():
            self._hb = _Heartbeat(lambda: self._wsapp, self._stop)
            self._hb.start()

        # auto-subscribe to desired topics + initial queries
        try:
            self.subscribe(self._subscribe_types)
            # prime the stream with explicit queries to get immediate payloads
            self.get_robot_state()
            self.get_task_state()
        except Exception:
            logging.exception("initial subscribe failed")

    def _cb_close(self, ws, status_code, msg):
        if self.on_close:
            try: self.on_close(status_code, msg)
            except Exception: logging.exception("on_close handler crashed")
        logging.warning("[oversee] WS closed (%s) %s", status_code, msg)

    def run(self):
        backoff = 3
        while not self._stop.is_set():
            url = f"{self.url_ws_base}{OVERSEE_PATH.format(robotId=self.robot_id)}"

            # Subprotocol = tokenKey
            try:
                token_key = self.token_key_provider()
            except Exception as e:
                logging.error("[oversee] token_key_provider failed: %s", e)
                self._stop.wait(backoff); backoff = min(30, backoff * 2)
                continue

            logging.info("[oversee] connecting %s", f"{urlparse(url).scheme}://{urlparse(url).netloc}{urlparse(url).path}")
            self._wsapp = websocket.WebSocketApp(
                url,
                subprotocols=[token_key],
                on_message=self._cb_message,
                on_error=self._cb_error,
                on_close=self._cb_close,
                on_open=self._cb_open,
            )

            try:
                if not self.use_env_proxies:
                    saved = {k: os.environ.pop(k) for k in list(os.environ.keys()) if k.lower().endswith("_proxy")}
                    saved_no = {k: os.environ.get(k) for k in ("NO_PROXY", "no_proxy")}
                    os.environ["NO_PROXY"] = "*"; os.environ["no_proxy"] = "*"
                    try:
                        self._wsapp.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
                    finally:
                        os.environ.update(saved)
                        for k, v in saved_no.items():
                            if v is None: os.environ.pop(k, None)
                            else: os.environ[k] = v
                else:
                    self._wsapp.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
            except Exception as e:
                logging.error("[oversee] run_forever exception: %s\n%s", e, traceback.format_exc())

            if self._stop.is_set():
                break
            logging.info("[oversee] reconnecting in %ss", backoff)
            self._stop.wait(backoff)
            backoff = min(30, backoff * 2)

# -------- Public API --------
class AutoXingWSOversee:
    """
    High-level client for: /robot-control/oversee/<robotId>
    - Auth via subprotocol (tokenKey)
    - App heartbeat every 5s
    - subscribe_oversee() returns a worker on which you can call:
        .subscribe(types), .get_robot_state(), .get_task_state(), .get_task_list()
        .control_pause()/resume()/stop()/go_home()/nav_to()
        .task_create()/task_cancel()/task_execute()
        .send({...}) for custom reqTypes
    """
    def __init__(
        self,
        *,
        token_key_provider: Optional[Callable[[], str]] = None,
        url_ws_base: str = DEFAULT_WS_BASE,
        use_env_proxies: bool = False,
    ):
        self.token_key_provider = token_key_provider or _default_token_key_provider
        self.url_ws_base = _normalize_ws_base(url_ws_base)
        self.use_env_proxies = use_env_proxies
        self._workers: list[_WSWorker] = []

    def subscribe_oversee(
        self,
        *,
        robotId: str,
        on_message: Callable[[dict], None],
        subscribe_types: Optional[List[str]] = None,
    ) -> _WSWorker:
        w = _WSWorker(
            robot_id=robotId,
            token_key_provider=self.token_key_provider,
            url_ws_base=self.url_ws_base,
            use_env_proxies=self.use_env_proxies,
            subscribe_types=subscribe_types,
        )
        w.on_message = on_message
        w.start()
        self._workers.append(w)
        return w

    def close_all(self):
        for w in list(self._workers):
            try: w.stop()
            except Exception: pass
        self._workers.clear()

# ---- Back-compat shim (so old code keeps running) ----
class AutoXingWS(AutoXingWSOversee):
    """
    Compatibility wrapper so existing code that imports AutoXingWS and calls
    subscribe_tasks(...) keeps working. Internally uses the oversee socket.
    """
    def __init__(
        self,
        default_business_id: str | None = None,   # ignored; kept for signature compat
        *,
        token_key_provider=None,
        url_ws_base: str = DEFAULT_WS_BASE,
        use_env_proxies: bool = False,
        **_ignored,
    ):
        if token_key_provider is None:
            token_key_provider = _default_token_key_provider
        super().__init__(
            token_key_provider=token_key_provider,
            url_ws_base=url_ws_base,
            use_env_proxies=use_env_proxies,
        )
        self.default_business_id = default_business_id
        logging.warning("AutoXingWS is deprecated. Migrate to AutoXingWSOversee + subscribe_oversee().")

    def subscribe_tasks(self, *, robotId: str, on_message, **kwargs):
        return self.subscribe_oversee(robotId=robotId, on_message=on_message, **kwargs)
