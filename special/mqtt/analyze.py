import time
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable, Tuple

import pandas as pd


# -----------------------------
# 1) Normalize get_status output
# -----------------------------

def status_to_payload(status_obj: Any) -> Dict[str, Any]:
    """
    Convert Robot(...).get_status() return into a flat dict of field->value.

    Supports:
      - pandas DataFrame shaped like: index=fields, columns include 'data'
      - dict already
    """
    if isinstance(status_obj, dict):
        return status_obj

    # Your printout looks like a DataFrame with rows areaId/battery/... and a 'data' column.
    if isinstance(status_obj, pd.DataFrame):
        if "data" not in status_obj.columns:
            raise ValueError(f"Unexpected status DataFrame columns: {status_obj.columns.tolist()}")
        # index are the keys, 'data' are the values
        payload = status_obj["data"].to_dict()
        return payload

    # If it's a Series or something else, try a best-effort conversion
    if hasattr(status_obj, "to_dict"):
        return status_obj.to_dict()

    raise TypeError(f"Unsupported status type: {type(status_obj)}")


def safe_json(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        # last resort: string
        return str(v)


# -----------------------------
# 2) Poller -> snapshot table
# -----------------------------

@dataclass
class PollConfig:
    interval_s: float = 5.0
    # stop after N seconds (None = run forever)
    run_seconds: Optional[float] = None


def poll_robots(
    robots_df: pd.DataFrame,
    robot_id_col: str = "robotId",
    cfg: PollConfig = PollConfig(),
) -> pd.DataFrame:
    """
    Poll all robots in robots_df repeatedly. Returns a DataFrame of snapshots.

    Snapshot schema:
      - poll_ts (UTC-ish local machine time), poll_ts_ms
      - robotId
      - core scalar fields (battery, isOnline, moveState, speed, etc if present)
      - errors_json, taskObj_json, path_json (raw blobs)
    """
    snapshots: List[Dict[str, Any]] = []
    t0 = time.time()

    try:
        while True:
            now = time.time()
            if cfg.run_seconds is not None and (now - t0) >= cfg.run_seconds:
                break

            poll_ts_ms = int(now * 1000)
            poll_ts = pd.to_datetime(poll_ts_ms, unit="ms", utc=True)

            for _, row in robots_df.iterrows():
                rid = row[robot_id_col]

                try:
                    status_obj = Robot(rid).get_status()
                    payload = status_to_payload(status_obj)
                except Exception as e:
                    # capture polling failure as a snapshot too
                    snapshots.append({
                        "poll_ts": poll_ts,
                        "poll_ts_ms": poll_ts_ms,
                        "robotId": rid,
                        "poll_error": repr(e),
                    })
                    continue

                # Pull out common scalar fields (add more as you need)
                rec: Dict[str, Any] = {
                    "poll_ts": poll_ts,
                    "poll_ts_ms": poll_ts_ms,
                    "robotId": payload.get("robotId", rid),

                    "timestamp_ms": payload.get("timestamp"),  # robot's timestamp if present
                    "battery": payload.get("battery"),
                    "hasNet": payload.get("hasNet"),
                    "isOnline": payload.get("isOnline"),
                    "isCharging": payload.get("isCharging"),
                    "isEmergencyStop": payload.get("isEmergencyStop"),
                    "moveState": payload.get("moveState"),
                    "speed": payload.get("speed"),
                    "locQuality": payload.get("locQuality"),

                    "x": payload.get("x"),
                    "y": payload.get("y"),
                    "yaw": payload.get("yaw"),
                    "ori": payload.get("ori"),

                    "businessId": payload.get("businessId"),
                    "buildingId": payload.get("buildingId"),
                    "areaId": payload.get("areaId"),
                }

                # Raw blobs (present only sometimes)
                rec["errors_json"] = safe_json(payload.get("errors"))
                rec["taskObj_json"] = safe_json(payload.get("taskObj"))
                rec["path_json"] = safe_json(payload.get("path"))

                # Helpful derived “presence flags”
                rec["has_errors"] = bool(payload.get("errors")) if payload.get("errors") is not None else False
                rec["is_tasking"] = payload.get("taskObj") is not None
                rec["has_path"] = payload.get("path") is not None

                snapshots.append(rec)

            time.sleep(cfg.interval_s)

    except KeyboardInterrupt:
        pass

    df = pd.DataFrame(snapshots)

    # Normalize types a bit
    if "timestamp_ms" in df.columns:
        df["robot_ts"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True, errors="coerce")
    return df


# -----------------------------
# 3) Uptime / downtime analytics
# -----------------------------

def add_uptime_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Define 'up' in one place so everything uses the same rule.
    Tweak as needed.
    """
    out = df.copy()
    # Common “robot is reachable + healthy enough” definition
    out["up"] = (
        out.get("isOnline").fillna(False).astype(bool)
        & out.get("hasNet").fillna(False).astype(bool)
        & ~out.get("isEmergencyStop").fillna(False).astype(bool)
    )
    return out


def compute_uptime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute uptime seconds and uptime percent per robot over the captured window.
    Uses the poll interval deltas (so make sure your polling interval is stable-ish).
    """
    d = add_uptime_flags(df).sort_values(["robotId", "poll_ts"]).copy()
    # delta to next sample per robot
    d["dt_s"] = (
        d.groupby("robotId")["poll_ts"]
        .diff()
        .dt.total_seconds()
        .fillna(0)
    )
    # Assign dt to the *current* row (time since previous poll); good enough for polling.
    d["up_s"] = d["dt_s"] * d["up"].astype(float)

    summary = (
        d.groupby("robotId", as_index=False)
        .agg(
            window_s=("dt_s", "sum"),
            uptime_s=("up_s", "sum"),
            samples=("up", "size"),
            last_seen=("poll_ts", "max"),
        )
    )
    summary["uptime_pct"] = (summary["uptime_s"] / summary["window_s"]).where(summary["window_s"] > 0) * 100.0
    return summary


def extract_downtime_episodes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return downtime episodes with start/end timestamps per robot.
    """
    d = add_uptime_flags(df).sort_values(["robotId", "poll_ts"]).copy()
    d["up_prev"] = d.groupby("robotId")["up"].shift(1)

    # episode starts when up goes True->False
    d["down_start"] = (d["up_prev"] == True) & (d["up"] == False)
    # episode ends when up goes False->True
    d["down_end"] = (d["up_prev"] == False) & (d["up"] == True)

    episodes: List[Dict[str, Any]] = []
    for rid, g in d.groupby("robotId"):
        start_ts = None
        for _, r in g.iterrows():
            if r["down_start"]:
                start_ts = r["poll_ts"]
            if start_ts is not None and r["down_end"]:
                end_ts = r["poll_ts"]
                episodes.append({
                    "robotId": rid,
                    "down_start": start_ts,
                    "down_end": end_ts,
                    "down_s": (end_ts - start_ts).total_seconds(),
                })
                start_ts = None

        # open episode (still down at end)
        if start_ts is not None:
            end_ts = g["poll_ts"].iloc[-1]
            episodes.append({
                "robotId": rid,
                "down_start": start_ts,
                "down_end": pd.NaT,
                "down_s": (end_ts - start_ts).total_seconds(),
            })

    return pd.DataFrame(episodes)


# -----------------------------
# 4) Error events (new / cleared)
# -----------------------------

def _parse_errors(errors_json: Any) -> List[Dict[str, Any]]:
    if errors_json is None or (isinstance(errors_json, float) and pd.isna(errors_json)):
        return []
    if isinstance(errors_json, str):
        try:
            v = json.loads(errors_json)
            return v if isinstance(v, list) else []
        except Exception:
            return []
    if isinstance(errors_json, list):
        return errors_json
    return []


def extract_error_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect *changes* in errors list between polls:
      - error_raised: appears now but wasn't present before
      - error_cleared: was present before but not now
    """
    d = df.sort_values(["robotId", "poll_ts"]).copy()

    events: List[Dict[str, Any]] = []
    for rid, g in d.groupby("robotId"):
        prev_codes: set = set()
        for _, r in g.iterrows():
            errs = _parse_errors(r.get("errors_json"))
            # represent an error by its code (add message/level in payload for events)
            now_codes = {e.get("code") for e in errs if "code" in e}

            raised = now_codes - prev_codes
            cleared = prev_codes - now_codes

            if raised:
                for code in raised:
                    # find matching error dict for details
                    detail = next((e for e in errs if e.get("code") == code), {})
                    events.append({
                        "robotId": rid,
                        "ts": r["poll_ts"],
                        "event": "error_raised",
                        "code": code,
                        "level": detail.get("level"),
                        "message": detail.get("message"),
                        "raw_json": safe_json(detail),
                    })

            if cleared:
                for code in cleared:
                    events.append({
                        "robotId": rid,
                        "ts": r["poll_ts"],
                        "event": "error_cleared",
                        "code": code,
                        "level": None,
                        "message": None,
                        "raw_json": None,
                    })

            prev_codes = now_codes

    return pd.DataFrame(events).sort_values(["robotId", "ts", "event", "code"])


# -----------------------------
# 5) Task sessions + path info
# -----------------------------

def _parse_json_blob(s: Any) -> Optional[Dict[str, Any]]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    if isinstance(s, dict):
        return s
    if isinstance(s, str):
        try:
            v = json.loads(s)
            return v if isinstance(v, dict) else None
        except Exception:
            return None
    return None


def extract_task_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task session = contiguous period where taskObj exists.
    Also captures first/last path snapshot for that session (if available).
    """
    d = df.sort_values(["robotId", "poll_ts"]).copy()
    d["is_tasking"] = d["taskObj_json"].notna()

    sessions: List[Dict[str, Any]] = []

    for rid, g in d.groupby("robotId"):
        in_task = False
        start_row = None
        last_row = None

        for _, r in g.iterrows():
            is_task = bool(r["is_tasking"])
            if is_task and not in_task:
                in_task = True
                start_row = r
            if in_task and is_task:
                last_row = r
            if in_task and not is_task:
                # close session
                sessions.append(_session_from_rows(rid, start_row, last_row))
                in_task = False
                start_row = None
                last_row = None

        if in_task and start_row is not None and last_row is not None:
            sessions.append(_session_from_rows(rid, start_row, last_row))

    return pd.DataFrame(sessions).sort_values(["robotId", "task_start"])


def _session_from_rows(robot_id: str, start_row: pd.Series, last_row: pd.Series) -> Dict[str, Any]:
    start_task = _parse_json_blob(start_row.get("taskObj_json"))
    end_task = _parse_json_blob(last_row.get("taskObj_json"))

    start_path = _parse_json_blob(start_row.get("path_json"))
    end_path = _parse_json_blob(last_row.get("path_json"))

    def path_positions_count(p: Optional[Dict[str, Any]]) -> Optional[int]:
        if not p:
            return None
        pos = p.get("positions")
        return len(pos) if isinstance(pos, list) else None

    return {
        "robotId": robot_id,
        "task_start": start_row["poll_ts"],
        "task_end": last_row["poll_ts"],
        "task_s": (last_row["poll_ts"] - start_row["poll_ts"]).total_seconds(),

        # Whatever is inside taskObj depends on your system; keep raw + a few common fields
        "taskObj_start_json": start_row.get("taskObj_json"),
        "taskObj_end_json": last_row.get("taskObj_json"),

        "actIndex_start": (start_task or {}).get("actIndex"),
        "actIndex_end": (end_task or {}).get("actIndex"),
        "duration_field_start": (start_task or {}).get("duration"),
        "duration_field_end": (end_task or {}).get("duration"),

        "path_start_positions_n": path_positions_count(start_path),
        "path_end_positions_n": path_positions_count(end_path),
        "path_start_json": start_row.get("path_json"),
        "path_end_json": last_row.get("path_json"),
    }


# -----------------------------
# 6) Example usage
# -----------------------------

# 1) Collect snapshots (e.g., poll for 10 minutes)
# snapshots = poll_robots(online_robots, cfg=PollConfig(interval_s=5, run_seconds=600))

# 2) Persist raw snapshots (pick one)
# snapshots.to_parquet("robot_snapshots.parquet", index=False)
# snapshots.to_csv("robot_snapshots.csv", index=False)

# 3) Uptime
# uptime = compute_uptime_summary(snapshots)
# downtime_eps = extract_downtime_episodes(snapshots)

# 4) Error events
# error_events = extract_error_events(snapshots)

# 5) Task sessions
# task_sessions = extract_task_sessions(snapshots)
