# webapp.py
import os, json, time
from datetime import datetime

import pandas as pd
from dash import Dash, html, dcc, Output, Input, State, no_update
import dash_bootstrap_components as dbc

from api_lib import Robot, create_task, cancel_task, render_full_map, get_online_robots

ROBOT_ID = get_online_robots().iloc[0].robotId
robot = Robot(ROBOT_ID)

ROOT = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
LIVE_PNG = os.path.join(ASSETS_DIR, "map_live.png")

# ---------- helpers ----------
def load_pois_df() -> pd.DataFrame:
    try:
        return robot.get_pois()
    except Exception as e:
        print(f"[ERR] load_pois: {e}")
        return pd.DataFrame(columns=["id","name","coordinate","type"])

def poi_options(df: pd.DataFrame):
    if df is None or df.empty:
        return []
    return [{"label": str(r["name"]), "value": str(r["name"])} for _, r in df.iterrows()]

def status_df() -> pd.DataFrame:
    try:
        return robot.get_status()
    except Exception as e:
        return pd.DataFrame({"error":[str(e)]})

def df_table(df: pd.DataFrame):
    if df is None or df.empty:
        return dbc.Alert("No data.", color="warning", className="mb-0")
    head = [html.Th(c) for c in df.columns]
    rows = []
    for _, row in df.head(60).iterrows():
        cells = []
        for v in row:
            if isinstance(v, (dict, list)):
                s = json.dumps(v, ensure_ascii=False)
                cells.append(html.Td(s if len(s) <= 200 else s[:197]+"..."))
            else:
                cells.append(html.Td(v))
        rows.append(html.Tr(cells))
    return dbc.Table([html.Thead(html.Tr(head)), html.Tbody(rows)], bordered=True, hover=True, size="sm", className="mb-0")

def figure_for_assets_png(path: str) -> dict:
    # zoomable PNG as a background in dcc.Graph
    import PIL.Image as P
    try:
        with P.open(path) as im:
            w, h = im.width, im.height
    except Exception:
        w, h = 1280, 720
    ts = int(time.time())
    return {
        "data": [],
        "layout": {
            "images": [{
                "source": f"/assets/{os.path.basename(path)}?ts={ts}",
                "xref": "x", "yref": "y",
                "x": 0, "y": h, "sizex": w, "sizey": h,
                "sizing": "stretch", "layer": "below",
            }],
            "xaxis": {"range": [0, w], "scaleanchor": "y", "constrain": "domain", "visible": False},
            "yaxis": {"range": [0, h], "autorange": "reversed", "visible": False},
            "margin": {"l":0,"r":0,"t":0,"b":0},
            "dragmode": "pan",
            "uirevision": "keep",
        }
    }

# ---------- preload POIs once; no auto refresh ----------
POIS = load_pois_df()

# ---------- UI ----------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title="Robot POI + Map")

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Robot POI / Map / Tasks", className="ms-2"),
        dbc.Badge(ROBOT_ID, color="secondary", className="ms-2"),
    ]), color="dark", dark=True, className="mb-3"
)

left = dbc.Card(dbc.CardBody([
    html.H6("POI"),
    dcc.Dropdown(id="poi-dd", options=poi_options(POIS), placeholder="Select a POI…", clearable=True),
    html.Div(id="poi-fields", className="mt-3", children=[
        dbc.Row([dbc.Col("x:", width=2), dbc.Col(html.Code(id="f-x"))]),
        dbc.Row([dbc.Col("y:", width=2), dbc.Col(html.Code(id="f-y"))]),
        dbc.Row([dbc.Col("yaw:", width=2), dbc.Col(html.Code(id="f-yaw"))]),
        dbc.Row([dbc.Col("type:", width=2), dbc.Col(html.Code(id="f-type"))]),
        dbc.Row([dbc.Col("poi_id:", width=2), dbc.Col(html.Code(id="f-id"))]),
        dbc.Row([dbc.Col("areaId:", width=2), dbc.Col(html.Code(id="f-area"))]),
    ]),
    dcc.Store(id="poi-details"),
    html.Hr(),
    html.H6("Task config"),
    dbc.Row([
        dbc.Col(dbc.Input(id="task-name", placeholder="Task name (auto)"), width=6),
        dbc.Col(dbc.Input(id="run-type", type="number", value=22, step=1), width=3),
        dbc.Col(dbc.Input(id="speed", type="number", value=0.4, min=0.1, max=1.2, step=0.1), width=3),
    ], className="mb-2"),
    dbc.Row([
        dbc.Col(dbc.Input(id="source-type", type="number", value=6, step=1), width=3),
        dbc.Col(dbc.Input(id="detour-radius", type="number", value=1.0, step=0.5), width=3),
        dbc.Col(dbc.Input(id="stop-radius", type="number", value=1.0, step=0.5), width=3),
        dbc.Col(dbc.Button("Create Task", id="btn-create", color="primary"), width=3),
    ]),
    html.Hr(),
    dbc.ButtonGroup([
        dbc.Button("Update Map", id="btn-map", color="secondary"),
        dbc.Button("Refresh Status", id="btn-status", color="info"),
        dbc.Button("Cancel Current Task", id="btn-cancel", color="danger"),
    ]),
]), className="mb-3")

right = dbc.Card(dbc.CardBody([
    html.H6("Map (click Update Map to regenerate)"),
    dcc.Graph(id="map", figure=figure_for_assets_png(LIVE_PNG),
              config={"displayModeBar": True, "scrollZoom": True, "doubleClick": "reset"},
              style={"height":"70vh","border":"1px solid #ddd","borderRadius":"6px"}),
    html.Div(id="last-op", className="text-muted mt-2", style={"fontSize":"0.9rem"}),
    html.Hr(),
    html.H6("Status / Response"),
    dcc.Loading(html.Pre(id="out", style={"whiteSpace":"pre-wrap"})),
]), className="mb-3")

toast = dbc.Toast(id="toast", header="", is_open=False, duration=4000, dismissable=True,
                  icon="info", children="", style={"position":"fixed","top":15,"right":15,"zIndex":2000})

app.layout = dbc.Container([navbar, dbc.Row([dbc.Col(left, width=4), dbc.Col(right, width=8)]), toast], fluid=True)

# ---------- callbacks ----------

# 1) POI -> details
@app.callback(
    Output("f-x","children"), Output("f-y","children"), Output("f-yaw","children"),
    Output("f-type","children"), Output("f-id","children"), Output("f-area","children"),
    Output("poi-details","data"),
    Input("poi-dd","value"),
    prevent_initial_call=True
)
def on_poi(poi_name):
    if not poi_name:
        return "", "", "", "", "", "", None
    print(f"[UI] POI selected: {poi_name}")
    details = robot.get_poi_details(poi_name)
    coord = details.get("coordinate") or [None, None]
    x, y = coord[0], coord[1]
    yaw = details.get("yaw")
    t   = details.get("type")
    pid = details.get("id")
    area= details.get("areaId")
    return (f"{x:.4f}" if isinstance(x,(int,float)) else "",
            f"{y:.4f}" if isinstance(y,(int,float)) else "",
            f"{yaw:.4f}" if isinstance(yaw,(int,float)) else "",
            str(t), str(pid), str(area), details)

# 2) Update Map (no auto refresh; this actually calls render_full_map and reloads the figure)
@app.callback(
    Output("map","figure"),
    Output("last-op","children"),
    Output("toast","is_open", allow_duplicate=True), Output("toast","header", allow_duplicate=True), Output("toast","children", allow_duplicate=True), Output("toast","icon", allow_duplicate=True),
    Input("btn-map","n_clicks"),
    prevent_initial_call=True
)
def on_update_map(_n):
    print(f"[UI] Update Map clicked at {datetime.now().isoformat(timespec='seconds')}")
    try:
        render_full_map(ROBOT_ID, out_png=LIVE_PNG)
        fig = figure_for_assets_png(LIVE_PNG)
        ts  = time.strftime("%Y-%m-%d %H:%M:%S")
        return fig, f"Map updated at {ts}", True, "Map", "Updated successfully.", "success"
    except Exception as e:
        fig = figure_for_assets_png(LIVE_PNG)
        return fig, "Map update failed.", True, "Map", f"Error: {e}", "danger"

# 3) Refresh Status (shows Robot.get_status())
@app.callback(
    Output("out","children", allow_duplicate=True),
    Output("toast","is_open", allow_duplicate=True), Output("toast","header", allow_duplicate=True), Output("toast","children", allow_duplicate=True), Output("toast","icon", allow_duplicate=True),
    Input("btn-status","n_clicks"),
    prevent_initial_call=True
)
def on_refresh_status(_n):
    print(f"[UI] Refresh Status clicked at {datetime.now().isoformat(timespec='seconds')}")
    df = status_df()
    try:
        table_like = df.to_string()
    except Exception:
        table_like = str(df)
    return table_like, True, "Status", "Status fetched.", "info"

# 4) Create Task (uses selected POI details to build a 1-point task)
@app.callback(
    Output("out","children", allow_duplicate=True),
    Output("toast","is_open", allow_duplicate=True), Output("toast","header", allow_duplicate=True), Output("toast","children", allow_duplicate=True), Output("toast","icon", allow_duplicate=True),
    Input("btn-create","n_clicks"),
    State("poi-details","data"),
    State("task-name","value"),
    State("run-type","value"),
    State("speed","value"),
    State("source-type","value"),
    State("detour-radius","value"),
    State("stop-radius","value"),
    prevent_initial_call=True
)
def on_create_task(_n, poi_details, task_name, run_type, speed, source_type, detour_radius, stop_radius):
    print(f"[UI] Create Task clicked at {datetime.now().isoformat(timespec='seconds')}")
    if not poi_details or not isinstance(poi_details, dict):
        msg = "Pick a POI first."
        return msg, True, "Create Task", msg, "warning"

    # Build a taskPts entry like the one you showed
    x, y = poi_details["coordinate"][0], poi_details["coordinate"][1]
    yaw  = poi_details.get("yaw", 0.0)
    pt_type = poi_details.get("type")
    poi_id  = poi_details.get("id")
    area_id = poi_details.get("areaId")

    taskPts = [{
        "x": float(x), "y": float(y), "yaw": float(yaw),
        "stopRadius": float(stop_radius or 1.0),
        "areaId": area_id,
        "ext": {"id": poi_id, "name": poi_details.get("name")},
        # "stepActs": [],                 # add acts if you need them
        "type": pt_type
    }]

    # Default name if empty
    if not (task_name and str(task_name).strip()):
        task_name = f"GoTo-{poi_details.get('name','POI')}-{datetime.now().strftime('%H%M%S')}"

    try:
        data = create_task(
            task_name=task_name,
            robot=robot.df,                # create_task expects robot row
            runType=int(run_type or 22),
            sourceType=int(source_type or 6),
            taskPts=taskPts,
            speed=float(speed or 0.4),
            detourRadius=float(detour_radius or 1.0),
        )
        msg = f"Task created: {data.get('taskId','?')}"
        pretty = json.dumps(data, ensure_ascii=False, indent=2)
        return pretty, True, "Create Task", msg, "success"
    except Exception as e:
        msg = f"Create failed: {e}"
        return msg, True, "Create Task", msg, "danger"

# 5) Cancel current task (reads taskObj.taskId)
@app.callback(
    Output("out","children", allow_duplicate=True),
    Output("toast","is_open", allow_duplicate=True), Output("toast","header", allow_duplicate=True), Output("toast","children", allow_duplicate=True), Output("toast","icon", allow_duplicate=True),
    Input("btn-cancel","n_clicks"),
    prevent_initial_call=True
)
def on_cancel(_n):
    print(f"[UI] Cancel Task clicked at {datetime.now().isoformat(timespec='seconds')}")
    try:
        df = status_df()
        if not isinstance(df, pd.DataFrame) or "taskObj" not in df.index:
            msg = "No active task (taskObj not found)."
            return msg, True, "Cancel Task", msg, "warning"
        row = df.loc["taskObj"]
        payload = row.get("data") if isinstance(row, pd.Series) else None
        if not isinstance(payload, dict):
            msg = "taskObj present but malformed."
            return msg, True, "Cancel Task", msg, "warning"
        task_id = payload.get("taskId")
        if not task_id:
            msg = "taskId missing in taskObj."
            return msg, True, "Cancel Task", msg, "warning"
        if payload.get("isFinish"):
            msg = f"Task {task_id} already finished."
            return msg, True, "Cancel Task", msg, "info"
        if payload.get("isCancel"):
            msg = f"Task {task_id} already cancelled."
            return msg, True, "Cancel Task", msg, "info"

        resp = cancel_task(task_id)
        out = f"cancel_task({task_id}) -> {str(resp)[:300]}"
        return out, True, "Cancel Task", f"Cancel sent for {task_id}.", "success"
    except Exception as e:
        msg = f"Cancel error: {e}"
        return msg, True, "Cancel Task", msg, "danger"

# ---------- main ----------
if __name__ == "__main__":
    # IMPORTANT: you’ll see console logs for each button click
    app.run(host="0.0.0.0", port=8050, debug=True)
