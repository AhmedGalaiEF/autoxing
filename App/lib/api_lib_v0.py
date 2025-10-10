from api_lib import *
from api_lib import _feats_scale_from_pois, _scale_feats_inplace


TYPE_LABELS = {
    "9": "charging_pile",
    "10": "waiting",
    "11": "poi",
    "12": "no_go_or_zone",
    "17": "business_area",
    "34": "shelf",
    "36": "docking_point",
    "38": "stopover",
    # lines: '1' driving, '2' virtual walls/tracks (varies by site)
}

def yaw_deg_norm(y):
    # str → float, wrap to [0, 360)
    y = float(y)
    y = y % 360.0
    return y if y >= 0 else y + 360.0

def yaw_rad(y):
    import math
    return math.radians(yaw_deg_norm(y))

_scale_cache = {}  # {area_id: float}

def compute_and_cache_scale(area_id, feats, pois_df):
    if area_id in _scale_cache:
        return _scale_cache[area_id]
    s = _feats_scale_from_pois(feats, pois_df)
    if not s:
        flat = [abs(v) for f in feats for (x, y) in f["coords"] for v in (x, y)]
        if flat and max(flat) < 0.02:  # km → m heuristic
            s = 1000.0
    _scale_cache[area_id] = s or 1.0
    return _scale_cache[area_id]

def clean_pois_table(pois_df_raw: pd.DataFrame) -> pd.DataFrame:
    if pois_df_raw is None or pois_df_raw.empty:
        return pd.DataFrame(columns=[
            "id","name","type_code","type","x","y","yaw_deg","yaw_rad",
            "relatedShelvesAreaId","shelvePointId","dockingPointId","subtype",
            "shelveDepth","shelveWidth"
        ])

    df = pois_df_raw.copy()
    # flatten coordinate
    def _xy(v):
        try:
            return float(v[0]), float(v[1])
        except Exception:
            return (float("nan"), float("nan"))
    df["x"], df["y"] = zip(*df["coordinate"].map(_xy))

    # normalize yaw
    def _yaw(v):
        if pd.isna(v): return float("nan")
        try: return yaw_deg_norm(v)
        except Exception: return float("nan")
    df["yaw_deg"] = df.get("yaw", float("nan")).map(_yaw)
    df["yaw_rad"] = df["yaw_deg"].map(yaw_rad)

    # explode properties (safely)
    props = df.get("properties")
    if props is not None:
        df["relatedShelvesAreaId"] = props.map(lambda p: (p or {}).get("relatedShelvesAreaId"))
        df["shelvePointId"]      = props.map(lambda p: (p or {}).get("shelvePointId"))
        df["dockingPointId"]     = props.map(lambda p: (p or {}).get("dockingPointId"))
        df["subtype"]            = props.map(lambda p: (p or {}).get("subtype"))
        df["shelveDepth"]        = props.map(lambda p: (p or {}).get("shelveDepth"))
        df["shelveWidth"]        = props.map(lambda p: (p or {}).get("shelveWidth"))

    # type & labels
    df["type_code"] = df["type"].astype(str)
    df["type"] = df["type_code"].map(TYPE_LABELS).fillna(df["type_code"])

    # keep only useful
    keep = ["id","name","type_code","type","x","y","yaw_deg","yaw_rad",
            "relatedShelvesAreaId","shelvePointId","dockingPointId","subtype",
            "shelveDepth","shelveWidth"]
    df = df[keep].drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df


def build_feature_tables(area_id: str, robot_sn: str, pois_df_clean: pd.DataFrame):
    raw = get_map_features(area_id=area_id, robot_sn=robot_sn)
    feats = normalize_features_geojson(raw)  # [{'kind','coords','name','props', 'id'?}]
    scale = compute_and_cache_scale(area_id, feats, pois_df_clean)

    if scale and abs(scale - 1.0) > 1e-9:
        _scale_feats_inplace(feats, scale)

    # Polygons
    areas = []
    for f in feats:
        if f["kind"] != "polygon": continue
        props = f.get("props", {}) or {}
        ring = [(float(x), float(y)) for (x,y) in f["coords"]]
        if len(ring) < 3: continue
        areas.append({
            "id": f.get("id") or props.get("id") or props.get("id"),
            "name": (f.get("name") or props.get("name") or "").strip(),
            "type_code": str(props.get("regionType", props.get("type",""))),
            "regionType": props.get("regionType"),
            "desc": props.get("desc"),
            "blocked": props.get("blocked"),
            "speed": props.get("speed"),
            "polygon": ring,
        })
    areas_df = pd.DataFrame(areas)

    # Lines
    lines = []
    for f in feats:
        if f["kind"] != "polyline": continue
        props = f.get("props", {}) or {}
        seg = [(float(x), float(y)) for (x,y) in f["coords"]]
        if len(seg) < 2: continue
        lines.append({
            "id": f.get("id") or props.get("id"),
            "lineType": str(props.get("lineType", "")),
            "direction": props.get("direction"),
            "driveType": props.get("driveType"),
            "rcsTrack": props.get("rcsTrack"),
            "trackSize": props.get("trackSize"),
            "polyline": seg,
        })
    lines_df = pd.DataFrame(lines)

    # Points (from features, independent from POIs api)
    points = []
    for f in feats:
        if f["kind"] != "point": continue
        props = f.get("props", {}) or {}
        (x,y) = f["coords"][0]
        points.append({
            "id": f.get("id") or props.get("id"),
            "name": (f.get("name") or props.get("name") or "").strip(),
            "type_code": str(props.get("type","")),
            "type": TYPE_LABELS.get(str(props.get("type","")), str(props.get("type",""))),
            "x": float(x),
            "y": float(y),
            "yaw_deg": yaw_deg_norm(props.get("yaw", 0)) if "yaw" in props else float("nan"),
            "yaw_rad": yaw_rad(props.get("yaw", 0)) if "yaw" in props else float("nan"),
            "relatedShelvesAreaId": props.get("relatedShelvesAreaId"),
            "shelvePointId": props.get("shelvePointId"),
            "dockingPointId": props.get("dockingPointId"),
            "subtype": props.get("subtype"),
        })
    points_df = pd.DataFrame(points)
    return areas_df, lines_df, points_df, scale

def link_shelves_docks_areas(pois_df: pd.DataFrame, areas_df: pd.DataFrame):
    shelves = pois_df[pois_df["type"]=="shelf"].copy()
    docks   = pois_df[pois_df["type"]=="docking_point"].copy()

    # shelves → area
    shelves = shelves.merge(
        areas_df[["id","name","regionType","desc"]].rename(columns={"id":"relatedShelvesAreaId","name":"area_name"}),
        on="relatedShelvesAreaId",
        how="left"
    )

    # dock ↔ shelf
    pair1 = docks.merge(
        shelves[["id","name","dockingPointId"]].rename(columns={"id":"shelf_id","name":"shelf_name"}),
        on="dockingPointId",
        how="left"
    )

    # shelf → dock via shelvePointId (reverse link)
    pair2 = shelves.merge(
        docks[["id","name","shelvePointId"]].rename(columns={"id":"dock_id","name":"dock_name"}),
        left_on="id",
        right_on="shelvePointId",
        how="left"
    )

    return shelves, docks, pair1, pair2


def poly_metrics(poly):
    import math
    n = len(poly)
    if n < 3:
        return dict(area_m2=0.0, perimeter_m=0.0, cx=float("nan"), cy=float("nan"),
                    minx=float("inf"), miny=float("inf"), maxx=-float("inf"), maxy=-float("inf"))
    x = [p[0] for p in poly]
    y = [p[1] for p in poly]
    area2 = 0.0; cxn=0.0; cyn=0.0; per=0.0
    minx=min(x); maxx=max(x); miny=min(y); maxy=max(y)
    for i in range(n):
        j=(i+1)%n
        cross = x[i]*y[j]-x[j]*y[i]
        area2 += cross
        cxn += (x[i]+x[j])*cross
        cyn += (y[i]+y[j])*cross
        dx=x[j]-x[i]; dy=y[j]-y[i]
        per += (dx*dx+dy*dy)**0.5
    A = 0.5*area2
    if abs(A) < 1e-12:
        cx=cy=float("nan")
    else:
        cx = cxn/(3.0*area2)
        cy = cyn/(3.0*area2)
    return dict(area_m2=abs(A), perimeter_m=per, cx=cx, cy=cy, minx=minx, miny=miny, maxx=maxx, maxy=maxy)

def enrich_areas(areas_df: pd.DataFrame) -> pd.DataFrame:
    if areas_df is None or areas_df.empty:
        return pd.DataFrame(columns=[
            "id","name","type_code","regionType","desc","blocked","speed",
            "polygon","area_m2","perimeter_m","centroid_x","centroid_y",
            "bbox_minx","bbox_miny","bbox_maxx","bbox_maxy"
        ])
    m = areas_df["polygon"].map(poly_metrics)
    m = pd.json_normalize(m)
    out = areas_df.copy()
    out["area_m2"]      = m["area_m2"]
    out["perimeter_m"]  = m["perimeter_m"]
    out["centroid_x"]   = m["cx"]; out["centroid_y"] = m["cy"]
    out["bbox_minx"]    = m["minx"]; out["bbox_miny"]=m["miny"]
    out["bbox_maxx"]    = m["maxx"]; out["bbox_maxy"]=m["maxy"]
    return out


class Robot_v0():
    def __init__(self, serial_number):
        self.SN = serial_number
        self.refresh()
        self.business = try_get_unique_row(business_df,'id',self.df.businessId).name
        # self.building = try_get_unique_row(get_buildings(),'id',self.df.buildingId).name

    def refresh(self):
        self.df = try_get_unique_row(get_robots(), 'robotId', self.SN)

    def __str__(self):
        # Human-friendly description
        rinspect(self, methods=True)
        return ""
        # return None

    def __repr__(self):
        # Debug/developer-friendly with all key attributes
        return f"{self.__class__.__name__}(id={self.SN!r}, business={self.business}, model={self.df.model}, isOnLine={self.df.isOnLine})"

    def get_map_image(self):
        return get_map_image(self.df.areaId)

    def get_pois(self):
        return get_pois(self.df).drop(columns=["businessId", "buildingId", "areaId"])
    
    def get_poi_details(self, poi_name):
        return get_poi_details(self.df, poi_name)

    def get_poi_pose(self, poi_name):
        return (*get_poi_coordinates(self.df, poi_name), self.get_poi_details(poi_name)['yaw'])

    def get_areas(self):
        return get_areas(self.df)

##### require refresh
    def get_status(self):
        self.refresh()
        if self.df.isOnLine:
            return get_robot_status(self.df)
        log.info("robot is offline")
        return pd.DataFrame(columns=["status","message","data"])

    def get_pose(self):
        self.refresh()
        return (*get_robot_curr_pos(self.df), self.get_status().loc['yaw','data'])

    def get_task(self):
        try:
            return self.get_status().loc['taskObj','data']
        except KeyError:
            return None
        
    def cancel_task(self):
        taskObj = self.get_task()
        if not (taskObj is None):
            task_id = taskObj["taskId"]
            cancel_task(task_id)

    def get_state(self):
        self.refresh()
        status = self.get_status()
        return {
            "isOnLine": self.df.isOnLine,
            "isEmergencyStop": status.loc['isEmergencyStop','data'],
            "version": status.loc['vers','data'],
            "hasObstruction": status.loc['hasObstruction','data'],
            "isCharging": self.df.isCharging,
            "battery": status.loc['battery','data'],
            "task": self.get_task(),
            "pose": self.get_pose(),
            "speed": status.loc['speed','data'],
            "relPos": None,
            "isAt": None,
            "errors": status.loc['errors','data']
        }

    def get_world(self):
        """
        Return a DataFrame of the robot and all POIs in world coordinates.
        Columns: x, y, yaw, poiname
        Index: (x, y, yaw)
        """
        if pd is None:
            raise ImportError("pandas not installed")

        # Robot pose
        try:
            rx, ry, ryaw = self.get_pose()
        except Exception as e:
            log.exception("failed to read robot pose")
            rx = ry = ryaw = float("nan")

        rows = [{
            "x": float(rx),
            "y": float(ry),
            "yaw": float(ryaw),
            "poiname": "__robot__"
        }]

        # POIs
        try:
            pois = self.get_pois()
            if isinstance(pois, pd.DataFrame) and not pois.empty:
                for _, r in pois.iterrows():
                    name = str(r.get("name") or "")
                    coord = r.get("coordinate")
                    yaw = r.get("yaw", None)
                    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                        px, py = float(coord[0]), float(coord[1])
                        pyaw = float(yaw) if yaw is not None else float("nan")
                        rows.append({"x": px, "y": py, "yaw": pyaw, "poiname": name})
        except Exception:
            log.exception("failed to read POIs")

        df = pd.DataFrame(rows, columns=["x", "y", "yaw", "poiname"])
        # MultiIndex as requested
        return df.set_index(["x", "y", "yaw"]).sort_index()


    def get_local_areas(self):
        """
        Polygon features in WORLD meters with extra info.
        Columns:
          - id, name, type, regionType, blocked, speed, mapOverlay, shelvesAreaStopPoints
          - polygon [(x,y)...]  (outer ring)
          - area_m2, perimeter_m, centroid_x, centroid_y, bbox_minx, bbox_miny, bbox_maxx, bbox_maxy
        """
        if pd is None:
            raise ImportError("pandas not installed")

        # --- helpers (no extra deps) ---
        def _poly_metrics(poly):
            # Shoelace area (signed), centroid, perimeter; poly is [(x,y),...], closed not required
            n = len(poly)
            if n < 3:
                return 0.0, 0.0, float("nan"), float("nan"), float("inf"), float("inf"), -float("inf"), -float("inf")
            # ensure open form
            x = [p[0] for p in poly]
            y = [p[1] for p in poly]
            # close ring on the fly
            area2 = 0.0
            cx_num = 0.0
            cy_num = 0.0
            perim = 0.0
            minx = min(x); maxx = max(x); miny = min(y); maxy = max(y)
            for i in range(n):
                j = (i + 1) % n
                cross = x[i]*y[j] - x[j]*y[i]
                area2 += cross
                cx_num += (x[i] + x[j]) * cross
                cy_num += (y[i] + y[j]) * cross
                dx = x[j] - x[i]; dy = y[j] - y[i]
                perim += (dx*dx + dy*dy) ** 0.5
            area = 0.5 * area2
            if abs(area) < 1e-12:
                cx = float("nan"); cy = float("nan")
            else:
                cx = cx_num / (3.0 * area2)
                cy = cy_num / (3.0 * area2)
            return abs(area), perim, cx, cy, minx, miny, maxx, maxy

        try:
            area_id = self.df.areaId
            feat_raw = get_map_features(area_id=area_id, robot_sn=self.SN)
            feats = normalize_features_geojson(feat_raw)  # [{'kind','coords','name','props'}, ...]

            # scale (POI-anchored; fallback ×1000 if tiny)
            try:
                pois_df = self.get_pois()
            except Exception:
                pois_df = None
            scale = _feats_scale_from_pois(feats, pois_df)
            if scale:
                _scale_feats_inplace(feats, scale)
            else:
                flat = [abs(v) for f in feats for (x, y) in f["coords"] for v in (x, y)]
                if flat and max(flat) < 0.02:
                    _scale_feats_inplace(feats, 1000.0)

            rows = []
            for f in feats:
                if f.get("kind") != "polygon":
                    continue
                props = f.get("props", {}) or {}
                name = (f.get("name") or props.get("name") or "").strip()
                ftype = str(props.get("type", "")).strip()
                poly = [(float(x), float(y)) for (x, y) in (f.get("coords") or [])]
                if len(poly) < 3:
                    continue
                area_m2, perim_m, cx, cy, minx, miny, maxx, maxy = _poly_metrics(poly)

                rows.append({
                    "id": f.get("id") or props.get("id"),
                    "name": name,
                    "type": ftype,
                    "regionType": props.get("regionType"),
                    "blocked": props.get("blocked"),
                    "speed": props.get("speed"),
                    "mapOverlay": props.get("mapOverlay"),
                    "shelvesAreaStopPoints": props.get("shelvesAreaStopPoints"),
                    "polygon": poly,
                    "area_m2": area_m2,
                    "perimeter_m": perim_m,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "bbox_minx": minx,
                    "bbox_miny": miny,
                    "bbox_maxx": maxx,
                    "bbox_maxy": maxy,
                })

            cols = ["id","name","type","regionType","blocked","speed","mapOverlay","shelvesAreaStopPoints",
                    "polygon","area_m2","perimeter_m","centroid_x","centroid_y","bbox_minx","bbox_miny","bbox_maxx","bbox_maxy"]
            return pd.DataFrame(rows, columns=cols)
        except Exception:
            log.exception("failed to build local areas (detailed)")
            return pd.DataFrame(columns=["id","name","type","regionType","blocked","speed","mapOverlay",
                                         "shelvesAreaStopPoints","polygon","area_m2","perimeter_m",
                                         "centroid_x","centroid_y","bbox_minx","bbox_miny","bbox_maxx","bbox_maxy"])

# --- add near your other helpers ---
def _point_segment_dist(px, py, x1, y1, x2, y2):
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    vv = vx*vx + vy*vy
    if vv <= 0.0:
        dx, dy = px - x1, py - y1
        return (dx*dx + dy*dy) ** 0.5
    t = (wx*vx + wy*vy) / vv
    if t <= 0.0:
        dx, dy = px - x1, py - y1
        return (dx*dx + dy*dy) ** 0.5
    if t >= 1.0:
        dx, dy = px - x2, py - y2
        return (dx*dx + dy*dy) ** 0.5
    projx, projy = x1 + t*vx, y1 + t*vy
    dx, dy = px - projx, py - projy
    return (dx*dx + dy*dy) ** 0.5

def _point_polygon_min_dist(px, py, ring):
    if not ring or len(ring) < 2:
        return float("inf")
    dmin = float("inf")
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        dmin = min(dmin, _point_segment_dist(px, py, x1, y1, x2, y2))
    return dmin



import math
import pandas as pd
# --- replace your Robot_v1 with this version ---
class Robot_v1(Robot_v0):
    _ctx = None

    def _refresh_context(self, force: bool = False):
        if (self._ctx is not None) and not force:
            return self._ctx
        pois_raw = self.get_pois()
        pois = clean_pois_table(pois_raw)
        areas_df, _lines_df, _points_df, scale = build_feature_tables(self.df.areaId, self.SN, pois)
        areas_rich = enrich_areas(areas_df)
        self._ctx = {"pois": pois, "areas": areas_df, "areas_rich": areas_rich, "scale": scale}
        return self._ctx

    def get_relpos_df(self):
        if pd is None:
            raise ImportError("pandas not installed")

        rx, ry, _ = self.get_pose()
        rx = float(rx); ry = float(ry)
        ctx = self._refresh_context()

        # ---- areas: distances to centroid + min edge distance ----
        areas = ctx["areas_rich"].copy()
        if areas is None or areas.empty:
            area_rows = []
        else:
            disp = areas["desc"].fillna("").astype(str).str.strip()
            disp = disp.where(disp != "", areas["name"].fillna("").astype(str).str.strip())

            # centroid distance
            cdx = areas["centroid_x"] - rx
            cdy = areas["centroid_y"] - ry
            centroid_dist = (cdx.pow(2) + cdy.pow(2)).pow(0.5)

            # min distance to polygon boundary
            edge_dists = []
            for _, r in areas.iterrows():
                edge_dists.append(_point_polygon_min_dist(rx, ry, r["polygon"]))

            area_rows = pd.DataFrame({
                "kind": "area",
                "id": areas["id"],
                "name": disp,
                "type": areas["regionType"].astype(str),
                "desc": areas["desc"],
                "regionType": areas["regionType"],
                "centroid_x": areas["centroid_x"],
                "centroid_y": areas["centroid_y"],
                "x": areas["centroid_x"],
                "y": areas["centroid_y"],
                "centroid_dist_m": centroid_dist,
                "edge_min_dist_m": pd.Series(edge_dists, index=areas.index),
            })

        # ---- pois: point distance ----
        pois = ctx["pois"].copy()
        if pois is None or pois.empty:
            poi_rows = []
        else:
            dx = pois["x"] - rx
            dy = pois["y"] - ry
            dist = (dx.pow(2) + dy.pow(2)).pow(0.5)
            poi_rows = pd.DataFrame({
                "kind": "poi",
                "id": pois["id"],
                "name": pois["name"],
                "type": pois["type"],
                "type_code": pois["type_code"],
                "x": pois["x"],
                "y": pois["y"],
                "yaw_deg": pois["yaw_deg"],
                "yaw_rad": pois["yaw_rad"],
                "distance_m": dist,  # for POIs only
            })

        frames = []
        if isinstance(area_rows, pd.DataFrame) and not area_rows.empty:
            frames.append(area_rows)
        if isinstance(poi_rows, pd.DataFrame) and not poi_rows.empty:
            frames.append(poi_rows)

        if frames:
            relpos = pd.concat(frames, ignore_index=True).sort_values(
                by=[("distance_m" if "distance_m" in frames[-1].columns else "centroid_dist_m"), "kind", "name"],
                na_position="last"
            ).reset_index(drop=True)
        else:
            relpos = pd.DataFrame(columns=[
                "kind","id","name","type","x","y",
                "distance_m","centroid_dist_m","edge_min_dist_m",
                "centroid_x","centroid_y","regionType","desc","yaw_deg","yaw_rad","type_code"
            ])

        return relpos

    def get_state(self, poi_threshold: float = 0.5, min_dist_area: float = 1.0):
        """
        POIs:  isAt if distance_m <= poi_threshold
        Areas: isAt if centroid_dist_m > min_dist_area   (as requested)
        """
        base = super().get_state()
        try:
            rel = self.get_relpos_df()

            # build masks
            poi_mask = (rel["kind"] == "poi")
            area_mask = (rel["kind"] == "area")

            is_at_poi = pd.Series(False, index=rel.index)
            is_at_area = pd.Series(False, index=rel.index)

            if poi_mask.any():
                is_at_poi[poi_mask] = rel.loc[poi_mask, "distance_m"] <= float(poi_threshold)

            if area_mask.any():
                # your rule
                is_at_area[area_mask] = rel.loc[area_mask, "centroid_dist_m"] < float(min_dist_area)

            is_at = rel[is_at_poi | is_at_area].reset_index(drop=True)

            base["relPos"] = rel
            base["isAt"] = is_at
            base["params"] = {"poi_threshold": float(poi_threshold), "min_dist_area": float(min_dist_area)}
        except Exception:
            log.exception("failed to compute relPos/isAt")
            base["relPos"] = pd.DataFrame()
            base["isAt"] = pd.DataFrame()
        return base


# ---- human-readable → int maps ----
TASK_TYPE = {
    "disinfection": 0, "return_to_dock": 1, "restaurant": 2, "hotel": 3,
    "delivery": 4, "factory": 5, "chassis_miniapp": 6, "charge_sched": 7,
    "lift": 29, "default": 5  # pick your backend default
}

RUN_TYPE = {
    "sched_disinfect": 0, "temp_disinfect": 1,
    "quick_meal": 20, "multi_meal": 21, "direct": 22, "roam": 23,
    "return": 24, "charging_station": 25, "summon": 26, "birthday": 27,
    "guide": 28, "lift": 29, "lift_cruise": 30, "flex_carry": 31,
    "roll": 32, "full_unplug": 33, "sequential": 21
}

ACTION = {
    "open_door": 6, "close_door": 28, "pause": 18, "play_audio": 5,
    "spray": 32, "set_speed": 41, "light_on": 37, "light_off": 38,
    "wait_interaction": 40, "lift_up": 47, "lift_down": 48
}

RUN_MODE = {"flex_avoid": 1, "traj_limited": 2, "traj_no_avoid": 3, "traj_no_dock_repl": 4}
ROUTE_MODE = {"sequential": 1, "shortest": 2}
SOURCE_TYPE = {
    "unknown": 0, "head_app": 1, "miniapp": 2, "pager": 3, "chassis": 4,
    "dispatch": 5, "sdk": 6, "pad": 7
}

import math

class Task:
    """
    Fluent builder for create_task payloads that conforms to:
      - Area delivery (lift up/down using areaId in stepActs.data.useAreaId)
      - Shelf lift at shelf points (type 34)
      - Dock drops (type 36)
      - Pause actions (type 18, data.pauseTime)
    """

    TASK_TYPE   = TASK_TYPE
    RUN_TYPE    = RUN_TYPE
    RUN_MODE    = RUN_MODE
    ROUTE_MODE  = ROUTE_MODE
    SOURCE_TYPE = SOURCE_TYPE

    # action codes
    ACT_WAIT     = ACTION.get("pause", 18)
    ACT_AUDIO     = ACTION.get("play_audio", 5)
    ACT_LIFT_UP  = ACTION.get("lift_up", 47)
    ACT_LIFT_DOWN= ACTION.get("lift_down", 48)

    # POI type gates (string codes from backend)
    TYPE_SHELF = "34"
    TYPE_DOCK  = "36"
    TYPE_TABLE  = "11"
    TYPE_CHARGER  = "9"
    TYPE_STANDBY  = "10"

    def __init__(self,
                 robot: Robot_v1,
                 name: str = "SDK Task",
                 runType="sequential",
                 taskType="default",
                 routeMode="sequential",
                 runMode="flex_avoid",
                 sourceType="sdk",
                 runNum: int = 1,
                 speed: float = -1,
                 detourRadius: float = 1.0,
                 ignorePublicSite: bool = False):
        self.robot = robot
        self._name = name
        self._runType    = self._norm(Task.RUN_TYPE, runType)
        self._taskType   = self._norm(Task.TASK_TYPE, taskType)
        self._routeMode  = self._norm(Task.ROUTE_MODE, routeMode)
        self._runMode    = self._norm(Task.RUN_MODE, runMode)
        self._sourceType = self._norm(Task.SOURCE_TYPE, sourceType)

        self._runNum        = int(runNum)
        self._speed         = float(speed)
        self._detourRadius  = float(detourRadius)
        self._ignorePublic  = bool(ignorePublicSite)

        self._taskPts: list[dict] = []
        self._curPt: dict | None  = None
        self._backPt: dict | None = None

    # ---------- utils ----------
    @staticmethod
    def _norm(table, val):
        return table[val] if isinstance(val, str) else int(val)

    def _poi_details(self, name: str) -> dict:
        det = self.robot.get_poi_details(name)
        if not isinstance(det, dict):
            raise ValueError(f"POI '{name}' not found")
        return det

    def _poi_pose(self, name: str):
        x, y, yaw = self.robot.get_poi_pose(name)
        return float(x), float(y), (None if yaw is None else float(yaw))

    def _require_poi_type(self, name: str, allowed: set[str], label: str) -> dict:
        det = self._poi_details(name)
        t = str(det.get("type", "")).strip()
        if t not in allowed:
            raise ValueError(f"{label}: '{name}' has type {t}, expected one of {sorted(allowed)}")
        return det

    def _find_area(self, area_name: str):
        ctx = self.robot._refresh_context()
        areas = ctx["areas_rich"]
        if areas is None or areas.empty:
            raise ValueError("No areas available")

        # prefer desc, fallback to name
        m = areas[areas["desc"].astype(str).str.strip() == area_name]
        if m.empty:
            m = areas[areas["name"].astype(str).str.strip() == area_name]
        if m.empty:
            raise ValueError(f"Area '{area_name}' not found (match by desc or name)")

        row = m.iloc[0]
        return {
            "id": row["id"],
            "name": (row["desc"] if isinstance(row["desc"], str) and row["desc"].strip() else row["name"]),
            "x": float(row["centroid_x"]),
            "y": float(row["centroid_y"]),
        }

    @staticmethod
    def _maybe_yaw(yaw):
        if yaw is None:
            return None
        if isinstance(yaw, float) and math.isnan(yaw):
            return None
        return float(yaw)

    @staticmethod
    def _mk_point(*, x, y, yaw=None, areaId=None, ext: dict | None = None, stepActs: list | None = None):
        pt = {
            "x": float(x),
            "y": float(y),
            # "ext": ext or {},          # always include ext
        }
        if not (ext is None):
            pt["ext"] = ext
        myaw = Task._maybe_yaw(yaw)
        if myaw is not None:
            pt["yaw"] = myaw
        if areaId:
            pt["areaId"] = areaId
        if stepActs:
            pt["stepActs"] = stepActs
        return pt

    # ---------- chainable API ----------

    def cur_at_robot(self):
        """Set curPt to the robot's current pose (optional)."""
        x, y, yaw = self.robot.get_pose()
        self._curPt = Task._mk_point(x=x, y=y, yaw=y, ext={"name": "__cur__", "id": "__robot__"})
        return self

    # --- shelf pickup / lift at shelf point (type 34 required) ---

    def pickup(self, shelf_name: str, *, lift_up: bool | None = None, lift_down: bool | None = None,
               stopRadius: float | None = None, extra_ext: dict | None = None):
        det = self._require_poi_type(shelf_name, {Task.TYPE_SHELF}, "pickup")
        x, y, yaw = self._poi_pose(shelf_name)
        ext = {"name": shelf_name, "id": det.get("id")}
        if extra_ext:
            ext.update(extra_ext)

        acts = None
        if lift_up is True:
            acts = [{"type": Task.ACT_LIFT_UP, "data": {}}]
        elif lift_down is True:
            acts = [{"type": Task.ACT_LIFT_DOWN, "data": {}}]

        pt = Task._mk_point(x=x, y=y, yaw=yaw, areaId=det.get("areaId") or det.get("relatedShelvesAreaId"),
                            ext=ext, stepActs=acts)
        if stopRadius is not None:
            pt["stopRadius"] = float(stopRadius)
        self._taskPts.append(pt)
        return self

    def lift_up_here(self):
        if not self._taskPts:
            raise ValueError("No point to attach lift_up to")
        self._taskPts[-1].setdefault("stepActs", []).append({"type": Task.ACT_LIFT_UP, "data": {}})
        return self

    def lift_down_here(self):
        if not self._taskPts:
            raise ValueError("No point to attach lift_down to")
        self._taskPts[-1].setdefault("stepActs", []).append({"type": Task.ACT_LIFT_DOWN, "data": {}})
        return self

    # --- dock drop (type 36 required) ---

    def drop_at_dock(self, dock_name: str, *, extra_ext: dict | None = None):
        det = self._require_poi_type(dock_name, {Task.TYPE_DOCK}, "drop_at_dock")
        x, y, yaw = self._poi_pose(dock_name)
        ext = {"name": dock_name, "id": det.get("id")}
        if extra_ext:
            ext.update(extra_ext)
        pt = Task._mk_point(x=x, y=y, yaw=yaw, areaId=det.get("areaId"), ext=ext)
        self._taskPts.append(pt)
        return self
    
    def go_charge(self, charger_name: str, *, extra_ext: dict | None = None):
        det = self._require_poi_type(charger_name, {Task.TYPE_CHARGER}, "charge")
        x, y, yaw = self._poi_pose(charger_name)
        ext = {"name": charger_name, "id": det.get("id")}
        if extra_ext:
            ext.update(extra_ext)
        pt = Task._mk_point(x=x, y=y, yaw=yaw, areaId=det.get("areaId"), ext=ext)
        self._taskPts.append(pt)
        return self
    
    def go_back(self, back_name: str, *, extra_ext: dict | None = None):
        det = self._require_poi_type(back_name, {Task.TYPE_STANDBY, Task.TYPE_CHARGER}, "charge")
        x, y, yaw = self._poi_pose(back_name)
        ext = {"name": back_name, "id": det.get("id")}
        if extra_ext:
            ext.update(extra_ext)
        pt = Task._mk_point(x=x, y=y, yaw=yaw, areaId=det.get("areaId"), ext=ext)
        self._taskPts.append(pt)
        return self
    
    def goto(self, pose, *, extra_ext: dict | None = None):
        # det = self._require_poi_type(back_name, {Task.TYPE_STANDBY, Task.TYPE_CHARGER}, "charge")
        x, y, yaw = pose
        # ext = {"name": back_name, "id": det.get("id")}
        # if extra_ext:
        #     ext.update(extra_ext)
        pt = Task._mk_point(x=x, y=y, yaw=yaw, areaId=self.robot.df.areaId)
        self._taskPts.append(pt)
        return self


    # --- AREA delivery (exactly like your example) ---

    def to_area(self, area_name: str, *, lift: str | None = None, extra_ext: dict | None = None):
        """
        Add a point at the area's centroid. If lift in {'up','down'}, add stepAct
        with data.useAreaId = areaId.
        """
        area = self._find_area(area_name)
        ext = {"name": area["name"], "id": area["id"]}
        if extra_ext:
            ext.update(extra_ext)

        acts = None
        if lift:
            if lift not in ("up", "down"):
                raise ValueError("lift must be 'up', 'down', or None")
            act_type = Task.ACT_LIFT_UP if lift == "up" else Task.ACT_LIFT_DOWN
            acts = [{"type": act_type, "data": {"useAreaId": area["id"]}}]

        pt = Task._mk_point(x=area["x"], y=area["y"], ext=ext, areaId=area["id"], stepActs=acts)
        self._taskPts.append(pt)
        return self

    # alias for readability
    def dropdown(self, target_name: str, *, useArea: bool = False, lift: str | None = None):
        if useArea:
            return self.to_area(target_name, lift=lift)
        return self.drop_at_dock(target_name)

    def wait(self, target_name: str, seconds: float):
        """Attach a pause stepAct to the last point, or create a point at robot pose if empty."""
        det = self._require_poi_type(target_name, {Task.TYPE_DOCK, Task.TYPE_SHELF, Task.TYPE_TABLE}, "wait")
        x, y, yaw = self._poi_pose(target_name)
        ext = {"name": target_name, "id": det.get("id")}
        sec = int(max(0, round(float(seconds))))
        if sec <= 0:
            return self
        
        act = {"type": Task.ACT_WAIT, "data": {"pauseTime": sec}}
        pt = Task._mk_point(x=x, y=y, yaw=yaw, ext=ext, areaId=det.get("areaId"))
        self._taskPts.append(pt)
        if not self._taskPts:
            x, y, yaw = self.robot.get_pose()
            pt = Task._mk_point(x=x, y=y, yaw=yaw, ext=ext, areaId=det.get("areaId"), stepActs=[act])
            self._taskPts.append(pt)
        else:
            self._taskPts[-1].setdefault("stepActs", []).append(act)
        return self

    def back(self, name: str | None = None, *, is_area: bool = False):
        """
        Set optional backPt. If not called (or name=None), backPt is omitted.
        """
        if name is None:
            self._backPt = None
            return self

        if is_area:
            area = self._find_area(name)
            self._backPt = Task._mk_point(x=area["x"], y=area["y"],
                                          ext={"name": area["name"], "id": area["id"]},
                                          areaId=area["id"])
            return self

        # POI back point (any type)
        det = self._poi_details(name)
        x, y, yaw = self._poi_pose(name)
        self._backPt = Task._mk_point(x=x, y=y, yaw=yaw,
                                      ext={"name": name, "id": det.get("id")},
                                      areaId=det.get("areaId"))
        return self

    # ---------- payload ----------
    @property
    def task_dict(self):
        """
        Build payload for create_task(**payload). Omits backPt if not set.
        """
        base = {
            "task_name": self._name,
            "robot": self.robot.df,
            # "businessId": self.robot.df.businessId,
            "runNum": self._runNum,
            "taskType": self._taskType,
            "runType": self._runType,
            "routeMode": self._routeMode,
            "runMode": self._runMode,
            "ignorePublicSite": self._ignorePublic,
            "sourceType": self._sourceType,
            "speed": self._speed,
            "detourRadius": self._detourRadius,
            "taskPts": self._taskPts,
        }
        if self._curPt:
            base["curPt"] = self._curPt
        if self._backPt:
            base["backPt"] = self._backPt
        return base
    

#### Music playing
# In [94]: task_dict = {
#     ...:     'name': 'test',
#     ...:     'robotId': '8982412804553br',
#     ...:     'businessId': '668fbe1568f09f1ae8873ca3',
#     ...:     'runNum': 1,
#     ...:     'taskType': 2,
#     ...:     'runType': 22,
#     ...:     'routeMode': 1,
#     ...:     'runMode': 1,
#     ...:     'taskPts': [
#     ...:         {
#     ...:             'x': 0.10250000000000001,
#     ...:             'y': 1.8025000000000002,
#     ...:             'ext': {'name': 'P1', 'id': '68e399a23c27696b19a962cf'},
#     ...:             'yaw': 119.75,
#     ...:             'areaId': '68e39ade1782dd9aff690228',
#     ...:             'stepActs': [{"type":5,"data":{"mode":1}}]
#     ...:         }
#     ...:     ],
#     ...:     'sourceType': 6,
#     ...:     'ignorePublicSite': False,
#     ...:     'speed': -1.0,
#     ...:     'detourRadius': 1.0
#     ...: }

# In [95]: r = requests.post(url, headers={"X-Token": X_TOKEN}, json=task_dict)

# In [96]:




class Robot_v2(Robot_v1):
    def go_to_pose(self, pose):
        task = Task(self, "goto", taskType="delivery",runType="roam").goto(pose)
        resp = create_task(**task.task_dict)
        return self
    def go_charge(self):
        pois = self.get_pois()
        charging_station = pois[pois.type == 9].iloc[0]['name']
        # print(charging_station)
        task = Task(self, "charge", taskType="return_to_dock",runType="charging_station")
        # pose = task._poi_pose(charging_station)
        task = task.go_charge(charging_station)
        resp = create_task(**task.task_dict)
        return self
    def go_to_poi(self, poi_name):
        task = Task(self, "goto", taskType="delivery",runType="roam")
        pose = task._poi_pose(poi_name)
        task = task.goto(pose)
        resp = create_task(**task.task_dict)
        return self

    def wait_at(self, poi_name, wait_seconds=10):
        task = Task(self, "wait", taskType="delivery",runType="roam").wait(poi_name, wait_seconds)
        resp = create_task(**task.task_dict)
        return self

    def pickup_at(self, poi_name, area_delivery=False):
        task = Task(self, "pickup", taskType="factory",runType="lift").pickup(poi_name, lift_up=True)
        resp = create_task(**task.task_dict)
        return self

    def dropdown_at(self, poi_name, area_delivery=False):
        task = Task(self, "dropdown", taskType="factory",runType="lift").dropdown(poi_name, useArea=area_delivery)
        resp = create_task(**task.task_dict)
        return self

    def go_back(self):
        pois = self.get_pois()
        standby = pois[pois.type == 10].iloc[0]['name']
        # print(charging_station)
        task = Task(self, "back", taskType="return_to_dock",runType="return")
        pose = task._poi_pose(standby)
        task = task.goto(pose)
        resp = create_task(**task.task_dict)
        return self

    def shelf_to_shelf(self, pickup_shelf, dropdown_shelf, area_delivery=False):
        task = (
            Task(self, "shelf_delivery", taskType="factory",runType="lift")
            .pickup(pickup_shelf, lift_up=True)
            .dropdown(dropdown_shelf, useArea=area_delivery)
        )
        resp = create_task(**task.task_dict)
        return self
    
    def evacuate(self, area_name=None, evac_pts=[]):
        pass

    def wrap(self):
        pass

    # def say(self, audio_dict):
    #     if audio_dict["mp3_id"] is None:
    #         ...
    #     else :
    #         ...