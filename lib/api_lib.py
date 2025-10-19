
"""
##########################################################################################
  Imports
##########################################################################################
"""

#### api abstraction
# from rich import pretty
from rich import print as rprint
from rich import inspect as rinspect

# from __future__ import annotations
import time
import hashlib

import logging
import requests

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None  # Only used if caller asks for DataFrame
    np = None
# ---- Rich logging setup (drop-in, safe to call multiple times) ----
try:
    from rich.logging import RichHandler
    _root = logging.getLogger()
    if not any(isinstance(h, RichHandler) for h in _root.handlers):
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True, show_time=True, show_level=True, show_path=False)],
        )
except Exception:
    # Fall back silently if rich isn't installed; user can pip install rich
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

log = logging.getLogger(__name__)
# pretty.install()
"""
##########################################################################################
  Inits
##########################################################################################
"""

##### api reöated

## Ruwen's tokens:
from dotenv import load_dotenv
import io, os

# Clean the file in place
with open('.env', 'rb') as f:
    data = f.read().replace(b'\x00', b'')

with open('.env', 'wb') as f:
    f.write(data)

load_dotenv()

app_id = os.getenv("APPID")
app_secret = os.getenv("APPSECRET")
app_code = os.getenv("APPCODE")

####### Urls:
URL_BASE = r"https://apiglobal.autoxing.com"
URL_ROUTING_DICT = {
    "x-token" : r"/auth/v1.1/token",
    "business_list" : r"/business/v1.1/list",
    "robot_list" : r"/robot/v1.1/list",
    "building_list" : r"/building/v1.1/list",
    "create_task" : r"/task/v3/create",
    "task_list" : r"/task/v1.1/list",
    "task_details": r"/task/v3/taskId",
    "cancel_task" : r"/task/v3/taskId/cancel",
    "area_list" : r"/map/v1.1/area/list",
    "poi_list" : r"/map/v1.1/poi/list",
    "map_image" : r"/map/v1.1/area/areaId/base-map",
    "execute_task" : r"/task/v1.1/taskId/execute",
    "task_status" : r"/task/v2.0/taskId/state",
    "robot_status" : r"/robot/v2.0/robotId/state",
    "poi_details": r"/map/v1.1/poi/poiId",
}

#### sdk related


### raspberrypi related

"""
##########################################################################################
  util
##########################################################################################
"""

def try_get_unique_row(df, col, value):
    matches = df[df[col] == value]
    if len(matches) > 1:
        log.warning("Expected Unique Id, Received duplicate values:")
        print(matches)
    if len(matches) == 1:
        return matches.iloc[0]
    return None

def _build_url(base_url, path):
    return base_url.rstrip("/") + "/" + str(path).lstrip("/")

# from googletrans import Translator

# translator = Translator()


"""
##########################################################################################
  API Functions
##########################################################################################
"""

#### auth

def get_token_and_key():
    import time, hashlib, requests, os
    app_id = os.getenv("APPID")
    app_secret = os.getenv("APPSECRET")
    app_code = os.getenv("APPCODE")

    ts = int(time.time())
    sign = hashlib.md5(f"{app_id}{ts}{app_secret}".encode("utf-8")).hexdigest()
    url = _build_url(URL_BASE, URL_ROUTING_DICT['x-token'])  # same endpoint
    r = requests.post(
        url,
        headers={"Authorization": app_code},
        json={"appId": app_id, "timestamp": ts, "sign": sign},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json().get("data", {})
    return {
        "key": data.get("key"),           # <-- tokenKey for WS subprotocol
        "token": data.get("token"),       # <-- JWT for REST (X-Token)
        "expireTime": data.get("expireTime"),
    }

def get_token_key():
    return get_token_and_key()["key"]

# If you still need the REST token provider:
def get_token():
    return get_token_and_key()["token"]


X_TOKEN = get_token() #### remember to reset !

#### buildings and business
def get_buildings():
    url = _build_url(URL_BASE, URL_ROUTING_DICT['building_list'])
    r = requests.post(
        url,
        headers = {
            "X-Token" : X_TOKEN
        },
        json = {
            
    })
    try:
        return pd.DataFrame(r.json()["data"]["lists"])
    except Exception as exp:
        log.exception("api abstraction error")
        # rinspect(r)
        return r.text
    
buildings_df = get_buildings()


def get_business(name=None):
    url = _build_url(URL_BASE, URL_ROUTING_DICT['business_list'])
    r = requests.post(
        url,
        headers = {
            "X-Token" : X_TOKEN
        },
        json = {
            
    })
    try:
        df = pd.DataFrame(r.json()["data"]["lists"])

        if name:
            df = df[df.name.str.startswith(name)]
        return df
    except Exception as exp:
        log.exception("api abstraction error")
        # rinspect(r)
        return r.text
    
business_df = get_business()

#### robots

def get_robots(robot_id = None):
    url = _build_url(URL_BASE, URL_ROUTING_DICT['robot_list'])
    r = requests.post(
        url,
        headers = {
            "X-Token" : X_TOKEN
        },
        json = {
            
    })
    try:
        df = pd.DataFrame(r.json()["data"]["list"])
        df = pd.merge(df, business_df[["id","name"]].rename(columns={"id":"businessId", "name":"business_name"}), on="businessId", how="left")
        # df['model_en'] = df['model'].apply(
        #     lambda x: translator.translate(x, src='zh-cn', dest='en').text
        # )
        if robot_id:
            df = df[df.robotId.str.endswith(robot_id)]
        return df
    except Exception as exp:
        log.exception("api abstraction error")
        rinspect(r)
        return r.text
    
robots_df = get_robots()

def get_online_robots():
    return robots_df[robots_df.isOnLine]
    
def get_ef_robots():
    return get_business_robots('EF') # use regex

def get_business_robots(business_name_start):
    ef_business_ids = get_business(business_name_start).id.values
    robots = robots_df[robots_df['businessId'].isin(ef_business_ids)]
    return robots

def get_robot_curr_pos(robot):
    x, y = robot.x, robot.y
    return (x, y)

def get_robot_status(robot):
    url = _build_url(URL_BASE, URL_ROUTING_DICT["robot_status"].replace("robotId", robot.robotId))
    r = requests.get(
        url,
        headers = {
            "X-Token" : X_TOKEN
        })
    try:
        return pd.DataFrame(r.json())
    except Exception as exp:
        # rinspect(r)
        return r.text
    # return r

def connect_to_robot(robot):
    return Robot(robot.robotId)

################# map
from PIL import Image
# import matplotlib.pyplot as plt

from PIL import Image, UnidentifiedImageError  # <-- add this import

def get_map_image(area_id):
    url = _build_url(URL_BASE, URL_ROUTING_DICT["map_image"].replace("areaId", area_id))
    r = requests.get(url, headers={"X-Token": X_TOKEN})
    try:
        img_data = r.content
        image = Image.open(io.BytesIO(img_data))
        return image
    except UnidentifiedImageError:
        log.info("UnidentifiedImageError")
        return r.text
    except Exception:
        log.exception("api abstraction error")
        return r.text

# def plt_map_img(plt, img):
#     plt.imshow(img)
#     plt.axis('off')  # Hide axes
#     plt.show()

def get_pois(robot):
    url = _build_url(URL_BASE, URL_ROUTING_DICT['poi_list'])
    r = requests.post(
        url,
        headers = {
            "X-Token" : X_TOKEN
        },
        json = {
            "businessId" : robot.businessId,
            "robotId" : robot.robotId,
            "areaId" : robot.areaId
    })
    try:
        return pd.DataFrame(r.json()['data']["list"])
    except Exception as exp:
        log.exception("api abstraction error")
        # rinspect(r)
        return r.text

def get_poi_coordinates(robot, poi_name):
    poi_df = get_pois(robot)
    poi = try_get_unique_row(poi_df, "name", poi_name)
    x, y = poi.coordinate[0], poi.coordinate[1]
    return (x,y)

def get_poi_details(robot, poi_name):
    poi_df = get_pois(robot)
    poi_id = try_get_unique_row(poi_df, "name", poi_name).id
    url = _build_url(URL_BASE, URL_ROUTING_DICT["poi_details"].replace("poiId", poi_id))
    r = requests.get(
        url,
        headers = {
            "X-Token" : X_TOKEN
        })
    try:
        return r.json()['data']
    except Exception as exp:
        log.exception("api abstraction error")
        # rinspect(r)
        return r.text

def get_areas(robot):
    url = _build_url(URL_BASE, URL_ROUTING_DICT['area_list'])
    r = requests.post(
        url,
        headers = {
            "X-Token" : X_TOKEN
        },
        json = {
            "businessId" : robot.businessId,
            "robotId" : robot.robotId
    })
    try:
        return pd.DataFrame(r.json()['data']["list"])
    except Exception as exp:
        log.exception("api abstraction error")
        # rinspect(r)
        return r.text

########### tasks

def get_tasks():
    url = _build_url(URL_BASE, URL_ROUTING_DICT['task_list'])
    r = requests.post(
        url,
        headers = {
            "X-Token" : X_TOKEN
        },
        json = {
            "pageSize": 100,
            "pageNum" : 1,
            # "startTime": sdk.time.time(),
            # "endTime" : sdk.time.time() + 100,

    })
    try:
        return pd.DataFrame(r.json()["data"]["list"])
    except Exception as exp:
        log.exception("api abstraction error")
        # rinspect(r)
        return r.text

def execute_task(task_id):
    url = _build_url(URL_BASE, URL_ROUTING_DICT["execute_task"].replace("taskId", task_id))
    r = requests.post(
        url,
        headers = {
            "X-Token" : X_TOKEN
        },
        json = {
    })
    try:
      # return pd.DataFrame(r.json()['data']["list"])
      return r.text
    except Exception as exp:
        log.exception("api abstraction error")
    #   rinspect(r)
    # raise NotImplementedError
    # rinspect(r)
        return r.text
    # return r

def cancel_task(task_id):
    url = _build_url(URL_BASE, URL_ROUTING_DICT["cancel_task"].replace("taskId", task_id))
    r = requests.post(
        url,
        headers = {
            "X-Token" : X_TOKEN
        },
        json = {
    })
    try:
      # return pd.DataFrame(r.json()['data']["list"])
      return r.text
    except Exception as exp:
        log.exception("api abstraction error")
      # rinspect(r)
    # raise NotImplementedError
    # rinspect(r)
        return r.text
    # return r

def get_task_details(task_id):
    url = _build_url(URL_BASE, URL_ROUTING_DICT["task_details"].replace("taskId", task_id))
    r = requests.get(
        url,
        headers = {
            "X-Token" : X_TOKEN
        })
    try:
        return pd.DataFrame(r.json()['data'])
    except Exception as exp:
        log.exception("api abstraction error")
        # rinspect(r)
        return r.text

def get_task_status(task_id):
    url = _build_url(URL_BASE, URL_ROUTING_DICT["task_status"].replace("taskId", task_id))
    r = requests.get(
        url,
        headers = {
            "X-Token" : X_TOKEN
        })
    try:
        return pd.DataFrame(r.json()['data'])
    except Exception as exp:
        log.exception("api abstraction error")
        # rinspect(r)
        return r.text


def create_task(
    task_name,
    robot,
    runType,
    sourceType,
    taskPts=None,
    runNum=1,
    taskType=5,
    routeMode=1,
    runMode=1,
    ignorePublicSite=False,
    speed=0.4,
    detourRadius=1,
    backPt=None
):
    if taskPts is None:
        taskPts = []
    if backPt is None:
        backPt = {}

    task_dict = {
        "name": task_name,
        "robotId": robot.robotId,
        "businessId": robot.businessId,
        "runNum": runNum,
        "taskType": taskType,
        "runType": runType,
        "routeMode": routeMode,
        "runMode": runMode,
        "taskPts": taskPts,
        "sourceType": sourceType,
        "ignorePublicSite": ignorePublicSite,
        "speed": speed,
        "detourRadius": detourRadius,
        # "backPt": backPt,
    }
    if not (backPt == {}):
        task_dict["backPt"] = backPt

    url = _build_url(URL_BASE, URL_ROUTING_DICT['create_task'])
    rprint(task_dict)
    r = requests.post(url, headers={"X-Token": X_TOKEN}, json=task_dict)

    if r.status_code != 200:
        raise RuntimeError(f"Task creation failed {r.status_code}: {r.text}")

    try:
        data = r.json()
    except Exception:
        raise RuntimeError(f"Invalid JSON response: {r.text}")

    if data.get("status") != 200:
        raise RuntimeError(f"API error: {data}")

    # Return the useful stuff, not a DataFrame
    return data["data"]   # e.g. contains taskId, etc.



"""
##########################################################################################
  Classes
##########################################################################################
"""

class Robot():
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
        return get_pois(self.df)

    def get_poi_coordinates(self, poi_name):
        return get_poi_coordinates(self.df, poi_name)
    
    def get_poi_details(self, poi_name):
        return get_poi_details(self.df, poi_name)

    def get_areas(self):
        return get_areas(self.df)

##### require refresh

    def get_curr_pos(self):
        self.refresh()
        return get_robot_curr_pos(self.df)
    def get_status(self):
        self.refresh()
        if self.df.isOnLine:
            return get_robot_status(self.df)
        log.info("robot is offline")
        return pd.DataFrame(columns=["status","message","data"])


    # def execute_task(self, task_id):
    #     raise NotImplementedError

    # def get_task_details(self, task_id):
    #     raise NotImplementedError

    # def get_task_status(self, task_id):
    #     raise NotImplementedError

    # def create_task(self, task_dict):
    #     raise NotImplementedError

    # def get_tasks(self, task_dict):
    #     raise NotImplementedError

    # def get_business(self, task_dict):
    #     raise NotImplementedError

    # def get_buildings(self, task_dict):
    #     raise NotImplementedError

    # def get_robots(self, task_dict):
    #     raise NotImplementedError

    # def cancel_task(self):
    #     raise NotImplementedError

"""
##########################################################################################
  Raspberrypi abstractions
##########################################################################################
"""

##### web requests

# import requests

# # Replace with the machine's API endpoint
# url = "http://192.168.1.100/start"  
# payload = {"command": "start"}  # Depends on machine API
# headers = {"Content-Type": "application/json"}

# response = requests.post(url, json=payload, headers=headers)

# if response.status_code == 200:
#     print("Wrapping machine started successfully.")
# else:
#     print(f"Failed to start machine: {response.status_code} {response.text}")

######### GPIO output

# import RPi.GPIO as GPIO
# import time

# GPIO.setmode(GPIO.BCM)
# relay_pin = 18  # GPIO pin connected to relay
# GPIO.setup(relay_pin, GPIO.OUT)

# print("Turning on wrapping machine...")
# GPIO.output(relay_pin, GPIO.HIGH)  # Close relay (simulate button press)
# time.sleep(1)                      # Hold for 1 second
# GPIO.output(relay_pin, GPIO.LOW)   # Release relay

# GPIO.cleanup()
# print("Done.")


##### serial client

# from pymodbus.client import ModbusSerialClient

# # For RS-485 / Modbus RTU
# client = ModbusSerialClient(port='/dev/ttyUSB0', baudrate=9600, parity='N', stopbits=1, bytesize=8, timeout=2)

# if client.connect():
#     # Example: Write to coil 1 (turn machine ON)
#     result = client.write_coil(1, True, unit=1)
#     if result.isError():
#         print("Failed to send command.")
#     else:
#         print("Machine start command sent.")
#     client.close()
# else:
#     print("Failed to connect to machine.")


##### SPI Input

# import spidev

# # SPI setup for MCP3008 ADC
# spi = spidev.SpiDev()
# spi.open(0, 0)  # Bus 0, Device (CS) 0
# spi.max_speed_hz = 1350000

# # Read MCP3008 channel function
# def read_adc(channel):
#     if channel < 0 or channel > 7:
#         raise ValueError("ADC channel must be 0-7")
#     adc = spi.xfer2([1, (8 + channel) << 4, 0])
#     data = ((adc[1] & 3) << 8) + adc[2]
#     return data  # 0-1023 range (10-bit ADC)


##### GPIO Input

# import RPi.GPIO as GPIO
# import time

# # Use BCM pin numbering
# GPIO.setmode(GPIO.BCM)

# # List of input pins you want to check
# input_pins = [17, 27, 22]  # Change these to your desired GPIO pins

# # Set up pins as inputs with pull-down resistors
# for pin in input_pins:
#     GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# print("Reading GPIO inputs (press Ctrl+C to exit)...")

# try:
#     while True:
#         for pin in input_pins:
#             state = GPIO.input(pin)
#             print(f"GPIO {pin}: {'HIGH' if state else 'LOW'}")
#         time.sleep(0.5)  # Delay between readings
# except KeyboardInterrupt:
#     print("\nExiting...")
# finally:
#     GPIO.cleanup()

def get_ws_client(default_business_id=None):
    from lib.ws_ext import AutoXingWSOversee
    return AutoXingWSOversee(token_key_provider=get_token_key, use_env_proxies=False)

# ---- MAP helpers (replace your current block from URL_ROUTING_DICT.update(...) down) ----
URL_ROUTING_DICT.update({
    "map_meta_by_area":      r"/map/v1.1/area/areaId/map-meta",     # GET
    "map_features_by_area":  r"/map/v1.1/area/areaId/features",     # GET  (GeoJSON FeatureCollection)
    "base_map_by_area":      r"/map/v1.1/area/areaId/base-map",     # GET  (binary)
})

import math, io, time, json
from PIL import Image, ImageDraw, ImageFont

# ---------- transforms ----------
def world_to_pixel(x_m, y_m, *, origin_x_m, origin_y_m, res_m_per_px, img_h_px, rotation_deg=0.0):
    dx, dy = x_m - origin_x_m, y_m - origin_y_m
    if rotation_deg:
        th = math.radians(rotation_deg)
        dx, dy = dx*math.cos(th) - dy*math.sin(th), dx*math.sin(th) + dy*math.cos(th)
    px, py = dx / res_m_per_px, dy / res_m_per_px
    return (px, img_h_px - py)  # flip Y

def norm_to_pixel(nx, ny, *, img_w_px, img_h_px):
    # normalized overlay coords (0..~0.01 in your sample): multiply by raster size and flip Y
    return (nx * img_w_px, img_h_px - (ny * img_h_px))

# ---------- lookups ----------
def _get_area_id_from_robot_sn(robot_sn: str) -> str | None:
    try:
        df = get_robots()
        if "serialNum" in df.columns:
            m = df[df["serialNum"] == robot_sn]
            if not m.empty: return m.iloc[0]["areaId"]
        m = df[df["robotId"] == robot_sn]
        if not m.empty: return m.iloc[0]["areaId"]
    except Exception:
        pass
    return None

# ---------- API calls ----------
def get_map_meta(area_id: str, robot_sn: str) -> dict:
    url = _build_url(URL_BASE, URL_ROUTING_DICT["map_meta_by_area"].replace("areaId", area_id))
    r = requests.get(url, params={"robotSn": robot_sn}, headers={"X-Token": X_TOKEN}, timeout=10)
    r.raise_for_status()
    return r.json()

def get_map_features(area_id: str, robot_sn: str) -> dict:
    url = _build_url(URL_BASE, URL_ROUTING_DICT["map_features_by_area"].replace("areaId", area_id))
    r = requests.get(url, params={"robotSn": robot_sn}, headers={"X-Token": X_TOKEN}, timeout=12)
    r.raise_for_status()
    return r.json()

def get_base_map_image_by_area(area_id: str) -> Image.Image:
    url = _build_url(URL_BASE, URL_ROUTING_DICT["base_map_by_area"].replace("areaId", area_id))
    r = requests.get(url, headers={"X-Token": X_TOKEN}, timeout=15)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content))

# ---------- normalization ----------
def normalize_map_meta(meta_json: dict) -> tuple[dict, str]:
    d = meta_json.get("data", meta_json) or {}
    area_id = d.get("areaId") or d.get("area_id") or d.get("mapAreaId")
    origin = d.get("origin") or {}
    ox = origin.get("x", d.get("originX"))
    oy = origin.get("y", d.get("originY"))
    res = d.get("resolution") or d.get("res")
    ppm = d.get("pixelPerMeter") or d.get("pixelsPerMeter")
    if res is None and ppm: res = 1.0/float(ppm)
    rot = d.get("rotationDeg") or d.get("rotation") or d.get("bearing") or 0.0
    meta = {}
    if ox is not None and oy is not None: meta["origin_x_m"], meta["origin_y_m"] = float(ox), float(oy)
    if res is not None: meta["res_m_per_px"] = float(res)
    meta["rotation_deg"] = float(rot) if rot else 0.0
    # also pass through raster size (needed for normalized overlays)
    if "pixelWidth" in d and "pixelHeight" in d:
        meta["pixel_w"] = int(d["pixelWidth"]); meta["pixel_h"] = int(d["pixelHeight"])
    return meta, area_id

def normalize_features_geojson(feat_json: dict) -> list[dict]:
    """
    Returns a list of simplified features:
      { "kind": "polygon"|"polyline"|"point",
        "coords": [...],       # list of (nx, ny) normalized coords
        "name": str|None,
        "props": dict          # original properties
      }
    Geo is under data.features[*] with geometry.type Point|LineString|Polygon
    """
    root = feat_json.get("data", feat_json) or {}
    feats = root.get("features") if root.get("type") == "FeatureCollection" else root.get("features")
    if not isinstance(feats, list): return []

    out = []
    for f in feats:
        geom = f.get("geometry") or {}
        gtype = (geom.get("type") or "").lower()
        props = f.get("properties", {}) or {}
        name = props.get("name")
        if gtype == "point":
            x, y = geom.get("coordinates", [None, None])[:2]
            if x is None or y is None: continue
            out.append({"kind":"point", "coords":[(float(x), float(y))], "name":name, "props":props})
        elif gtype == "linestring":
            line = geom.get("coordinates", []) or []
            coords = [(float(x), float(y)) for x,y in line if isinstance(x,(int,float)) and isinstance(y,(int,float))]
            if len(coords) >= 2:
                out.append({"kind":"polyline", "coords":coords, "name":name, "props":props})
        elif gtype == "polygon":
            rings = geom.get("coordinates", [])
            if rings and isinstance(rings[0], list):
                ring = rings[0]  # outer
                coords = [(float(x), float(y)) for x,y in ring if isinstance(x,(int,float)) and isinstance(y,(int,float))]
                if len(coords) >= 3:
                    out.append({"kind":"polygon", "coords":coords, "name":name, "props":props})
    return out


def _scale_to_meters(coords):
    """
    Features arrive as meters (normal) or kilometers (values ~0.00x).
    If the largest magnitude < 0.02, treat as km and multiply by 1000.
    """
    flat = [abs(v) for xy in coords for v in xy]
    maxv = max(flat) if flat else 0.0
    scale = 1000.0 if maxv < 0.02 else 1.0  # heuristic that matches your dump
    return [(x * scale, y * scale) for (x, y) in coords]

def normalize_features(feat_json: dict):
    """
    Accepts the FeatureCollection you printed and returns lists in METERS:
      - areas:  [{"name":..., "polygon":[(x,y),...]}, ...]
      - vwalls: [{"polyline":[(x,y),...]}, ...]
      - pois:   [{"name":..., "coordinate":(x,y), "type": str, "yaw": float}, ...]
    """
    d = feat_json.get("data", feat_json) or {}
    if d.get("type") == "FeatureCollection":
        features = d.get("features", [])
    else:
        # older schema fallback (won't hit on your dump)
        features = (d.get("features") or
                    d.get("pois") or d.get("points") or d.get("waypoints") or [])

    areas_out, vwalls_out, pois_out = [], [], []

    for f in features:
        g = f.get("geometry", {})
        t = (g.get("type") or "").lower()
        props = f.get("properties", {})
        name = f.get("name") or props.get("name") or ""
        ftype = str(props.get("type", "")).strip()
        yaw = props.get("yaw")

        if t == "point":
            x, y = g.get("coordinates", [None, None])[:2]
            if x is None or y is None:
                continue
            x, y = _scale_to_meters([(float(x), float(y))])[0]
            pois_out.append({"name": name, "coordinate": (x, y), "type": ftype, "yaw": float(yaw) if yaw is not None else None})

        elif t == "linestring":
            coords = g.get("coordinates", [])
            if not coords:
                continue
            pts = _scale_to_meters([(float(x), float(y)) for x, y in coords])
            vwalls_out.append({"polyline": pts, "type": ftype})

        elif t == "polygon":
            rings = g.get("coordinates", [])
            if not rings:
                continue
            # use the outer ring
            outer = rings[0] if rings else []
            pts = _scale_to_meters([(float(x), float(y)) for x, y in outer])
            areas_out.append({"name": name, "polygon": pts, "type": ftype})

    return areas_out, vwalls_out, pois_out


# ---------- robot pose ----------
def get_robot_pose(robot_sn_or_id: str) -> dict:
    df = get_robots()
    row = None
    if "serialNum" in df.columns:
        m = df[df["serialNum"] == robot_sn_or_id]
        if not m.empty: row = m.iloc[0]
    if row is None:
        m = df[df["robotId"] == robot_sn_or_id]
        if not m.empty: row = m.iloc[0]
    if row is None:
        raise RuntimeError("Robot not found in get_robots()")
    url = _build_url(URL_BASE, URL_ROUTING_DICT["robot_status"].replace("robotId", row["robotId"]))
    r = requests.get(url, headers={"X-Token": X_TOKEN}, timeout=8)
    r.raise_for_status()
    obj = r.json()
    data = obj.get("data", obj)
    state = data.get("state", data)
    x, y = float(state.get("x")), float(state.get("y"))
    yaw = state.get("yaw")
    if yaw is None:
        ori = state.get("ori")
        yaw = math.radians(float(ori)) if ori is not None else 0.0
    return {"x": x, "y": y, "yaw": float(yaw)}

# ---------- drawing ----------
def _draw_arrow(draw, p0, p1, color=(255,0,0,220), width=3, head_len=8, head_w=6):
    draw.line([p0, p1], fill=color, width=width)
    vx, vy = p1[0]-p0[0], p1[1]-p0[1]
    L = max((vx*vx+vy*vy)**0.5, 1e-6)
    ux, uy = vx/L, vy/L
    left  = (p1[0]-ux*head_len - uy*head_w/2, p1[1]-uy*head_len + ux*head_w/2)
    right = (p1[0]-ux*head_len + uy*head_w/2, p1[1]-uy*head_len - ux*head_w/2)
    draw.polygon([p1, left, right], fill=color)

def draw_robot_arrow(img: Image.Image, meta: dict, pose: dict, length_m=0.8, color=(255,0,0,220)):
    ox, oy = meta["origin_x_m"], meta["origin_y_m"]
    res, rot = meta["res_m_per_px"], float(meta.get("rotation_deg", 0.0))
    H = img.height
    yaw = pose["yaw"]
    if abs(yaw) > 2*math.pi: yaw = math.radians(yaw)
    x0, y0 = pose["x"], pose["y"]
    x1, y1 = x0 + length_m*math.cos(yaw), y0 + length_m*math.sin(yaw)
    p0 = world_to_pixel(x0, y0, origin_x_m=ox, origin_y_m=oy, res_m_per_px=res, img_h_px=H, rotation_deg=rot)
    p1 = world_to_pixel(x1, y1, origin_x_m=ox, origin_y_m=oy, res_m_per_px=res, img_h_px=H, rotation_deg=rot)
    drw = ImageDraw.Draw(img, "RGBA")
    _draw_arrow(drw, p0, p1, color=color, width=4, head_len=10, head_w=8)


COLORS = {
    "polygon.default": (0, 150, 255, 40),
    "polygon.outline": (0, 150, 255, 180),
    "polyline.default": (255, 80, 0, 200),
    "poi.default": (0, 200, 0, 220),
    "poi.9": (0, 160, 255, 220),      # charging pile
    "poi.10": (255, 0, 255, 220),     # waiting
    "poi.34": (0, 120, 0, 220),       # shelf
    "poi.36": (200, 120, 0, 220),     # docking point
    "poi.38": (120, 0, 120, 220),     # stopover
}

def draw_overlays_on_map(base_map_img, *, meta, pois=None, areas=None, vwalls=None, poi_radius_px=4):
    img = base_map_img.convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    ox, oy = meta["origin_x_m"], meta["origin_y_m"]
    res, rot, H = meta["res_m_per_px"], float(meta.get("rotation_deg", 0.0)), img.height

    # areas
    for a in (areas or []):
        pts = [world_to_pixel(x, y, origin_x_m=ox, origin_y_m=oy, res_m_per_px=res, img_h_px=H, rotation_deg=rot)
               for (x, y) in a.get("polygon", [])]
        if len(pts) >= 3:
            draw.polygon(pts, fill=COLORS["polygon.default"], outline=COLORS["polygon.outline"])

    # lines
    for w in (vwalls or []):
        pts = [world_to_pixel(x, y, origin_x_m=ox, origin_y_m=oy, res_m_per_px=res, img_h_px=H, rotation_deg=rot)
               for (x, y) in w.get("polyline", [])]
        if len(pts) >= 2:
            draw.line(pts, fill=COLORS["polyline.default"], width=3)

    # points
    for p in (pois or []):
        x, y = p["coordinate"]
        px, py = world_to_pixel(x, y, origin_x_m=ox, origin_y_m=oy, res_m_per_px=res, img_h_px=H, rotation_deg=rot)
        t = str(p.get("type") or "").strip()
        color = COLORS.get(f"poi.{t}", COLORS["poi.default"])
        r = poi_radius_px
        draw.ellipse((px-r, py-r, px+r, py+r), fill=color, outline=(0,0,0,200))
        name = p.get("name") or ""
        if name:
            draw.text((px + r + 2, py - 8), name, fill=(0,0,0,255))
    return img


LEGEND = [
    ("Trajectory / Lines", COLORS["polyline.default"]),
    ("Areas (regions)",    COLORS["polygon.outline"]),
    ("Charging (type 9)",  COLORS["poi.9"]),
    ("Waypoint (type 10)", COLORS["poi.10"]),
    ("Shelf (type 34)",    COLORS["poi.34"]),
    ("Dock (type 36)",     COLORS["poi.36"]),
    ("Stopover (type 38)", COLORS["poi.38"]),
    ("Robot",              (255,0,0,255)),
]

def _draw_legend(img: Image.Image, legend=LEGEND):
    draw = ImageDraw.Draw(img, "RGBA")
    pad, swatch = 8, 16
    x0, y0 = 10, 10
    # try default font
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    h = y0
    for label, color in legend:
        draw.rectangle((x0, h, x0+swatch, h+swatch), fill=color, outline=(0,0,0,200))
        draw.text((x0 + swatch + 6, h+1), label, fill=(0,0,0,255), font=font)
        h += swatch + pad

def draw_overlays_geojson(base_map_img: Image.Image, *, meta: dict, feats: list[dict]) -> Image.Image:
    """
    Draw GeoJSON features whose coordinates are in MAP METERS.
    We convert every vertex with world_to_pixel(origin/res/bearing).
    """
    img = base_map_img.convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    H = img.height

    ox = meta["origin_x_m"]
    oy = meta["origin_y_m"]
    res = meta["res_m_per_px"]
    rot = float(meta.get("rotation_deg", 0.0))

    # polygons first (fills), then polylines, then points on top
    for f in feats:
        if f["kind"] != "polygon":
            continue
        pts = [
            world_to_pixel(x, y,
                           origin_x_m=ox, origin_y_m=oy,
                           res_m_per_px=res, img_h_px=H, rotation_deg=rot)
            for (x, y) in f["coords"]
        ]
        if len(pts) >= 3:
            draw.polygon(pts, fill=COLORS["polygon.default"], outline=COLORS["polygon.outline"])

    for f in feats:
        if f["kind"] != "polyline":
            continue
        pts = [
            world_to_pixel(x, y,
                           origin_x_m=ox, origin_y_m=oy,
                           res_m_per_px=res, img_h_px=H, rotation_deg=rot)
            for (x, y) in f["coords"]
        ]
        if len(pts) >= 2:
            draw.line(pts, fill=COLORS["polyline.default"], width=3)

    for f in feats:
        if f["kind"] != "point":
            continue
        (x, y) = f["coords"][0]
        px, py = world_to_pixel(x, y,
                                origin_x_m=ox, origin_y_m=oy,
                                res_m_per_px=res, img_h_px=H, rotation_deg=rot)
        props = f.get("props", {})
        t = str(props.get("type", "")).strip()
        color = COLORS.get(f"poi.{t}", COLORS["poi.default"])
        r = 4
        draw.ellipse((px-r, py-r, px+r, py+r), fill=color, outline=(0,0,0,200))
        name = f.get("name") or props.get("name") or ""
        if name:
            draw.text((px + r + 2, py - 8), name, fill=(0,0,0,255))

    _draw_legend(img)
    return img


def _feats_scale_from_pois(feats: list[dict], pois_df) -> float | None:
    """
    Try to infer a unit scale between GeoJSON features (unknown units) and the
    POI table (meters). We look for name matches and compute median(|X_poi|/|X_feat|).
    Returns a scalar (e.g., 1000.0), or None if we cannot infer.
    """
    if pois_df is None or pois_df.empty:
        return None

    # build {lower_name: (x_m, y_m)}
    poi_xy = {}
    for _, row in pois_df.iterrows():
        name = str(row.get("name") or "").strip().lower()
        coord = row.get("coordinate")
        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
            px, py = float(coord[0]), float(coord[1])
            if name:
                poi_xy[name] = (px, py)

    ratios = []
    for f in feats:
        if f.get("kind") != "point":
            continue
        name = (f.get("name") or f.get("props", {}).get("name") or "").strip().lower()
        if not name or name not in poi_xy:
            continue
        (fx, fy) = f["coords"][0]
        (px, py) = poi_xy[name]
        # ignore near-zero to avoid blowing up ratios
        for a, b in ((abs(px), abs(fx)), (abs(py), abs(fy))):
            if b > 1e-9 and a > 0:
                ratios.append(a / b)

    if not ratios:
        return None

    # robust middle
    ratios.sort()
    mid = ratios[len(ratios)//2]
    return float(mid)

def _scale_feats_inplace(feats: list[dict], scale: float) -> None:
    if not scale or abs(scale - 1.0) < 1e-9:
        return
    for f in feats:
        newc = []
        for (x, y) in f["coords"]:
            newc.append((x * scale, y * scale))
        f["coords"] = newc

def render_full_map(robot_sn: str, out_png="map_with_layers.png"):
    area_id = _get_area_id_from_robot_sn(robot_sn)
    if not area_id:
        raise RuntimeError("No areaId for robotSn; cannot call /area/{areaId}/...")

    base = get_base_map_image_by_area(area_id)
    meta_raw = get_map_meta(area_id=area_id, robot_sn=robot_sn)
    meta, _ = normalize_map_meta(meta_raw)

    feat_raw = get_map_features(area_id=area_id, robot_sn=robot_sn)
    feats = normalize_features_geojson(feat_raw)

    # ---- NEW: auto-scale GeoJSON coords to meters using POI table ----
    try:
        pois_df = Robot(robot_sn).get_pois()  # your POI list (meters)
    except Exception:
        pois_df = None
    scale = _feats_scale_from_pois(feats, pois_df)
    if scale:
        log.info(f"Applying feature scale factor ≈ {scale:.3f}")
        _scale_feats_inplace(feats, scale)
    else:
        # fallback heuristic (what you had): km → m if everything is tiny
        flat = [abs(v) for f in feats for (x, y) in f["coords"] for v in (x, y)]
        if flat and max(flat) < 0.02:
            log.info("Applying fallback ×1000 (kilometers → meters)")
            _scale_feats_inplace(feats, 1000.0)

    img = draw_overlays_geojson(base, meta=meta, feats=feats)

    pose = get_robot_pose(robot_sn)
    draw_robot_arrow(img, meta, pose, length_m=0.8, color=(255,0,0,220))

    img.save(out_png)
    log.info(f"Wrote: {out_png} (features={len(feats)})")
    return out_png

def pixel_to_world(px, py_screen, *, origin_x_m, origin_y_m, res_m_per_px, img_h_px, rotation_deg=0.0):
    """
    Inverse of world_to_pixel(). Accepts screen Y (origin top, downwards).
    Returns world meters (x_m, y_m).
    """
    # un-flip Y
    py = img_h_px - py_screen
    dxr, dyr = px * res_m_per_px, py * res_m_per_px
    if rotation_deg:
        th = math.radians(rotation_deg)
        c, s = math.cos(th), math.sin(th)
        # inverse rotation: [dx;dy] = [ c  s; -s  c ] [dxr;dyr]
        dx = c * dxr + s * dyr
        dy = -s * dxr + c * dyr
    else:
        dx, dy = dxr, dyr
    return (origin_x_m + dx, origin_y_m + dy)

import heapq

def _astar_on_cost(cost: np.ndarray, start_rc, goal_rc, block_threshold: float = 0.99):
    """
    A* on a float32 cost grid in [0,1]. start_rc/goal_rc: (row, col) integers.
    A cell is blocked if cost>=block_threshold.
    8-connected moves, move cost = base * (1 + 10*cell_cost_at_dest).
    Returns list of (row, col) or [] if no path.
    """
    H, W = cost.shape
    def inb(r,c): return 0 <= r < H and 0 <= c < W
    if not (inb(*start_rc) and inb(*goal_rc)): return []
    if cost[start_rc] >= block_threshold or cost[goal_rc] >= block_threshold: return []

    # precompute penalty and allowed mask
    pen = (1.0 + 10.0 * np.clip(cost, 0.0, 1.0)).astype(np.float32)
    blocked = cost >= block_threshold

    nbrs = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
            (-1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),(1,-1,math.sqrt(2)),(1,1,math.sqrt(2))]

    def h(r,c):
        dr = goal_rc[0]-r; dc = goal_rc[1]-c
        return math.hypot(dr, dc)

    start = start_rc
    goal = goal_rc

    g = {start: 0.0}
    parent = {start: None}
    openq = []
    heapq.heappush(openq, (h(*start), 0.0, start))

    while openq:
        f, gcur, cur = heapq.heappop(openq)
        if cur == goal:
            # reconstruct
            path = []
            n = cur
            while n is not None:
                path.append(n)
                n = parent[n]
            path.reverse()
            return path

        if gcur > g.get(cur, float('inf')):  # stale
            continue

        r, c = cur
        for dr, dc, w in nbrs:
            nr, nc = r+dr, c+dc
            if not inb(nr, nc): continue
            if blocked[nr, nc]: continue
            gcand = gcur + w * float(pen[nr, nc])
            if gcand < g.get((nr, nc), float('inf')):
                g[(nr, nc)] = gcand
                parent[(nr, nc)] = cur
                heapq.heappush(openq, (gcand + h(nr, nc), gcand, (nr, nc)))
    return []

def _draw_path_pixels(img: Image.Image, pts_px: list[tuple[float,float]], color=(30, 144, 255, 255), width=4):
    dr = ImageDraw.Draw(img, "RGBA")
    if len(pts_px) >= 2:
        dr.line(pts_px, fill=color, width=width, joint="curve")
    # endpoints
    if pts_px:
        r = 5
        x0,y0 = pts_px[0]
        dr.ellipse((x0-r,y0-r,x0+r,y0+r), outline=(0,0,0,220), fill=(0,255,0,220))
        x1,y1 = pts_px[-1]
        dr.ellipse((x1-r,y1-r,x1+r,y1+r), outline=(0,0,0,220), fill=(255,0,0,220))


def follow_robot(robot_sn: str, out_png="map_live.png", interval=5, duration=None):
    area_id = _get_area_id_from_robot_sn(robot_sn)
    if not area_id:
        raise RuntimeError("No areaId for robotSn; cannot call /area/{areaId}/...")

    base = get_base_map_image_by_area(area_id)
    meta_raw = get_map_meta(area_id=area_id, robot_sn=robot_sn)
    meta, _ = normalize_map_meta(meta_raw)
    feat_raw = get_map_features(area_id=area_id, robot_sn=robot_sn)
    feats = normalize_features_geojson(feat_raw)

    # scale features once
    try:
        pois_df = Robot(robot_sn).get_pois()
    except Exception:
        pois_df = None
    scale = _feats_scale_from_pois(feats, pois_df)
    if scale:
        log.info(f"Applying feature scale factor ≈ {scale:.3f}")
        _scale_feats_inplace(feats, scale)
    else:
        flat = [abs(v) for f in feats for (x, y) in f["coords"] for v in (x, y)]
        if flat and max(flat) < 0.02:
            log.info("Applying fallback ×1000 (kilometers → meters)")
            _scale_feats_inplace(feats, 1000.0)

    static_img = draw_overlays_geojson(base, meta=meta, feats=feats)

    t0 = time.time()
    i = 0
    while True:
        img = static_img.copy()
        pose = get_robot_pose(robot_sn)
        draw_robot_arrow(img, meta, pose, length_m=0.8, color=(255,0,0,220))
        img.save(out_png)
        i += 1
        log.info(f"[{i}] updated {out_png}  x={pose['x']:.2f} y={pose['y']:.2f} yaw={pose['yaw']:.2f}")
        if duration and (time.time() - t0) >= duration:
            break
        time.sleep(max(0.1, float(interval)))




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
        try:
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
        except KeyError:
            return {
                "isOnLine": self.df.isOnLine,
                # "isEmergencyStop": status.loc['isEmergencyStop','data'],
                # "version": status.loc['vers','data'],
                # "hasObstruction": status.loc['hasObstruction','data'],
                "isCharging": self.df.isCharging,
                # "battery": status.loc['battery','data'],
                "task": self.get_task(),
                # "pose": self.get_pose(),
                # "speed": status.loc['speed','data'],
                "relPos": None,
                "isAt": None,
                # "errors": status.loc['errors','data']
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

from PIL import ImageFilter

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
        self._ctx = {
            "pois": pois,
            "areas": areas_df,
            "areas_rich": areas_rich,
            "lines": _lines_df,              # <-- add this
            "scale": scale
        }
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
               stopRadius: float | None = None, extra_ext: dict | None = None, areaDelivery=False):
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
        
        if areaDelivery:
            shelf_area_id = det.get("properties").get("relatedShelvesAreaId")
            acts[0]["data"] = {"useAreaId": shelf_area_id}
            pt = Task._mk_point(x=x, y=y, yaw=yaw, areaId=det.get("areaId"), ext={"id": shelf_area_id}, stepActs=acts)
        else :
            pt = Task._mk_point(x=x, y=y, yaw=yaw, areaId=det.get("areaId"), ext=ext, stepActs=acts)

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

    # def drop_at_dock(self, dock_name: str, *, extra_ext: dict | None = None):
    #     det = self._require_poi_type(dock_name, {Task.TYPE_DOCK}, "drop_at_dock")
    #     x, y, yaw = self._poi_pose(dock_name)
    #     ext = {"name": dock_name, "id": det.get("id")}
    #     if extra_ext:
    #         ext.update(extra_ext)
    #     pt = Task._mk_point(x=x, y=y, yaw=yaw, areaId=det.get("areaId"), ext=ext)
    #     self._taskPts.append(pt)
    #     return self
    
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
    # def dropdown(self, target_name: str, *, useArea: bool = False, lift: str | None = None):
    #     if useArea:
    #         return self.to_area(target_name, lift=lift)
    #     return self.drop_at_dock(target_name)

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

# --- cost → colored background (low=black, high=red) ---
def cost_to_rgba(cost: np.ndarray) -> Image.Image:
    """
    cost: float32 [0..1], HxW
    returns RGBA image with cost mapped to red channel: (R=255*cost, G=0, B=0).
    """
    cost_u8 = (np.clip(cost, 0.0, 1.0) * 255).astype(np.uint8)
    H, W = cost.shape
    R = Image.fromarray(cost_u8, mode="L")
    Z = Image.new("L", (W, H), 0)
    A = Image.new("L", (W, H), 255)
    return Image.merge("RGBA", (R, Z, Z, A))

def _polyline_length(pts):
    # pts: [(x,y), ...]
    return sum(math.hypot(x2 - x1, y2 - y1) for (x1,y1), (x2,y2) in zip(pts, pts[1:]))

# --- dashed polyline in pixel space ---
def draw_dashed_polyline(draw: ImageDraw.ImageDraw, pts_px, *, dash_px=8, gap_px=6, width=3, color=(255,255,0,255)):
    """
    pts_px: list of (x,y). Draws a dashed line along all segments.
    """
    import math
    if not pts_px or len(pts_px) < 2: return
    dash = float(max(1, dash_px)); gap = float(max(1, gap_px))
    pattern = [dash, gap]  # on, off
    pat_len = sum(pattern)

    for i in range(len(pts_px)-1):
        (x1, y1), (x2, y2) = pts_px[i], pts_px[i+1]
        vx, vy = x2 - x1, y2 - y1
        seg_len = math.hypot(vx, vy)
        if seg_len <= 1e-6: 
            continue
        ux, uy = vx/seg_len, vy/seg_len

        # walk the segment with on/off pattern
        dist = 0.0
        on = True
        pat_idx = 0
        rem = pattern[pat_idx]
        curx, cury = x1, y1
        while dist < seg_len:
            step = min(rem, seg_len - dist)
            nx, ny = curx + ux*step, cury + uy*step
            if on:
                draw.line([(curx, cury), (nx, ny)], fill=color, width=width, joint="curve")
            curx, cury = nx, ny
            dist += step
            pat_idx = (pat_idx + 1) % 2
            rem = pattern[pat_idx]
            on = (pat_idx == 0)



class Robot_v2(Robot_v1):


    def get_env(
        self,
        *,
        dark_thresh: int = 80,          # base-map grayscale threshold (0..255): lower = darker → obstacle
        robot_radius_m: float = 0.25,   # inflate obstacles by this radius (meters)
        line_width_m: float = 0.10,     # thickness to rasterize lineType==2 walls (meters)
        base_weight: float = 0.4,       # weight for dark pixels
        area_weight: float = 1.0,       # weight for forbidden areas
        line_weight: float = 1.0,       # weight for virtual walls
        blur_px: int = 0,               # optional final smoothing (pixels)
        return_uint8: bool = False      # if True → uint8 [0..255]; else float32 [0..1]
    ) -> np.ndarray:
        """
        Build a cost map from base map darkness + forbidden areas (regionType==1) + virtual walls (lineType==2).
        Returns HxW numpy array (float32 0..1 by default).
        """
        # --- fetch map + meta
        area_id = self.df.areaId
        base_img = get_base_map_image_by_area(area_id)  # PIL
        meta_raw = get_map_meta(area_id=area_id, robot_sn=self.SN)
        meta, _ = normalize_map_meta(meta_raw)
        if not {"origin_x_m","origin_y_m","res_m_per_px"}.issubset(meta.keys()):
            raise RuntimeError("map meta missing required fields (origin_x_m, origin_y_m, res_m_per_px)")

        ox, oy = meta["origin_x_m"], meta["origin_y_m"]
        res = meta["res_m_per_px"]
        rot = float(meta.get("rotation_deg", 0.0))
        W, H = base_img.width, base_img.height

        ctx = self._refresh_context()
        areas_df = ctx.get("areas")
        lines_df = ctx.get("lines")

        # --- base darkness mask (binary 0/255)
        gray = base_img.convert("L")
        dark_mask = np.asarray(gray) < int(dark_thresh)
        base_mask = (dark_mask.astype(np.uint8) * 255)

        # --- helpers to rasterize (PIL draw in image pixel space)
        def world_poly_to_px(poly):
            return [
                world_to_pixel(x, y, origin_x_m=ox, origin_y_m=oy,
                               res_m_per_px=res, img_h_px=H, rotation_deg=rot)
                for (x, y) in poly
            ]

        def rasterize_polygons(polys):
            if not polys:
                return np.zeros((H, W), dtype=np.uint8)
            m = Image.new("L", (W, H), 0)
            dr = ImageDraw.Draw(m, "L")
            for ring in polys:
                if len(ring) >= 3:
                    dr.polygon(world_poly_to_px(ring), fill=255, outline=255)
            return np.asarray(m, dtype=np.uint8)

        def rasterize_lines(lines, width_m):
            if not lines:
                return np.zeros((H, W), dtype=np.uint8)
            px_w = max(1, int(round(width_m / res)))
            m = Image.new("L", (W, H), 0)
            dr = ImageDraw.Draw(m, "L")
            for seg in lines:
                pts = world_poly_to_px(seg)
                if len(pts) >= 2:
                    dr.line(pts, fill=255, width=px_w, joint="curve")
            return np.asarray(m, dtype=np.uint8)

        # --- forbidden areas (regionType == 1)
        forb_polys = []
        if isinstance(areas_df, pd.DataFrame) and not areas_df.empty:
            forb = areas_df[(areas_df["regionType"].astype(str) == "1") & areas_df["polygon"].notna()]
            forb_polys = forb["polygon"].tolist()
        forb_mask = rasterize_polygons(forb_polys)

        # --- virtual walls: lineType == '2'
        wall_lines = []
        if isinstance(lines_df, pd.DataFrame) and not lines_df.empty:
            walls = lines_df[(lines_df["lineType"].astype(str) == "2") & lines_df["polyline"].notna()]
            wall_lines = walls["polyline"].tolist()
        wall_mask = rasterize_lines(wall_lines, width_m=line_width_m)

        # --- combine weighted → cost in [0,1]
        # normalize masks to 0..1 then weighted sum, clamp
        base_f = (base_mask.astype(np.float32) / 255.0) * base_weight
        forb_f = (forb_mask.astype(np.float32) / 255.0) * area_weight
        wall_f = (wall_mask.astype(np.float32) / 255.0) * line_weight
        cost = np.clip(base_f + forb_f + wall_f, 0.0, 1.0)

        # --- inflate by robot radius (MaxFilter on obstacle-like portion)
        inflate_px = int(np.ceil(float(robot_radius_m) / float(res)))
        if inflate_px > 0:
            # build a binary obstacle mask from cost>0, dilate, then blend back to max(cost, inflated)
            obs = (cost > 0).astype(np.uint8) * 255
            m = Image.fromarray(obs, mode="L")
            # kernel size must be odd
            k = max(1, inflate_px * 2 + 1)
            m = m.filter(ImageFilter.MaxFilter(size=k))
            inflated = (np.asarray(m, dtype=np.uint8) > 0).astype(np.float32)
            cost = np.maximum(cost, inflated)

        # --- optional blur (visual nicety / softening)
        if blur_px and blur_px > 0:
            from PIL import ImageFilter as IF
            imgc = Image.fromarray((np.clip(cost, 0, 1) * 255).astype(np.uint8), mode="L")
            imgc = imgc.filter(IF.GaussianBlur(radius=float(blur_px)))
            cost = np.asarray(imgc, dtype=np.uint8).astype(np.float32) / 255.0

        # --- return desired dtype
        if return_uint8:
            return (np.clip(cost, 0, 1) * 255).astype(np.uint8)
        return np.clip(cost, 0, 1).astype(np.float32)


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
        task = Task(self, "pickup", taskType="factory",runType="lift").pickup(poi_name, lift_up=True, areaDelivery=area_delivery)
        resp = create_task(**task.task_dict)
        return self

    def dropdown_at(self, poi_name, area_delivery=False):
        task = Task(self, "dropdown", taskType="factory",runType="lift").pickup(poi_name, lift_down=True, areaDelivery=area_delivery)
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

    # def shelf_to_shelf(self, pickup_shelf, dropdown_shelf, area_delivery=False):
    #     task = (
    #         Task(self, "shelf_delivery", taskType="factory",runType="lift")
    #         .pickup(pickup_shelf, lift_up=True)
    #         .dropdown(dropdown_shelf, useArea=area_delivery)
    #     )
    #     resp = create_task(**task.task_dict)
    #     return self
    
    def evacuate(self, area_name=None, evac_pts=[]):
        pass

    def wrap(self):
        pass

    # def plan_path(self, poi_name: str, out_png: str = "map_with_layers_with_path.png", *, block_threshold: float = 0.99):
    #     """
    #     Plans a grid path from the robot's current pose to a POI by name using A* on get_env().
    #     Draws the path over base+overlays and saves PNG. Returns dict with png and path points.
    #     """
    #     # --- map + meta
    #     area_id = self.df.areaId
    #     base = get_base_map_image_by_area(area_id)
    #     meta_raw = get_map_meta(area_id=area_id, robot_sn=self.SN)
    #     meta, _ = normalize_map_meta(meta_raw)
    #     if not {"origin_x_m","origin_y_m","res_m_per_px"}.issubset(meta.keys()):
    #         raise RuntimeError("map meta missing required fields (origin_x_m, origin_y_m, res_m_per_px)")
    #     ox, oy = meta["origin_x_m"], meta["origin_y_m"]
    #     res = meta["res_m_per_px"]
    #     rot = float(meta.get("rotation_deg", 0.0))
    #     W, H = base.width, base.height

    #     # --- overlays (areas/lines/points as meters-aware GeoJSON)
    #     feat_raw = get_map_features(area_id=area_id, robot_sn=self.SN)
    #     feats = normalize_features_geojson(feat_raw)
    #     # scale features to meters (matches your other funcs)
    #     try:
    #         pois_df = self.get_pois()
    #     except Exception:
    #         pois_df = None
    #     scale = _feats_scale_from_pois(feats, pois_df)
    #     if scale:
    #         _scale_feats_inplace(feats, scale)
    #     else:
    #         flat = [abs(v) for f in feats for (x, y) in f["coords"] for v in (x, y)]
    #         if flat and max(flat) < 0.02:
    #             _scale_feats_inplace(feats, 1000.0)
    #     img = draw_overlays_geojson(base, meta=meta, feats=feats)

    #     # --- current pose (meters)
    #     pose = get_robot_pose(self.SN)
    #     sx, sy = float(pose["x"]), float(pose["y"])
    #     # --- goal from POI name (meters)
    #     poi_df = self.get_pois()
    #     if isinstance(poi_df, pd.DataFrame):
    #         target = poi_df[poi_df["name"].astype(str).str.strip() == str(poi_name).strip()]
    #         if target.empty:
    #             raise RuntimeError(f"POI '{poi_name}' not found")
    #         gx, gy = float(target.iloc[0]["coordinate"][0]), float(target.iloc[0]["coordinate"][1])
    #     else:
    #         raise RuntimeError("POI table unavailable")

    #     # --- convert to pixel coords (screen-space Y)
    #     s_px = world_to_pixel(sx, sy, origin_x_m=ox, origin_y_m=oy, res_m_per_px=res, img_h_px=H, rotation_deg=rot)
    #     g_px = world_to_pixel(gx, gy, origin_x_m=ox, origin_y_m=oy, res_m_per_px=res, img_h_px=H, rotation_deg=rot)

    #     # --- cost grid
    #     cost = self.get_env(return_uint8=False)  # float32 [0..1]
    #     Hc, Wc = cost.shape
    #     if (Hc, Wc) != (H, W):
    #         # safety: resize cost to raster size if needed
    #         cost_img = Image.fromarray((np.clip(cost, 0, 1) * 255).astype(np.uint8), mode="L").resize((W, H), Image.NEAREST)
    #         cost = np.asarray(cost_img, dtype=np.uint8).astype(np.float32) / 255.0

    def plan_path(self, poi_name: str, out_png: str = "map_with_layers_with_path.png", *, block_threshold: float = 0.99):
        """
        Draw on top of the COST image (low=black, high=red):
          - type==1 lines dashed yellow (scaled correctly)
          - A* path in blue
          - robot (green circle), POI (purple circle + label)
        Geometry is anchored to the cost grid size to avoid scale/offset drift.
        """
        # --- META (origin/res/rot)
        area_id = self.df.areaId
        meta_raw = get_map_meta(area_id=area_id, robot_sn=self.SN)
        meta, _ = normalize_map_meta(meta_raw)
        req = {"origin_x_m","origin_y_m","res_m_per_px"}
        if not req.issubset(meta.keys()):
            raise RuntimeError(f"map meta missing {req - set(meta.keys())}")
        ox, oy = meta["origin_x_m"], meta["origin_y_m"]
        res    = meta["res_m_per_px"]
        rot    = float(meta.get("rotation_deg", 0.0))

        # --- COST (defines raster H,W)
        cost = self.get_env(return_uint8=False)  # float32 [0..1], HxW
        H, W = cost.shape

        # Make the image we’ll draw on
        img = cost_to_rgba(cost)
        dr  = ImageDraw.Draw(img, "RGBA")

        # --- Current pose & target POI (world meters)
        pose = get_robot_pose(self.SN)
        sx, sy = float(pose["x"]), float(pose["y"])

        poi_df = self.get_pois()
        if not isinstance(poi_df, pd.DataFrame) or poi_df.empty:
            raise RuntimeError("POI table unavailable/empty")
        target = poi_df[poi_df["name"].astype(str).str.strip() == str(poi_name).strip()]
        if target.empty:
            raise RuntimeError(f"POI '{poi_name}' not found")
        gx, gy = float(target.iloc[0]["coordinate"][0]), float(target.iloc[0]["coordinate"][1])

        # --- World→Pixel using COST height (H) as the screen height
        s_px = world_to_pixel(sx, sy, origin_x_m=ox, origin_y_m=oy, res_m_per_px=res, img_h_px=H, rotation_deg=rot)
        g_px = world_to_pixel(gx, gy, origin_x_m=ox, origin_y_m=oy, res_m_per_px=res, img_h_px=H, rotation_deg=rot)

        # --- A* over cost
        clamp = lambda p: (min(max(int(round(p[0])), 0), W-1), min(max(int(round(p[1])), 0), H-1))
        sxi, syi = clamp(s_px); gxi, gyi = clamp(g_px)
        path_rc  = _astar_on_cost(cost, (syi, sxi), (gyi, gxi), block_threshold=block_threshold)

        # --- Get **scaled** features (same scale as used elsewhere), then draw type==1 lines
        feat_raw = get_map_features(area_id=area_id, robot_sn=self.SN)
        feats    = normalize_features_geojson(feat_raw)  # coords likely need scaling (m vs km)
        # Scale features to meters using POI-anchored factor (exactly like your overlay code)
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

        # Extract type==1 lines from feats.props.lineType
        typ1_lines = []
        for f in feats:
            if f.get("kind") != "polyline":
                continue
            lt = str((f.get("props") or {}).get("lineType", "")).strip()
            if lt == "1":
                typ1_lines.append(f["coords"])

        # Draw dashed yellow lines in *pixel* space using same origin/res/rot and same H
        for seg in typ1_lines:
            pts_px = [world_to_pixel(x, y, origin_x_m=ox, origin_y_m=oy, res_m_per_px=res, img_h_px=H, rotation_deg=rot)
                      for (x, y) in seg]
            draw_dashed_polyline(dr, pts_px, dash_px=10, gap_px=6, width=3, color=(255, 230, 0, 255))

        # --- Path (blue) or fallback straight hint if no path
        if path_rc:
            path_px = [(float(c), float(r)) for (r, c) in path_rc]
            _draw_path_pixels(img, path_px, color=(0, 102, 255, 255), width=4)
        else:
            _draw_path_pixels(img, [s_px, g_px], color=(120, 120, 120, 180), width=2)

        # --- Robot & POI markers (green / purple) + label
        def circ(draw, center, r, fill, outline=(0,0,0,220)):
            x,y = center; draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline=outline)
        circ(dr, s_px, 6, fill=(0, 200, 0, 255))       # robot = green
        circ(dr, g_px, 6, fill=(160, 0, 200, 255))     # poi   = purple
        dr.text((g_px[0]+8, g_px[1]-10), str(target.iloc[0].get("name") or poi_name), fill=(255,255,255,255))

        # --- Save & return world path
        img.save(out_png)

        path_world = []
        if path_rc:
            for (r, c) in path_rc:
                wx, wy = pixel_to_world(float(c), float(r),
                                        origin_x_m=ox, origin_y_m=oy,
                                        res_m_per_px=res, img_h_px=H, rotation_deg=rot)
                path_world.append((wx, wy))

        
        length_m  = _polyline_length(path_world) if path_world else 0.0
        length_px = _polyline_length(path_px)    if path_rc    else 0.0  # if you want pixels too

        return {
            "png": out_png,
            "pixels": (path_px if path_rc else []),
            "world": path_world,
            "length_m": length_m,    # total path length in meters
            "length_px": length_px,  # (optional) same in pixels
        }



        # --- clamp and index as (row, col)
        def _clamp_pt(p):
            x, y = p
            return (min(max(int(round(x)), 0), W-1),
                    min(max(int(round(y)), 0), H-1))
        sxi, syi = _clamp_pt(s_px)
        gxi, gyi = _clamp_pt(g_px)
        start_rc = (syi, sxi)
        goal_rc  = (gyi, gxi)

        # --- A* path (list of (row, col))
        path_rc = _astar_on_cost(cost, start_rc, goal_rc, block_threshold=block_threshold)
        if not path_rc:
            # still write an image with start/goal
            _draw_path_pixels(img, [s_px, g_px], color=(200,200,200,200), width=2)
            img.save(out_png)
            log.warning("No path found; saved straight-line placeholder.")
            return {"png": out_png, "pixels": [], "world": []}

        # --- pixels (x,y) for drawing
        path_px = [(float(c), float(r)) for (r, c) in path_rc]

        # --- draw path
        _draw_path_pixels(img, path_px, color=(30,144,255,255), width=4)

        # --- also draw the robot arrow on top for direction
        draw_robot_arrow(img, meta, pose, length_m=0.8, color=(255,0,0,220))

        # --- save
        img.save(out_png)

        # --- convert to world meters
        path_world = [
            pixel_to_world(px, py,
                           origin_x_m=ox, origin_y_m=oy,
                           res_m_per_px=res, img_h_px=H, rotation_deg=rot)
            for (px, py) in path_px
        ]

        return {"png": out_png, "pixels": path_px, "world": path_world}


    # def say(self, audio_dict):
    #     if audio_dict["mp3_id"] is None:
    #         ...
    #     else :
    #         ...   


Robot = Robot_v2