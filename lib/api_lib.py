
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
except ImportError:
    pd = None  # Only used if caller asks for DataFrame

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

########### Decorators
# def api_post(route_key):
#     """
#     Decorator: turns a payload-factory function into an API call.
#     The wrapped function should return a dict payload (or None).
#     """
#     def deco(func):
#         def wrapped(
#             *,
#             base_url,
#             routes,
#             token_provider,
#             timeout=10.0,
#             max_retries=3,
#             as_dataframe=False,
#             **kwargs
#         ):
#             url = _build_url(base_url, routes[route_key])
#             headers = {
#                 "Accept": "application/json",
#                 "Content-Type": "application/json",
#                 "X-Token": token_provider(),
#             }
#             payload = func(**kwargs) or {}

#             data = None
#             for attempt in range(1, max_retries + 1):
#                 try:
#                     log.debug("POST %s attempt=%d payload=%s", url, attempt, payload)
#                     r = requests.post(url, json=payload, headers=headers, timeout=timeout)
#                 except requests.RequestException as e:
#                     log.warning("Network error on attempt %d: %s", attempt, e)
#                 else:
#                     try:
#                         data = r.json()
#                     except ValueError:
#                         log.error("Non-JSON response from %s: %s", url, r.text[:300])
#                         time.sleep(1); continue
#                     if r.status_code in (401, 403):
#                         raise RuntimeError(f"Auth failed ({r.status_code})")
#                     if r.ok:
#                         break
#                     log.warning("API error on attempt %d: %s", attempt, data)
#                 time.sleep(1)
#             else:
#                 raise RuntimeError(f"Failed to POST to {url} after {max_retries} attempts")

#             lists = (((data or {}).get("data") or {}).get("lists")) or []
#             if not isinstance(lists, list):
#                 raise RuntimeError("'data.lists' is not a list")

#             if as_dataframe:
#                 try:
#                     import pandas as pd
#                 except ImportError:
#                     raise ImportError("pandas not installed")
#                 return pd.DataFrame(lists)

#             return lists
#         return wrapped
#     return deco

# # --- Use it: your function only defines the payload (if any) ---
# @api_post("task_list")
# def get_tasks_payload(payload=None):
#     # do any payload shaping/validation here; or just pass through
#     return payload or {}

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

# def create_task(task_name, robot, runType, sourceType, taskPts=[], runNum=1, taskType=5, routeMode=1, runMode=1, ignorePublicSite=False, speed=0.4, detourRadius=1, backPt={}):
#     task_dict = {
#       "name": task_name,
#       "robotId": robot.robotId,
#       "runNum": runNum,
#       "taskType": taskType,
#       "runType": runType, # 29 for lifting
#       "routeMode": routeMode, # 1 for sequential routing , 2 for shortest distance routing
#       "runMode": runMode, # 1 flexible obst avoi
#       "taskPts": taskPts,
#       "businessId": robot.df.businessId,
#       "sourceType": sourceType, # 3, # for pager
#       "ignorePublicSite": ignorePublicSite,
#       "speed": speed, # 0.4 to 1
#       "detourRadius": detourRadius, # safety dist in meters
#       "backPt": backPt
#     }

#     url = _build_url(URL_BASE, URL_ROUTING_DICT['create_task'])
#     r = requests.post(
#         url,
#         headers = {
#             "X-Token" : X_TOKEN
#         },
#         json = task_dict
#     )
#     try:
#       return pd.DataFrame(r.json()["data"])
#     except Exception as exp:
#         # log.exception("api abstraction error")
#     # raise NotImplementedError
#     #   rinspect(r)
#         return r.text


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

# class Client():
#     def __init__(self):
#         ...
#     def _get():
#     def _post_json():

#     def get_buildings():
#     def get_business():
#     def get_robots(): # as classes

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
    from ws_ext import AutoXingWSOversee
    from api_lib import get_token_key   # must return the short 'key' from auth API
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
