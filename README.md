# AutoXing Python SDK (api_lib)

This repository lets you talk to AutoXing robots over REST/WebSocket, inspect maps & POIs, plan paths, and create/dispatch tasks. It supersedes the legacy `AX_PY_SDK`.

## Features
- **Auth helpers:** `get_token`, `get_token_key`
- **REST wrappers:** businesses, robots, maps, tasks
- **High-level `Robot_v2` class:**
  - live status/pose
  - POIs, Areas, and Lines (GeoJSON normalization)
  - map rendering with overlays
  - costmap generation
  - grid A* path planning (meters-aware)
  - fluent `Task` builder for delivery/lift/area actions
- **Evacuation utilities** with regex-based POI filtering
- **Examples** in `examples/` (`tasks.py`, `get_robot_state.py`)

## Requirements
- Python **3.10+** (tested on 3.11)
- Linux/macOS/Windows; Raspberry Pi supported for the GPIO alarm example
- AutoXing API credentials

### Python dependencies
```bash
pip install requests python-dotenv pillow numpy pandas rich
# Raspberry Pi (optional alarm example)
pip install RPi.GPIO


### Setup
APPID=your_app_id
APPSECRET=your_app_secret
APPCODE=your_app_code   # used in Authorization header



### Quickstart

from api_lib import Robot, get_ef_robots

# pick an EF robot (online preferred)
rid = get_ef_robots().iloc[0].robotId
bot = Robot(rid)

# status
st = bot.get_state()  # dict with pose, battery, isAt tables, etc.
print(st["isOnLine"], st["battery"], st["pose"])

# go to a POI by name
bot.go_to_poi("Warten")

# go charge (first charger POI)
bot.go_charge()

# lift up at a shelf POI and drop at another (area delivery)
bot.pickup_at("Sicht 1", area_delivery=False)
bot.dropdown_at("Sicht 2", area_delivery=True)


### API Overview

get_token_and_key() → {"token": <REST>, "key": <WS subprotocol>, "expireTime": ...}

get_token() / get_token_key() convenience wrappers.


### DataFrames

get_business(name=None) → pandas.DataFrame

get_buildings()

get_robots(robot_id=None) (joined with business name)

get_online_robots(), get_ef_robots()


### MAP & POIs

get_map_meta(area_id, robot_sn) → origin, resolution, rotation

get_map_features(area_id, robot_sn) → GeoJSON FeatureCollection

get_map_image(area_id) → base raster

Normalizers: normalize_map_meta, normalize_features_geojson, build_feature_tables

Drawing: draw_overlays_geojson, draw_robot_arrow


### Tasks

create_task(...), execute_task(task_id), cancel_task(task_id)

get_tasks(), get_task_details(task_id), get_task_status(task_id)


### Robot Class

from api_lib import Robot

bot = Robot("2382306702057xb")

# State with proximity logic (meters)
state = bot.get_state(poi_threshold=0.5, min_dist_area=1.0)
print(state["isAt"])  # DataFrame for POIs/areas you're "at"

# Costmap (0..1 float)
env = bot.get_env(dark_thresh=80, robot_radius_m=0.25, line_width_m=0.10)

# Plan A* path to POI; draws path on PNG
plan = bot.plan_path("Evac1")
print(plan["length_m"], plan["png"])

# Motion helpers (post tasks under the hood)
bot.go_to_poi("Warten")
bot.wait_at("Warten", 10)
bot.go_charge()
bot.go_back()
bot.go_to_pose((0.0, 0.0, 0.0))


### Task Builder

from api_lib import Task, Robot, create_task

bot = Robot("...")

task = (
  Task(bot, "area_job", taskType="factory", runType="lift")
    .pickup("Sicht 1", lift_up=True)          # shelf POI (type 34)
    .pickup("Sicht 2", lift_down=True)        # drop action
    .to_area("Zone A", lift="down")           # area centroid with lift act
    .back("Warten")                           # optional return
)

resp = create_task(**task.task_dict)
print(resp)  # {"taskId": "..."}



### Common enums

taskType: "delivery" | "factory" | "return_to_dock" | ...

runType: "roam" | "lift" | "charging_station" | ...

Other: RUN_MODE, ROUTE_MODE, SOURCE_TYPE



