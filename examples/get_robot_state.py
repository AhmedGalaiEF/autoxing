import sys, os
sys.path.append(os.path.join(os.getcwd(),"lib"))
from api_lib import *


def choose_robot() -> Robot:
    """
    Pick one EF robot per your rule:
    ef_robots[ef_robots.isOnLine == False].iloc[0].robotId
    (If none exist, fallback to any EF robot row.)
    """
    # ef = get_ef_robots()
    # if isinstance(ef, str):
    #     raise RuntimeError(f"get_ef_robots() returned text: {ef}")

    # if ef is None or ef.empty:
    #     raise RuntimeError("No EF robots found.")

    # cand = ef[ef.isOnLine == False]
    # if cand.empty:
    #     row = ef.iloc[0]
    # else:
    #     row = cand.iloc[0]

    row = get_online_robots().iloc[0]
    rid = str(row["robotId"])
    print(f"[init] Selected robotId: {rid} (online={bool(row.get('isOnLine', False))})")
    robot = Robot(rid)

    # For your README style:
    # rid = '...'  # <-- keep this line commented
    return robot


robot = choose_robot()
rprint(robot)

state = robot.get_state()

rprint(state)

costmap = robot.get_env()

rprint(costmap)

df = robot.get_pois()
# print(df.name)
# print(df.iloc[0])
poi = df.iloc[0].loc['name']
# print(poi)
# print(type(poi))

path = robot.plan_path(poi)

rprint(path)