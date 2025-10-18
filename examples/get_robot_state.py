import sys, os
sys.path.append(os.path.join(os.getcwd(),"lib"))
from api_lib import *

ef_robots = get_ef_robots()
business_robots = get_business_robots("EF")
rprint(business_robots)

rid = ef_robots[ef_robots.robotId.str.endswith("539EX")].iloc[0].robotId

robot = Robot(rid)
rprint(robot)

state = robot.get_state()

rprint(state)
