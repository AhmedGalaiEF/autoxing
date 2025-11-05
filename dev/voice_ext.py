import os, sys
sys.path.append(r"C:\Users\Ahmed Galai\Desktop\dev\Sandbox\gitrepo\autoxing\lib")
from api_lib import *


# In [8]: robot.say_at({"mode": 1, "audioId": "Achtung - Copy.mp3", "volume": 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.25,
#             'y': -0.0475,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 1.69,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [
#                 {'type': 5, 'data': {'mode': 1, 'audioId': 'Achtung - Copy.mp3', 'volume': 70}}
#             ]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[8]: {'taskId': 'f801a303-3b5f-4f1b-aa03-e83a0b8d8cc2'}

# In [9]: robot.say_at({"mode": 1, "audioId": 1, "volume": 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.2575,
#             'y': -0.105,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.03,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [{'type': 5, 'data': {'mode': 1, 'audioId': '1', 'volume': 70}}]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[9]: {'taskId': 'd7721ba3-6102-4f66-8926-58346597e935'}

# In [10]: robot.say_at({"mode": 1, "volume": 70})
# ---------------------------------------------------------------------------
# ValueError                                Traceback (most recent call last)
# Cell In[10], line 1
# ----> 1 robot.say_at({"mode": 1, "volume": 70})

# File ~\Desktop\dev\Sandbox\gitrepo\autoxing\lib\api_lib.py:2561, in Robot_v2.say_at(self, data, poi_name, **kwargs)
#    2558 params.update(kwargs)
#    2560 task = Task(self, "say", taskType="restaurant", runType="direct")
# -> 2561 task.say(target_name=poi_name, **params)
#    2562 resp = create_task(**task.task_dict)
#    2563 return resp

# File ~\Desktop\dev\Sandbox\gitrepo\autoxing\lib\api_lib.py:2060, in Task.say(self, target_name, mode, audioId, url, volume, interval, num, duration)
#    2058     raise ValueError("mode must be 1 (upper computer) or 2 (chassis)")
#    2059 if audioId is None and url is None:
# -> 2060     raise ValueError("Provide at least one of audioId or url")
#    2062 data = {"mode": int(mode)}
#    2063 if audioId is not None: data["audioId"] = str(audioId)

# ValueError: Provide at least one of audioId or url

# In [11]: robot.say_at({"mode": 1, "url" = r"http://http://192.168.0.97:8000/Achtung - Copy.mp3", "volum
#        ⋮ e": 70})
#   Cell In[11], line 1
#     robot.say_at({"mode": 1, "url" = r"http://http://192.168.0.97:8000/Achtung - Copy.mp3", "volume": 70})
#                                  ^
# SyntaxError: ':' expected after dictionary key


# In [12]: robot.say_at({"mode": 1, "url" :r"http://http://192.168.0.97:8000/Achtung - Copy.mp3", "volume
#        ⋮ ": 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.295,
#             'y': -0.31,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': -0.94,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [
#                 {
#                     'type': 5,
#                     'data': {
#                         'mode': 1,
#                         'url': 'http://http://192.168.0.97:8000/Achtung - Copy.mp3',
#                         'volume': 70
#                     }
#                 }
#             ]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[12]: {'taskId': 'ea047034-94a6-493f-823c-fa8a1ce30935'}

# In [13]: robot.say_at({"mode": 2, "audioId": "Achtung - Copy.mp3", "volume": 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.2975,
#             'y': -0.2725,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.0,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [
#                 {'type': 5, 'data': {'mode': 2, 'audioId': 'Achtung - Copy.mp3', 'volume': 70}}
#             ]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[13]: {'taskId': '7614cc5e-6908-4023-b630-a97c36e61237'}

# In [14]: robot.say_at({"mode": 1, "url" :r"http://192.168.0.97:8000/Achtung - Copy.mp3", "volume": 70})
#        ⋮
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.2925,
#             'y': -0.275,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.01,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [
#                 {
#                     'type': 5,
#                     'data': {
#                         'mode': 1,
#                         'url': 'http://192.168.0.97:8000/Achtung - Copy.mp3',
#                         'volume': 70
#                     }
#                 }
#             ]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[14]: {'taskId': '78196bfa-4c07-4a0d-a0c2-9db2b0f54d75'}

# In [15]: robot.say_at({"mode": 2, "url" :r"http://192.168.0.97:8000/Achtung - Copy.mp3", "volume": 70})
#        ⋮
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.2475,
#             'y': -0.275,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.0,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [
#                 {
#                     'type': 5,
#                     'data': {
#                         'mode': 2,
#                         'url': 'http://192.168.0.97:8000/Achtung - Copy.mp3',
#                         'volume': 70
#                     }
#                 }
#             ]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[15]: {'taskId': 'b105ece0-8284-4cb2-a655-1d56bdfe2148'}

# In [16]: exit
# (venv) C:\Users\Ahmed Galai\Desktop\dev\Sandbox\gitrepo\autoxing> ipy
# Python 3.13.9 (tags/v3.13.9:8183fa5, Oct 14 2025, 14:09:13) [MSC v.1944 64 bit (AMD64)]
# Type 'copyright', 'credits' or 'license' for more information
# IPython 9.4.0 -- An enhanced Interactive Python. Type '?' for help.
# Tip: Use `object?` to see the help on `object`, `object??` to view its source

# In [1]: import os, sys

# In [2]: sys.path.append(os.path.join(os.getcwd(),"lib"))

# In [3]: from api_lib import *

# In [4]: rid = "8881307202099xR"

# In [5]: robot = Robot(rid)

# In [6]: robot.say_at({"mode": 1, "url" :r"http://192.168.0.97:8000/static/Achtung - Copy.mp3", "volume"
#       ⋮ : 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.2275,
#             'y': -0.085,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': -0.03,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [
#                 {
#                     'type': 5,
#                     'data': {
#                         'mode': 1,
#                         'url': 'http://192.168.0.97:8000/static/Achtung - Copy.mp3',
#                         'volume': 70
#                     }
#                 }
#             ]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[6]: {'taskId': '6ca03ce9-8369-4831-82c1-726367ec1c47'}

# In [7]: robot.say_at({"mode": 2, "volume": 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.225,
#             'y': -0.0875,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': -0.01,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [{'type': 5, 'data': {'mode': 2, 'volume': 70}}]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[7]: {'taskId': '06abd58b-75f9-4f43-9f56-4730015afa73'}

# In [8]: robot.say_at({"mode": 2, "volume": 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.225,
#             'y': -0.0625,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.25,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [{'type': 5, 'data': {'mode': 2, 'volume': 70}}]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[8]: {'taskId': 'bc0207a9-356d-470c-98d4-fe5fcd1551c9'}

# In [9]: robot.say_at({"mode": 1, "url" :r"http://192.168.0.97:8000/static/Achtung - Copy.mp3", "volume"
#       ⋮ : 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.22,
#             'y': -0.0425,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.01,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [
#                 {
#                     'type': 5,
#                     'data': {
#                         'mode': 1,
#                         'url': 'http://192.168.0.97:8000/static/Achtung - Copy.mp3',
#                         'volume': 70
#                     }
#                 }
#             ]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[9]: {'taskId': '289e5851-3686-47c8-aba2-ce7f29fe36c6'}

# In [10]: robot.say_at({"mode": 2, "url" :r"http://192.168.0.97:8000/Achtung - Copy.mp3", "volume": 70})
#        ⋮
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.225,
#             'y': -0.04,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.0,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [
#                 {
#                     'type': 5,
#                     'data': {
#                         'mode': 2,
#                         'url': 'http://192.168.0.97:8000/Achtung - Copy.mp3',
#                         'volume': 70
#                     }
#                 }
#             ]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[10]: {'taskId': '60cecfcb-4ee5-41de-aac4-81cc0e62dde8'}

# In [11]: robot.say_at({"mode": 1, "url" :r"http://192.168.0.97:8000/Achtung - Copy.mp3", "volume": 70})
#        ⋮
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.225,
#             'y': -0.04,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.0,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [
#                 {
#                     'type': 5,
#                     'data': {
#                         'mode': 1,
#                         'url': 'http://192.168.0.97:8000/Achtung - Copy.mp3',
#                         'volume': 70
#                     }
#                 }
#             ]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[11]: {'taskId': 'fb6142e6-8b91-476e-816d-c5fc5ef97271'}

# In [12]: robot.say_at({"mode": 2, "audioId": "Achtung - Copy.mp3", "volume": 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.225,
#             'y': -0.04,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.0,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [
#                 {'type': 5, 'data': {'mode': 2, 'audioId': 'Achtung - Copy.mp3', 'volume': 70}}
#             ]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[12]: {'taskId': 'e6aa5c01-544d-42b7-ba00-3cf6c1b0538d'}

# In [13]: robot.say_at({"mode": 1, "audioId": "Achtung - Copy.mp3", "volume": 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.225,
#             'y': -0.04,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.0,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [
#                 {'type': 5, 'data': {'mode': 1, 'audioId': 'Achtung - Copy.mp3', 'volume': 70}}
#             ]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[13]: {'taskId': 'e8a80fa1-53b9-4436-8d56-ac5ae4f65b84'}

# In [14]: robot.say_at({"mode": 1, "audioId": "Achtung - Copy.mp3", "volume": 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.225,
#             'y': -0.04,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.0,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [
#                 {'type': 5, 'data': {'mode': 1, 'audioId': 'Achtung - Copy.mp3', 'volume': 70}}
#             ]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[14]: {'taskId': 'f40324c3-e4d5-4277-a852-22d05929a641'}

# In [15]: robot.say_at({"mode": 1, "volume": 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.225,
#             'y': -0.04,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.0,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [{'type': 5, 'data': {'mode': 1, 'volume': 70}}]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[15]: {'taskId': 'b78a67c4-6265-4fb1-b425-abd5e1a194bb'}

# In [16]: robot.say_at({"mode": 1, "audioId": 1, "volume": 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.225,
#             'y': -0.04,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.0,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [{'type': 5, 'data': {'mode': 1, 'audioId': '1', 'volume': 70}}]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[16]: {'taskId': '6a9ebba7-720f-4f00-acd9-2f25d0540dfb'}

# In [17]: robot.say_at({"mode": 1, "audioId": 2, "volume": 70})
# {
#     'name': 'say',
#     'robotId': '8881307202099xR',
#     'businessId': '668fbe1568f09f1ae8873ca3',
#     'runNum': 1,
#     'taskType': 2,
#     'runType': 22,
#     'routeMode': 1,
#     'runMode': 1,
#     'taskPts': [
#         {
#             'x': -0.225,
#             'y': -0.04,
#             'ext': {'name': '__cur__', 'id': '__robot__'},
#             'yaw': 0.0,
#             'areaId': '68e39ade1782dd9aff690228',
#             'stepActs': [{'type': 5, 'data': {'mode': 1, 'audioId': '2', 'volume': 70}}]
#         }
#     ],
#     'sourceType': 6,
#     'ignorePublicSite': False,
#     'speed': -1.0,
#     'detourRadius': 1.0
# }
# Out[17]: {'taskId': 'c57cc1ab-e9ba-4f7c-9641-5568aa3d1cb4'}

# In [18]: