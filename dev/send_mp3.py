# pip install requests bs4
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from mimetypes import guess_type

URL = "http://192.168.0.253:8888/"
MP3_PATH = Path(r"C:\Users\Ahmed Galai\Desktop\client_specific\HöhneBäckerei\Achtung - Copy.mp3")  # change this

session = requests.Session()

# 1) Load page and parse the first form that has a file input
resp = session.get(URL, timeout=10)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, "html.parser")

form = None
file_input = None
for f in soup.find_all("form"):
    fi = f.find("input", {"type": "file"})
    if fi:
        form, file_input = f, fi
        break
if not form or not file_input:
    raise RuntimeError("No form with a file input found.")

action = form.get("action") or URL
if not action.startswith("http"):
    # make relative actions absolute
    from urllib.parse import urljoin
    action = urljoin(URL, action)
method = (form.get("method") or "post").lower()

file_field_name = file_input.get("name") or "file"

# Collect hidden inputs
data = {}
for inp in form.find_all("input"):
    itype = (inp.get("type") or "").lower()
    name = inp.get("name")
    if not name:
        continue
    if itype in ["hidden", "text"]:
        data[name] = inp.get("value", "")

mime = guess_type(MP3_PATH.name)[0] or "audio/mpeg"
with MP3_PATH.open("rb") as f:
    files = {file_field_name: (MP3_PATH.name, f, mime)}
    if method == "post":
        up = session.post(action, data=data, files=files, timeout=30)
    else:
        up = session.get(action, params=data, timeout=30)
    up.raise_for_status()
    print("Upload OK:", up.status_code)
    print(up.text[:500])
