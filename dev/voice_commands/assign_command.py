import re
import difflib
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ParseResult:
    ok: bool
    action: str = ""
    args: tuple = ()
    message: str = ""

class VoiceCommander:
    """
    Turn utterances into Robot_v2 actions.
    Supports:
      - "charge" / "go charge" / "go to charger"
      - "return" / "go back"
      - "go to <poi>"
      - "wait at <poi> for <N> seconds" / "wait <N> seconds at <poi>"
      - "pickup <shelf>" (lift up)
      - "drop at <target>" (dock or area) | "drop in area <area name>"
      - "shelf <A> to <B>" or "pickup <A> and drop <B>"
      - "goto x=<num> y=<num> [yaw=<num>]" (explicit pose)
    """

    def __init__(self, robot_v2):
        self.robot = robot_v2
        # cache names for fuzzy match
        self._refresh_names()

    def _refresh_names(self):
        pois = self.robot.get_pois()
        self.poi_names = sorted([str(n) for n in pois["name"].dropna().unique()])
        # areas: prefer desc, fall back to name
        ctx = self.robot._refresh_context(force=True)
        areas = ctx["areas_rich"]
        if areas is not None and not areas.empty:
            display = (areas["desc"].fillna("").str.strip())
            fallback = (areas["name"].fillna("").str.strip())
            self.area_names = sorted(
                list({(d if d else f) for d, f in zip(display.tolist(), fallback.tolist()) if (d or f)})
            )
        else:
            self.area_names = []

    # ---------- fuzzy helpers ----------
    @staticmethod
    def _best_match(name: str, candidates: list[str], cutoff: float = 0.6) -> Optional[str]:
        if not name or not candidates:
            return None
        m = difflib.get_close_matches(name, candidates, n=1, cutoff=cutoff)
        return m[0] if m else None

    def _match_poi(self, raw: str) -> Optional[str]:
        return self._best_match(raw, self.poi_names)

    def _match_area(self, raw: str) -> Optional[str]:
        return self._best_match(raw, self.area_names)

    # ---------- parser ----------
    # def parse(self, utt: str) -> ParseResult:
    #     if not utt:
    #         return ParseResult(False, message="Empty command")
    #     s = utt.strip().lower()

    #     # trivial commands
    #     if re.fullmatch(r"(go )?(to )?charge(r)?|charge", s):
    #         return ParseResult(True, "go_charge")
    #     if re.fullmatch(r"(go )?(to )?(back|return)", s) or s in ("back", "return"):
    #         return ParseResult(True, "go_back")

    #     # explicit pose: "goto x=1.2 y=3.4 [yaw=90]"
    #     m = re.search(r"(?:(go( to)?|goto)\s+)?x\s*=\s*([-\d\.]+)\s*[ ,;]\s*y\s*=\s*([-\d\.]+)(?:\s*[ ,;]\s*yaw\s*=\s*([-\d\.]+))?", s)
    #     if m:
    #         x = float(m.group(3)); y = float(m.group(4))
    #         yaw = float(m.group(5)) if m.group(5) else 0.0
    #         return ParseResult(True, "go_to_pose", args=((x, y, yaw),))

    #     # go to <poi>
    #     m = re.search(r"(go( to)?|goto)\s+(.*)", s)
    #     if m:
    #         raw_name = m.group(3).strip()
    #         poi = self._match_poi(raw_name)
    #         if poi:
    #             return ParseResult(True, "go_to_poi", args=(poi,))
    #         # If they said an area, default to centroid goto
    #         area = self._match_area(raw_name)
    #         if area:
    #             # find centroid and call go_to_pose
    #             ctx = self.robot._refresh_context()
    #             ar = ctx["areas_rich"]
    #             row = ar[(ar["desc"].fillna("").str.strip()==area) | (ar["name"].fillna("").str.strip()==area)].iloc[0]
    #             pose = (float(row["centroid_x"]), float(row["centroid_y"]), 0.0)
    #             return ParseResult(True, "go_to_pose", args=(pose,))
    #         return ParseResult(False, message=f"Unknown destination: {raw_name}")

    #     # wait at <poi> for <N> seconds
    #     m = re.search(r"wait(?:\s+at)?\s+(?P<poi>.+?)\s+(?:for\s+)?(?P<n>\d+)\s*(?:sec|secs|second|seconds|s)\b", s)
    #     if m:
    #         poi = self._match_poi(m.group("poi"))
    #         if not poi:
    #             return ParseResult(False, message=f"Unknown place to wait: {m.group('poi')}")
    #         secs = int(m.group("n"))
    #         return ParseResult(True, "wait_at", args=(poi, secs))

    #     # pickup <shelf>
    #     m = re.search(r"(pick ?up|pickup)\s+(.*)", s)
    #     if m:
    #         shelf = self._match_poi(m.group(2))
    #         if not shelf:
    #             return ParseResult(False, message=f"Unknown shelf: {m.group(2)}")
    #         return ParseResult(True, "pickup_at", args=(shelf,))

    #     # drop (dock)  | drop in area <name> | drop at <name>
    #     # area phrasing
    #     m = re.search(r"(drop ?(down)?|place|deliver)\s+(?:in|to|at)\s+(?:area\s+)?(.+)", s)
    #     if m:
    #         raw = m.group(3).strip()
    #         # try area first
    #         area = self._match_area(raw)
    #         if area:
    #             return ParseResult(True, "dropdown_at", args=(area, True))
    #         # then poi (dock)
    #         poi = self._match_poi(raw)
    #         if poi:
    #             return ParseResult(True, "dropdown_at", args=(poi, False))
    #         return ParseResult(False, message=f"Unknown drop target: {raw}")

    #     # shelf A to B
    #     m = re.search(r"(shelf|pickup)\s+(.+?)\s+(?:to|and drop|drop(?:down)?)\s+(.+)", s)
    #     if m:
    #         a_raw, b_raw = m.group(2).strip(), m.group(3).strip()
    #         a = self._match_poi(a_raw)
    #         # B can be area or dock
    #         b_area = self._match_area(b_raw)
    #         b_poi  = self._match_poi(b_raw)
    #         if not a:
    #             return ParseResult(False, message=f"Unknown pickup shelf: {a_raw}")
    #         if b_area:
    #             return ParseResult(True, "shelf_to_shelf", args=(a, b_area, True))
    #         if b_poi:
    #             return ParseResult(True, "shelf_to_shelf", args=(a, b_poi, False))
    #         return ParseResult(False, message=f"Unknown drop target: {b_raw}")

    #     # single words
    #     if s.startswith("charge"):
    #         return ParseResult(True, "go_charge")
    #     if s in ("back", "go back", "return"):
    #         return ParseResult(True, "go_back")

    #     return ParseResult(False, message="Could not understand command")

    def parse(self, utt: str) -> ParseResult:
        import re
        if not utt:
            return ParseResult(False, message="Leerer Befehl")
        s = utt.strip().lower()

        # normalize decimal commas -> dots (für Koordinaten)
        s = re.sub(r'(\d),(?=\d)', r'\1.', s)

        # --- simple commands ---
        # charge
        if re.fullmatch(r"(geh(e)? )?(zur|zur\s+)?ladestation|laden|lade(n)?|zum laden gehen|go to charger|charge", s):
            return ParseResult(True, "go_charge")

        # back / return
        if re.fullmatch(r"(geh(e)? )?zurück|zurueck|retour|return|go back", s):
            return ParseResult(True, "go_back")

        # --- explicit pose (both EN & DE) ---
        # "gehe zu x=1,2 y=3,4 yaw=90" or "goto x=1.2 y=3.4"
        m = re.search(r"(gehe|geh|goto|go)(\s+zu\s+)?\s*x\s*=\s*([-\d\.]+)\s*[ ,;]\s*y\s*=\s*([-\d\.]+)(?:\s*[ ,;]\s*(yaw|richtung|kurs)\s*=\s*([-\d\.]+))?", s)
        if m:
            x = float(m.group(3)); y = float(m.group(4))
            yaw = float(m.group(6)) if m.group(6) else 0.0
            return ParseResult(True, "go_to_pose", args=((x, y, yaw),))

        # --- go to POI/Area by name ---
        # "gehe zu <ort>", "fahr zu <ort>", "goto <place>"
        m = re.search(r"(gehe|geh|fahr|fahre|goto|go)\s+(zu|zum|zur|nach)?\s*(.+)", s)
        if m:
            raw_name = m.group(3).strip()
            poi = self._match_poi(raw_name)
            if poi:
                return ParseResult(True, "go_to_poi", args=(poi,))
            area = self._match_area(raw_name)
            if area:
                ctx = self.robot._refresh_context()
                ar = ctx["areas_rich"]
                row = ar[(ar["desc"].fillna("").str.strip()==area) | (ar["name"].fillna("").str.strip()==area)].iloc[0]
                pose = (float(row["centroid_x"]), float(row["centroid_y"]), 0.0)
                return ParseResult(True, "go_to_pose", args=(pose,))
            return ParseResult(False, message=f"Unbekanntes Ziel: {raw_name}")

        # --- wait ---
        # "warte bei <ort> <N> sekunden", "warte an <ort> für <N> sekunden"
        m = re.search(r"warte\s+(bei|an)?\s*(?P<poi>.+?)\s+(für|fuer)?\s*(?P<n>\d+)\s*(sek|sekunden|s)\b", s)
        if m:
            poi = self._match_poi(m.group("poi"))
            if not poi:
                return ParseResult(False, message=f"Unbekannter Ort zum Warten: {m.group('poi')}")
            secs = int(m.group("n"))
            return ParseResult(True, "wait_at", args=(poi, secs))

        # --- pickup (shelf) ---
        # "hebe bei <regal>", "pickup <name>"
        m = re.search(r"(hebe|heb)\s+(bei|an)?\s*(.+)|pickup\s+(.+)", s)
        if m:
            raw = (m.group(3) or m.group(4) or "").strip()
            shelf = self._match_poi(raw)
            if not shelf:
                return ParseResult(False, message=f"Unbekanntes Regal: {raw}")
            return ParseResult(True, "pickup_at", args=(shelf,))

        # --- drop ---
        # "abladen (in|zu|an) (bereich )?<ziel>", "drop (in|at|to) <ziel>"
        m = re.search(r"(abladen|abladen|ablegen|absetzen|drop|deliver)\s+(in|zu|an|auf|to|at)\s+(bereich\s+)?(.+)", s)
        if m:
            raw = m.group(4).strip()
            area = self._match_area(raw)
            if area:
                return ParseResult(True, "dropdown_at", args=(area, True))
            poi = self._match_poi(raw)
            if poi:
                return ParseResult(True, "dropdown_at", args=(poi, False))
            return ParseResult(False, message=f"Unbekanntes Ziel zum Abladen: {raw}")

        # --- shelf A to B ---
        # "regal <a> zu <b>", "pickup <a> und abladen <b>", "regal <a> nach bereich <b>"
        m = re.search(r"(regal|pickup)\s+(.+?)\s+(?:zu|nach|und\s+ab(?:laden|setzen)|and\s+drop|to)\s+(.+)", s)
        if m:
            a_raw, b_raw = m.group(2).strip(), m.group(3).strip()
            a = self._match_poi(a_raw)
            b_area = self._match_area(b_raw)
            b_poi  = self._match_poi(b_raw)
            if not a:
                return ParseResult(False, message=f"Unbekanntes Abholregal: {a_raw}")
            if b_area:
                return ParseResult(True, "shelf_to_shelf", args=(a, b_area, True))
            if b_poi:
                return ParseResult(True, "shelf_to_shelf", args=(a, b_poi, False))
            return ParseResult(False, message=f"Unbekanntes Ziel: {b_raw}")

        # single words fallbacks
        if s.startswith("laden") or s.startswith("lade"):
            return ParseResult(True, "go_charge")
        if s in ("zurück", "zurueck", "retour"):
            return ParseResult(True, "go_back")

        return ParseResult(False, message="Befehl wurde nicht verstanden")


    # ---------- dispatcher ----------
    def dispatch(self, parsed: ParseResult) -> str:
        if not parsed.ok:
            return f"❌ {parsed.message}"

        try:
            if parsed.action == "go_charge":
                self.robot.go_charge()
                return "⚡ Going to charge."
            if parsed.action == "go_back":
                self.robot.go_back()
                return "↩️ Returning."
            if parsed.action == "go_to_pose":
                (pose,) = parsed.args
                self.robot.go_to_pose(pose)
                return f"🧭 Going to pose x={pose[0]:.3f}, y={pose[1]:.3f}."
            if parsed.action == "go_to_poi":
                (poi,) = parsed.args
                self.robot.go_to_poi(poi)
                return f"🧭 Going to {poi}."
            if parsed.action == "wait_at":
                poi, secs = parsed.args
                self.robot.wait_at(poi, secs)
                return f"⏱️ Waiting {secs}s at {poi}."
            if parsed.action == "pickup_at":
                (poi,) = parsed.args
                self.robot.pickup_at(poi, area_delivery=False)
                return f"📦 Pickup at {poi} (lift up)."
            if parsed.action == "dropdown_at":
                poi_or_area, use_area = parsed.args
                self.robot.dropdown_at(poi_or_area, area_delivery=use_area)
                return f"📥 Drop at {'area ' if use_area else ''}{poi_or_area}."
            if parsed.action == "shelf_to_shelf":
                a, b, use_area = parsed.args
                self.robot.shelf_to_shelf(a, b, area_delivery=use_area)
                return f"🚚 Shelf delivery: {a} → {b} ({'area' if use_area else 'dock'})."

            return "❓ Unhandled action."
        except Exception as e:
            return f"❌ Execution failed: {e}"


# ---------- Hook into your Tkinter app ----------
# In your recognition success handler:
#   commander = VoiceCommander(robot_v2_instance)
#   parsed = commander.parse(recognized_text)
#   result_msg = commander.dispatch(parsed)
#   show result_msg in the UI

