import threading
import tkinter as tk
from tkinter import messagebox

# Speech deps
try:
    import speech_recognition as sr
except ImportError:
    sr = None


from assign_command import VoiceCommander
import sys
sys.path.append(r"C:\Users\Ahmed Galai\Desktop\dev\Sandbox\gitrepo\autoxing")
from api_lib_v0 import *

APP_TITLE = "Voice to Text (Tkinter)"

class VoiceApp:
    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE)
        root.geometry("520x180")

        self.label = tk.Label(root, text="Click the button and speak…", font=("Segoe UI", 12))
        self.label.pack(pady=12)

        self.button = tk.Button(root, text="🎤  Give vocal command", font=("Segoe UI", 12, "bold"),
                                command=self.on_click)
        self.button.pack(pady=8)

        self.result = tk.Text(root, height=4, wrap="word", font=("Consolas", 11))
        self.result.pack(fill="both", expand=True, padx=10, pady=8)
        robot_instance = Robot_v2("8982412804553br")
        self.commander = VoiceCommander(robot_instance)

        if sr is None:
            self.disable_with_msg("Missing package: SpeechRecognition. Install it and restart.")
    
    def disable_with_msg(self, msg):
        self.button.config(state="disabled")
        self.label.config(text=msg)

    def on_click(self):
        if sr is None:
            messagebox.showerror("Missing dependency", "Install SpeechRecognition and PyAudio.")
            return
        self.button.config(state="disabled")
        self.label.config(text="Listening… (allow mic access)")
        self.result.delete("1.0", "end")

        t = threading.Thread(target=self._listen_and_transcribe, daemon=True)
        t.start()

    def _listen_and_transcribe(self):
        try:
            r = sr.Recognizer()
            # Optional: tune energy threshold / pause
            r.dynamic_energy_threshold = True
            r.pause_threshold = 0.6

            with sr.Microphone() as source:
                # small ambient noise calibration
                self._ui_status("Calibrating mic…")
                r.adjust_for_ambient_noise(source, duration=0.6)
                self._ui_status("Listening… speak now.")
                audio = r.listen(source, timeout=6, phrase_time_limit=20)

            self._ui_status("Recognizing…")
            try:
                text = r.recognize_google(audio, language="de-DE")  # uses Google Web Speech API
                if not text.strip():
                    text = "[No speech recognized]"
                self._ui_result(text)
                parsed = self.commander.parse(text)
                result_msg = self.commander.dispatch(parsed)
                try:
                    msg = self.commander.dispatch(self.commander.parse(result_msg))
                except Exception as e:
                    msg = f"❌ {e}"
                self._ui_status(msg)   # show action/result
                self._ui_status("Done.")
            except sr.UnknownValueError:
                self._ui_result("[Could not understand audio]")
                self._ui_status("Try again.")
            except sr.RequestError as e:
                self._ui_result(f"[API error: {e}]")
                self._ui_status("Network/API issue.")
        except sr.WaitTimeoutError:
            self._ui_result("[Timeout: no speech detected]")
            self._ui_status("Try again.")
        except OSError as e:
            self._ui_result(f"[Mic error: {e}]")
            self._ui_status("Check microphone permissions/device.")
        finally:
            self.root.after(0, lambda: self.button.config(state="normal"))

    # --- UI helpers (thread-safe via .after) ---
    def _ui_status(self, msg):
        self.root.after(0, lambda: self.label.config(text=msg))

    def _ui_result(self, msg):
        def do():
            self.result.delete("1.0", "end")
            self.result.insert("1.0", msg)
        self.root.after(0, do)


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceApp(root)
    root.mainloop()

