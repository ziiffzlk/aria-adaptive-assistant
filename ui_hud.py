import threading
import time
import os
os.environ['TCL_LIBRARY'] = r'C:\Users\PC\AppData\Local\Programs\Python\Python313\tcl\tcl8.6'
os.environ['TK_LIBRARY'] = r'C:\Users\PC\AppData\Local\Programs\Python\Python313\tcl\tk8.6'
import psutil
import customtkinter as ctk
import pystray
from PIL import Image, ImageDraw
import network_state
import database
import patterns
import voice

# Explicit fallback switch for rolling back to legacy UI
USE_LEGACY_UI = False
if USE_LEGACY_UI:
    from ui_hud_legacy import LegacyAriaUI

# Global states
_hud_window = None
_tray_icon = None
_hud_visible = True
_active_user_id = None
_force_offline = False

def is_online():
    if _force_offline:
        return False
    return network_state.is_online()

def get_battery_status():
    battery = psutil.sensors_battery()
    if battery is None:
        return "Unknown", False
    return f"{battery.percent}%", battery.power_plugged

def create_image():
    image = Image.new('RGB', (64, 64), color=(30, 30, 30))
    d = ImageDraw.Draw(image)
    d.ellipse((16, 16, 48, 48), fill=(0, 200, 255))
    return image

class ModernAriaUI:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.overrideredirect(True)
        self.root.attributes("-alpha", 0.92)
        self.root.attributes("-topmost", True)
        self.root.configure(fg_color="#0F0F15")
        
        screen_width = self.root.winfo_screenwidth()
        # Set width to 500 to comfortably fit wraplength=460
        x_pos = screen_width - 520
        self.root.geometry(f"500x400+{x_pos}+40")
        
        self._offset_x = 0
        self._offset_y = 0
        self.root.bind("<Button-1>", self.click_window)
        self.root.bind("<B1-Motion>", self.drag_window)
        
        # 1. Dynamic Audio State Visualizer
        self.header_frame = ctk.CTkFrame(self.root, fg_color="#181820", corner_radius=18, height=40)
        self.header_frame.pack(fill="x", padx=10, pady=10)
        self.header_frame.pack_propagate(False)
        self.header_frame.bind("<Button-1>", self.click_window)
        self.header_frame.bind("<B1-Motion>", self.drag_window)
        
        self.lbl_status = ctk.CTkLabel(self.header_frame, text="● ARIA ONLINE", font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"), text_color="#00C8FF")
        self.lbl_status.pack(pady=6)
        
        # 2. Message Card Architecture (Bento-style Chat Frame)
        self.chat_frame = ctk.CTkFrame(self.root, fg_color="transparent")
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        self.msg_card = ctk.CTkFrame(self.chat_frame, fg_color="#1F1F2B", corner_radius=14)
        self.msg_card.pack(fill="x", side="bottom", pady=5)
        
        self.lbl_msg = ctk.CTkLabel(self.msg_card, text="Awaiting input...", font=ctk.CTkFont(family="Segoe UI", size=14), wraplength=480, anchor="w", justify="left")
        self.lbl_msg.pack(fill="x", padx=15, pady=12)
        
        # 3. Floating Pill Input Bar
        self.input_frame = ctk.CTkFrame(self.root, fg_color="#181820", corner_radius=25, height=50)
        self.input_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.input_frame.pack_propagate(False)
        
        self.entry_cmd = ctk.CTkEntry(self.input_frame, fg_color="transparent", border_width=0, placeholder_text="Command ARIA...", font=ctk.CTkFont(family="Segoe UI", size=14))
        self.entry_cmd.pack(side="left", fill="both", expand=True, padx=(20, 5))
        self.entry_cmd.bind("<Return>", self.send_command)
        
        self.btn_send = ctk.CTkButton(self.input_frame, text="➔", width=36, height=36, corner_radius=20, fg_color="#00C8FF", hover_color="#00A0CC", text_color="#181820", command=self.send_command)
        self.btn_send.pack(side="right", padx=7, pady=7)
        
        self._pulse_state = False
        
        # 4. Non-Blocking Thread Safety & Window Dragging
        self.update_telemetry()
        
    def click_window(self, event):
        self._offset_x = event.x
        self._offset_y = event.y

    def drag_window(self, event):
        x = self.root.winfo_pointerx() - self._offset_x
        y = self.root.winfo_pointery() - self._offset_y
        self.root.geometry(f"+{x}+{y}")
        
    def send_command(self, event=None):
        cmd = self.entry_cmd.get().strip()
        if cmd:
            self.lbl_msg.configure(text=f"Sent: {cmd}")
            self.entry_cmd.delete(0, 'end')
            # The backend hook for command execution would go here
            
    def update_telemetry(self):
        online = is_online()
        mic_info = voice.get_mic_state()
        state = mic_info.get("state", "IDLE")
        
        if not online:
            self.lbl_status.configure(text="⚡ OFFLINE MODE", text_color="#FFB300")
        elif state == "LISTENING" and mic_info.get("level", 0.0) > 0.05:
            # Pulsing effect
            self._pulse_state = not self._pulse_state
            color = "#00FFFF" if self._pulse_state else "#008888"
            self.lbl_status.configure(text="🎙️ LISTENING...", text_color=color)
        elif state == "THINKING":
            self.lbl_status.configure(text="● PROCESSING...", text_color="#AA00FF")
        elif state == "SPEAKING":
            self.lbl_status.configure(text="🔊 SPEAKING...", text_color="#00C8FF")
        else:
            self.lbl_status.configure(text="● ARIA ONLINE", text_color="#00C8FF")
        
        if _hud_visible:
            # 4. Fast polling at 1000ms so typing during background processes remains fluid
            self.root.after(1000, self.update_telemetry)
            
    def hide(self):
        self.root.withdraw()
        
    def show(self):
        self.root.deiconify()
        self.update_telemetry()
        
    def destroy(self):
        self.root.destroy()

def _hud_thread_func():
    global _hud_window
    ctk.set_appearance_mode("dark")
    if USE_LEGACY_UI:
        _hud_window = LegacyAriaUI()
    else:
        _hud_window = ModernAriaUI()
    _hud_window.root.mainloop()

def toggle_hud(icon, item):
    global _hud_visible
    _hud_visible = not _hud_visible
    if _hud_window:
        if _hud_visible:
            _hud_window.show()
        else:
            _hud_window.hide()

def force_offline(icon, item):
    global _force_offline
    _force_offline = not _force_offline

def exit_action(icon, item):
    icon.stop()
    if _hud_window:
        _hud_window.root.quit()

def _tray_thread_func():
    global _tray_icon
    image = create_image()
    menu = pystray.Menu(
        pystray.MenuItem('Toggle HUD Overlay', toggle_hud),
        pystray.MenuItem('Toggle Force Offline (Test)', force_offline),
        pystray.MenuItem('Exit ARIA HUD', exit_action)
    )
    _tray_icon = pystray.Icon("ARIA", image, "ARIA Telemetry", menu)
    _tray_icon.run()

def start_hud(user_id):
    global _active_user_id
    _active_user_id = user_id
    
    hud_thread = threading.Thread(target=_hud_thread_func, daemon=True)
    hud_thread.start()
    
    tray_thread = threading.Thread(target=_tray_thread_func, daemon=True)
    tray_thread.start()
