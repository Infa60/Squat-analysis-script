import cv2
import os
import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import gc
import math
import time

# ==============================================================================
# ⚙️ FOLDER CONFIGURATION (TO BE ADAPTED)
# ==============================================================================
folder_frontal = fr"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_video\CP_vicon\Frontal_View"
folder_sagittal = fr"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_video\CP_vicon\Sagittal_View"
folder_top = fr"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Raw\Squat_video\CP_vicon\Top_View"

# Dossier principal contenant les sous-dossiers ViTPose
folder_vitpose_base = fr"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Processed\CP_vicon\Results"

outcome_excel = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data\Processed\Results_Clinical_Table.xlsx"

# Maximum resolution in memory
MAX_CACHE_DIMENSION = 1500

# ==============================================================================
# 📝 DEFINITION OF CLINICAL CRITERIA
# ==============================================================================

# Colonnes exactes issues de votre fichier CSV
EXCEL_COLUMNS = [
    "Patient_ID", "Visit_ID", "Trial", "File_Frontal", "File_Sagittal", "File_Top", "File_ViTPose", "Status",
    "Knee valgus_R", "Knee valgus_L", "Knee outside_R", "Knee outside_L",
    "Heel rise_R", "Heel rise_L", "Foot roll_R", "Foot roll_L",
    "Arms movement", "Stepping forward", "Stepping backward",
    "Foot yaw_R", "Foot yaw_L", "Knee angle", "Trunk inclination",
    "Tibia inclination", "Trunk rotate", "Stance width",
    "Hand-to-ground contact", "Caregiver assistance", "Pose estimation", "Bounding box",
    "Stepping forward_R", "Stepping forward_L", "Stepping backward_R", "Stepping backward_L"
]

# 1. Critères AVEC distinction Droite / Gauche (Right/Left)
CRITERIA_RL = [
    ("Knee valgus", "YN"),
    ("Knee outside", "YN"),
    ("Heel rise", "YN"),
    ("Foot roll", "012"),
    ("Foot yaw", "012"),
    ("Stepping forward", "012"),
    ("Stepping backward", "012"),
]

# 2. Critères GLOBAUX (Uniques, pas de distinction de côté)
# Ajout du type "TEXT" pour les angles et inclinaisons
CRITERIA_UNIQUE = [
    ("Arms movement", "012"),
    ("Knee angle", "TEXT"),
    ("Trunk inclination", "TEXT"),
    ("Tibia inclination", "TEXT"),
    ("Trunk rotate", "012"),
    ("Stance width", "012"),
    ("Hand-to-ground contact", "012"),
    ("Caregiver assistance", "012"),
    ("Pose estimation", "YN"),
    ("Bounding box", "YN"),
]


class SquatAnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clinical Annotator - Squats")

        try:
            self.root.state('zoomed')
        except:
            self.root.attributes('-zoomed', True)

        # --- Data State ---
        self.sessions = []
        self.current_session_idx = 0

        self.cache_f = []  # Frontal
        self.cache_s = []  # Sagittal
        self.cache_t = []  # Top
        self.cache_v = []  # ViTPose
        self.total_frames = 0

        self.current_frame = 0
        self.is_playing = False
        self.fps = 30
        self._updating_slider = False
        self.play_loop_id = None
        self.last_frame_time = 0

        self.form_vars = {}
        self.df_global = self.init_excel()

        self.measure_mode = "OFF"
        self.angles = []
        self.current_points = []
        self.dragged_point = None

        # --- UI Initialization ---
        self.setup_ui()
        self.load_sessions_list()

        # --- Bindings ---
        self.root.bind('<space>', self.toggle_play)
        self.root.bind('<Right>', self.next_frame)
        self.root.bind('<d>', self.next_frame)
        self.root.bind('<Left>', self.prev_frame)
        self.root.bind('<a>', self.prev_frame)

        if self.sessions:
            self.load_session(0)
        else:
            messagebox.showwarning("Empty", "No valid videos found.")

    def init_excel(self):
        if os.path.exists(outcome_excel):
            try:
                df = pd.read_excel(outcome_excel)
                for col in EXCEL_COLUMNS:
                    if col not in df.columns: df[col] = None
                return df
            except Exception as e:
                print(f"Error loading Excel file: {e}. Creating a new one.")
        return pd.DataFrame(columns=EXCEL_COLUMNS)

    def load_sessions_list(self):
        dict_sessions = {}

        # 1. Charger les vues Classiques (Frontal, Sagittal, Top)
        folder_map = [
            (folder_frontal, 'frontal'),
            (folder_sagittal, 'sagittal'),
            (folder_top, 'top')
        ]

        for folder, view_type in folder_map:
            if not os.path.exists(folder):
                continue

            for f in os.listdir(folder):
                if f.lower().endswith(('.mp4', '.avi', '.mov')):
                    base_name = os.path.splitext(f)[0]
                    parts = base_name.split('_')
                    if len(parts) < 2: continue

                    pat = parts[0]
                    vis = parts[1]
                    trial = parts[-1] if parts[-1].isdigit() else "1"
                    key = f"{pat}_{vis}_{trial}"

                    if key not in dict_sessions:
                        dict_sessions[key] = {
                            'patient_id': pat, 'visit_id': vis, 'trial': trial,
                            'path_f': None, 'path_s': None, 'path_t': None, 'path_v': None,
                            'name_f': "", 'name_s': "", 'name_t': "", 'name_v': ""
                        }

                    if view_type == 'frontal':
                        dict_sessions[key]['path_f'] = os.path.join(folder, f)
                        dict_sessions[key]['name_f'] = f
                    elif view_type == 'sagittal':
                        dict_sessions[key]['path_s'] = os.path.join(folder, f)
                        dict_sessions[key]['name_s'] = f
                    elif view_type == 'top':
                        dict_sessions[key]['path_t'] = os.path.join(folder, f)
                        dict_sessions[key]['name_t'] = f

        # 2. Charger et associer la vue ViTPose à partir des sous-dossiers
        if os.path.exists(folder_vitpose_base):
            for folder_name in os.listdir(folder_vitpose_base):
                full_dir_path = os.path.join(folder_vitpose_base, folder_name)

                if os.path.isdir(full_dir_path):
                    parts = folder_name.split('_')
                    if len(parts) >= 4:
                        pat = parts[0]
                        vis = parts[1]
                        trial = parts[-1]
                        key = f"{pat}_{vis}_{trial}"

                        if key in dict_sessions:
                            vitpose_file_path = os.path.join(full_dir_path, "ViTPose_Huge_Corrected.avi")
                            if os.path.exists(vitpose_file_path):
                                dict_sessions[key]['path_v'] = vitpose_file_path
                                dict_sessions[key]['name_v'] = "ViTPose_Huge_Corrected.avi"

        self.sessions = sorted(list(dict_sessions.values()), key=lambda x: (x['patient_id'], x['visit_id'], x['trial']))
        print(f"Found {len(self.sessions)} synchronized video sessions.")

    def setup_ui(self):
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        # Left Panel (Video)
        self.video_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.video_frame, weight=3)

        self.lbl_info = tk.Label(self.video_frame, text="Patient Info", font=("Arial", 14, "bold"), bg="black",
                                 fg="white")
        self.lbl_info.pack(fill=tk.X)

        self.canvas = tk.Canvas(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        # Video controls
        controls_frame = ttk.Frame(self.video_frame)
        controls_frame.pack(fill=tk.X, pady=5, padx=10)

        self.btn_play = ttk.Button(controls_frame, text="▶ Play (Space)", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        ttk.Button(controls_frame, text="⏭ +1 Frame (Right)", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="⏮ -1 Frame (Left)", command=self.prev_frame).pack(side=tk.LEFT, padx=5)

        self.lbl_frame_counter = ttk.Label(controls_frame, text="0 / 0")
        self.lbl_frame_counter.pack(side=tk.LEFT, padx=15)

        self.slider = ttk.Scale(self.video_frame, from_=0, to=100, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.slider.pack(fill=tk.X, padx=10, pady=5)

        # View Toggles
        view_toggle_frame = ttk.Frame(self.video_frame)
        view_toggle_frame.pack(fill=tk.X, pady=5)

        self.show_f_var = tk.BooleanVar(value=True)
        self.show_s_var = tk.BooleanVar(value=True)
        self.show_t_var = tk.BooleanVar(value=True)
        self.show_v_var = tk.BooleanVar(value=True)

        ttk.Checkbutton(view_toggle_frame, text="Frontal", variable=self.show_f_var, command=self.on_view_toggle).pack(
            side=tk.LEFT, padx=5)
        ttk.Checkbutton(view_toggle_frame, text="Sagittal", variable=self.show_s_var, command=self.on_view_toggle).pack(
            side=tk.LEFT, padx=5)
        ttk.Checkbutton(view_toggle_frame, text="Top", variable=self.show_t_var, command=self.on_view_toggle).pack(
            side=tk.LEFT, padx=5)
        ttk.Checkbutton(view_toggle_frame, text="ViTPose", variable=self.show_v_var, command=self.on_view_toggle).pack(
            side=tk.LEFT, padx=5)

        # Right Panel (Form)
        self.form_frame_container = ttk.Frame(self.main_paned)
        self.main_paned.add(self.form_frame_container, weight=1)

        canvas_form = tk.Canvas(self.form_frame_container)
        scrollbar = ttk.Scrollbar(self.form_frame_container, orient="vertical", command=canvas_form.yview)
        self.form_frame = ttk.Frame(canvas_form)

        self.form_frame.bind("<Configure>", lambda e: canvas_form.configure(scrollregion=canvas_form.bbox("all")))
        canvas_form.create_window((0, 0), window=self.form_frame, anchor="nw")
        canvas_form.configure(yscrollcommand=scrollbar.set)

        canvas_form.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        ttk.Label(self.form_frame, text="📝 Clinical Criteria", font=("Arial", 16, "bold")).pack(pady=10)

        # Helper pour créer des boutons radio
        def create_radio(parent, text, var, options):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=2, padx=5)
            ttk.Label(frame, text=text, width=20, anchor="w").pack(side=tk.LEFT)
            for opt in options:
                ttk.Radiobutton(frame, text=opt, value=opt, variable=var).pack(side=tk.LEFT, padx=2)

        # Helper pour créer des champs de texte
        def create_text_input(parent, text, var):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=2, padx=5)
            ttk.Label(frame, text=text, width=20, anchor="w").pack(side=tk.LEFT)
            ttk.Entry(frame, textvariable=var, width=10).pack(side=tk.LEFT, padx=2)
            ttk.Label(frame, text="° (ou valeur)", foreground="gray").pack(side=tk.LEFT)

        # Build Form UI
        # 1. RL Criteria
        ttk.Label(self.form_frame, text="--- Right / Left ---", font=("Arial", 12, "bold")).pack(pady=5)
        for crit, type_input in CRITERIA_RL:
            f = ttk.LabelFrame(self.form_frame, text=crit)
            f.pack(fill=tk.X, padx=5, pady=5)
            opts = ["Y", "N"] if type_input == "YN" else ["0", "1", "2"]

            var_r = tk.StringVar(value="")
            var_l = tk.StringVar(value="")
            self.form_vars[f"{crit}_R"] = var_r
            self.form_vars[f"{crit}_L"] = var_l

            create_radio(f, "Right", var_r, opts)
            create_radio(f, "Left", var_l, opts)

        # 2. Unique Criteria
        ttk.Label(self.form_frame, text="--- Global ---", font=("Arial", 12, "bold")).pack(pady=10)
        for crit, type_input in CRITERIA_UNIQUE:
            var = tk.StringVar(value="")
            self.form_vars[crit] = var

            if type_input == "TEXT":
                create_text_input(self.form_frame, crit, var)
            else:
                opts = ["Y", "N"] if type_input == "YN" else ["0", "1", "2"]
                create_radio(self.form_frame, crit, var, opts)

        # Actions
        ttk.Button(self.form_frame, text="💾 Save & Next", command=self.save_and_next).pack(pady=20, fill=tk.X, padx=10)

    def _remove_focus(self):
        self.root.focus_set()

    def load_session(self, idx):
        if idx < 0 or idx >= len(self.sessions):
            messagebox.showinfo("Finished", "All videos have been processed!")
            self.root.quit()
            return

        self.current_session_idx = idx
        data = self.sessions[idx]

        mask = (self.df_global['Patient_ID'].astype(str) == str(data['patient_id'])) & \
               (self.df_global['Visit_ID'].astype(str) == str(data['visit_id'])) & \
               (self.df_global['Trial'].astype(str) == str(data['trial']))

        if mask.any():
            status = self.df_global[mask].iloc[0].get("Status")
            if pd.notna(status) and str(status).strip().lower() in ["analysé", "analyzed"]:
                print(f"⏭️ Skipped: Patient {data['patient_id']} is already 'Analyzed'.")
                self.root.after(10, lambda: self.load_session(idx + 1))
                return

        self.stop_playback()
        self.cache_f.clear()
        self.cache_s.clear()
        self.cache_t.clear()
        self.cache_v.clear()
        self.total_frames = 0
        gc.collect()

        v_status = "✅" if data['path_v'] else "❌"
        info_text = f"Patient: {data['patient_id']} | Visit: {data['visit_id']} | Trial: {data['trial']} | ViTPose: {v_status} (Loading...)"
        self.lbl_info.config(text=info_text)
        self.root.update()

        for key in self.form_vars:
            self.form_vars[key].set("")

        if mask.any():
            row_data = self.df_global[mask].iloc[0]
            for key in self.form_vars:
                if pd.notna(row_data.get(key)):
                    val = row_data[key]
                    if isinstance(val, (float, np.floating)) and val.is_integer():
                        val = int(val)
                    self.form_vars[key].set(str(val))

        threading.Thread(target=self.process_video_thread, args=(data,), daemon=True).start()

    def process_video_thread(self, data):
        cap_f = cv2.VideoCapture(data['path_f']) if data['path_f'] else None
        cap_s = cv2.VideoCapture(data['path_s']) if data['path_s'] else None
        cap_t = cv2.VideoCapture(data['path_t']) if data['path_t'] else None
        cap_v = cv2.VideoCapture(data['path_v']) if data['path_v'] else None

        fps_list = []
        for cap in [cap_f, cap_s, cap_t, cap_v]:
            if cap and cap.isOpened():
                v_fps = cap.get(cv2.CAP_PROP_FPS)
                if v_fps > 0: fps_list.append(v_fps)
        self.fps = max(fps_list) if fps_list else 30.0

        def process_frame(cap, cache_list):
            ret, frame = cap.read() if cap else (False, None)
            if ret and frame is not None:
                h, w = frame.shape[:2]
                if max(h, w) > MAX_CACHE_DIMENSION:
                    ratio = MAX_CACHE_DIMENSION / max(h, w)
                    frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
                cache_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return ret

        count = 0
        while True:
            rf = process_frame(cap_f, self.cache_f)
            rs = process_frame(cap_s, self.cache_s)
            rt = process_frame(cap_t, self.cache_t)
            rv = process_frame(cap_v, self.cache_v)

            if not rf and not rs and not rt and not rv:
                break

            count += 1
            if count % 50 == 0:
                self.lbl_info.config(text=f"Loading frame {count}...")
                self.root.update_idletasks()

        if cap_f: cap_f.release()
        if cap_s: cap_s.release()
        if cap_t: cap_t.release()
        if cap_v: cap_v.release()

        self.total_frames = max(len(self.cache_f), len(self.cache_s), len(self.cache_t), len(self.cache_v))
        self.current_frame = 0
        self.root.after(0, self.on_video_loaded, data)

    def on_video_loaded(self, data):
        v_status = "✅" if data['path_v'] else "❌"
        self.lbl_info.config(
            text=f"Patient: {data['patient_id']} | Visit: {data['visit_id']} | Trial: {data['trial']} | ViTPose: {v_status}")

        if self.total_frames > 0:
            self._updating_slider = True
            self.slider.config(to=self.total_frames - 1)
            self.slider.set(0)
            self._updating_slider = False
            self.show_frame()

        self.is_playing = True
        self.btn_play.config(text="⏸ Pause (Space)")
        self.last_frame_time = time.time()
        self.start_play_loop()

    def _build_combined_frame(self, idx):
        img_f = self.cache_f[idx] if self.show_f_var.get() and idx < len(self.cache_f) else None
        img_s = self.cache_s[idx] if self.show_s_var.get() and idx < len(self.cache_s) else None
        img_t = self.cache_t[idx] if self.show_t_var.get() and idx < len(self.cache_t) else None
        img_v = self.cache_v[idx] if self.show_v_var.get() and idx < len(self.cache_v) else None

        active_imgs = [img for img in [img_f, img_s, img_t, img_v] if img is not None]

        if not active_imgs:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def resize_to_h(img, target_h):
            if img.shape[0] == target_h: return img
            w = int(img.shape[1] * (target_h / img.shape[0]))
            return cv2.resize(img, (w, target_h), interpolation=cv2.INTER_AREA)

        std_h = active_imgs[0].shape[0]

        if len(active_imgs) == 1:
            return active_imgs[0]
        elif len(active_imgs) == 2:
            return cv2.hconcat([resize_to_h(img, std_h) for img in active_imgs])
        elif len(active_imgs) == 3:
            i1, i2, i3 = [resize_to_h(img, std_h) for img in active_imgs]
            top = cv2.hconcat([i1, i2])
            i3_resized = cv2.resize(i3, (top.shape[1], int(i3.shape[0] * (top.shape[1] / i3.shape[1]))))
            return cv2.vconcat([top, i3_resized])
        elif len(active_imgs) == 4:
            i1, i2, i3, i4 = [resize_to_h(img, std_h) for img in active_imgs]
            top = cv2.hconcat([i1, i2])
            bottom = cv2.hconcat([i3, i4])
            if top.shape[1] != bottom.shape[1]:
                bottom = cv2.resize(bottom, (top.shape[1], bottom.shape[0]))
            return cv2.vconcat([top, bottom])

    def show_frame(self):
        if self.total_frames == 0: return
        img_array = self._build_combined_frame(self.current_frame)
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if canvas_w > 10 and canvas_h > 10:
            img_h, img_w = img_array.shape[:2]
            ratio = min(canvas_w / img_w, canvas_h / img_h)
            new_w, new_h = int(img_w * ratio), int(img_h * ratio)
            img_resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
            img_pil = Image.fromarray(img_resized)
            self.display_offset_x = (canvas_w - new_w) // 2
            self.display_offset_y = (canvas_h - new_h) // 2
        else:
            img_pil = Image.fromarray(img_array)
            self.display_offset_x, self.display_offset_y = 0, 0

        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.canvas.delete("all")
        self.canvas.image = img_tk
        self.canvas.create_image(self.display_offset_x, self.display_offset_y, anchor=tk.NW, image=img_tk)

        self._updating_slider = True
        self.slider.set(self.current_frame)
        self._updating_slider = False
        self.lbl_frame_counter.config(text=f"{self.current_frame} / {self.total_frames - 1}")

    def on_canvas_resize(self, event):
        if not self.is_playing: self.show_frame()

    def on_view_toggle(self):
        self._remove_focus()
        self.show_frame()

    def start_play_loop(self):
        self.stop_playback()
        self.play_loop()

    def stop_playback(self):
        if self.play_loop_id is not None:
            self.root.after_cancel(self.play_loop_id)
            self.play_loop_id = None

    def play_loop(self):
        if self.is_playing and self.total_frames > 0:
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            target_delay = 1.0 / self.fps

            if elapsed >= target_delay:
                frames_to_advance = max(1, int(elapsed / target_delay))
                self.current_frame += frames_to_advance
                if self.current_frame >= self.total_frames:
                    self.current_frame = 0
                    self.last_frame_time = time.time()
                else:
                    self.last_frame_time += frames_to_advance * target_delay
                self.show_frame()
        self.play_loop_id = self.root.after(10, self.play_loop)

    def toggle_play(self, event=None):
        self._remove_focus()
        self.is_playing = not self.is_playing
        if self.is_playing: self.last_frame_time = time.time()
        self.btn_play.config(text="⏸ Pause (Space)" if self.is_playing else "▶ Play (Space)")
        return "break"

    def next_frame(self, event=None):
        self._remove_focus()
        self.is_playing = False
        self.btn_play.config(text="▶ Play (Space)")
        if self.total_frames > 0 and self.current_frame < self.total_frames - 1:
            self.current_frame += 1
            self.show_frame()

    def prev_frame(self, event=None):
        self._remove_focus()
        self.is_playing = False
        self.btn_play.config(text="▶ Play (Space)")
        if self.total_frames > 0 and self.current_frame > 0:
            self.current_frame -= 1
            self.show_frame()

    def on_slider_move(self, val):
        self._remove_focus()
        if getattr(self, '_updating_slider', False): return
        self.current_frame = int(float(val))
        if self.is_playing:
            self.is_playing = False
            self.btn_play.config(text="▶ Play (Space)")
        self.show_frame()

    def save_and_next(self):
        self._remove_focus()
        data = self.sessions[self.current_session_idx]

        new_row_data = {
            "Patient_ID": data['patient_id'], "Visit_ID": data['visit_id'], "Trial": data['trial'],
            "File_Frontal": data['name_f'], "File_Sagittal": data['name_s'],
            "File_Top": data['name_t'], "File_ViTPose": data['name_v'],
            "Status": "Analyzed"
        }

        for key, var in self.form_vars.items():
            val = var.get()
            new_row_data[key] = val if val != "" else None

        mask = (self.df_global['Patient_ID'].astype(str) == str(data['patient_id'])) & \
               (self.df_global['Visit_ID'].astype(str) == str(data['visit_id'])) & \
               (self.df_global['Trial'].astype(str) == str(data['trial']))

        if mask.any():
            idx = self.df_global[mask].index[0]
            for k, v in new_row_data.items():
                self.df_global.loc[idx, k] = v
        else:
            df_new_row = pd.DataFrame([new_row_data])
            self.df_global = pd.concat([self.df_global, df_new_row], ignore_index=True)

        try:
            self.df_global.to_excel(outcome_excel, index=False)
            print(f"✅ Saved successfully: Patient {data['patient_id']} (Trial {data['trial']})")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save Excel file:\n{e}")
            return

        self.load_session(self.current_session_idx + 1)

    def on_canvas_click(self, event):
        pass

    def on_canvas_drag(self, event):
        pass

    def on_canvas_release(self, event):
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = SquatAnnotatorApp(root)
    root.mainloop()