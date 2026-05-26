"""
Interactive Pose Tracking Correction Tool

This script provides an OpenCV-based interface to review and correct pose tracking
results (stored in .mat files) overlaid on video streams. It allows swapping identities
(Patient/Caregiver) or removing erroneous detections frame-by-frame or over a defined range.

Upon validation, the script generates a corrected .mat file and renders a new .avi video
with the clean data. Actions and review statuses are tracked in Excel files.

Controls:
    'd' : Next frame
    'a' : Previous frame
    'i' : Swap identities (current frame)
    'p' : Swap identities (frame range)
    'x' : Remove patient (current frame)
    'u' : Remove patient (frame range)
    'v' : Validate, export corrected files, and proceed to the next video
    'q' : Quit the application
"""

import os
import sys
import cv2
import numpy as np
import scipy.io as sio
import pandas as pd

# Path configuration
main_path = r"C:\Users\bourgema\OneDrive - Université de Genève\PHD\Part1\Data"
ORIGINAL_VIDEO_DIR = fr"{main_path}\Raw\Squat_video\CP_qualisys\Frontal_View"
RESULTS_DIR = fr"{main_path}\Processed\CP_qualisys\Frontal_View\Results"
EXCEL_LOG_PATH = fr"{main_path}\Processed\CP_qualisys\Frontal_View\Modifications_Log.xlsx"
EXCEL_STATUS_PATH = fr"{main_path}\Processed\CP_qualisys\Frontal_View\Review_Status.xlsx"
MODALITY = "ViTPose_Huge"

# Standard COCO 17-keypoint skeleton connections
COCO_PAIRS = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
              (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

COLOR_PATIENT = (0, 255, 0)  # Green
COLOR_CAREGIVER = (0, 165, 255)  # Orange


def log_action(session_logs, patient_name, action_type, start_f, end_f):
    """Records a correction action in the current session's log list."""
    session_logs.append({
        'Video_Name': patient_name,
        'Action': action_type,
        'Start_Frame': start_f,
        'End_Frame': end_f
    })
    print(f"Log: {action_type} (Frames {start_f}-{end_f})")


def draw_info(frame, kpts_frame, bboxes_frame):
    """Generates the visual overlay of bounding boxes and keypoints."""
    viz = frame.copy()
    colors = [COLOR_PATIENT, COLOR_CAREGIVER]
    labels = ["Patient", "Caregiver"]

    for p_idx in range(2):
        box = bboxes_frame[p_idx]
        # Draw bounding box and label if data exists
        if not np.isnan(box).all():
            cv2.rectangle(viz, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), colors[p_idx], 2)
            cv2.putText(viz, labels[p_idx], (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[p_idx], 2)

        # Draw keypoints
        points = kpts_frame[p_idx]
        for pt in points:
            if not np.isnan(pt).all():
                cv2.circle(viz, (int(pt[0]), int(pt[1])), 4, colors[p_idx], -1)
    return viz


def main():
    # Load review status to skip already processed videos
    if os.path.exists(EXCEL_STATUS_PATH):
        df_status = pd.read_excel(EXCEL_STATUS_PATH)
    else:
        df_status = pd.DataFrame(columns=['Video_Name', 'Status'])

    reviewed_videos = df_status[df_status['Status'] == 'Reviewed']['Video_Name'].tolist()
    patient_folders = [f for f in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, f))]

    for patient_name in patient_folders:
        if patient_name in reviewed_videos:
            continue

        # Define specific file paths for the current patient
        orig_video_path = os.path.join(ORIGINAL_VIDEO_DIR, f"{patient_name}.avi")
        video_path_ia = os.path.join(RESULTS_DIR, patient_name, f"{MODALITY}.avi")
        mat_file = os.path.join(RESULTS_DIR, patient_name, f"{patient_name}_Results_Filtered.mat")

        mat_file_corrected = os.path.join(RESULTS_DIR, patient_name, f"{patient_name}_Results_Corrected.mat")
        out_video_corrected = os.path.join(RESULTS_DIR, patient_name, f"{MODALITY}_Corrected.avi")

        if not all(map(os.path.exists, [orig_video_path, video_path_ia, mat_file])):
            print(f"Skipping {patient_name}: Missing required files.")
            continue

        print(f"\nProcessing: {patient_name}")

        # Load raw matrices into memory for editing
        mat_data = sio.loadmat(mat_file)
        kpts = mat_data[MODALITY][0, 0]['Keypoints']
        bboxes = mat_data[MODALITY][0, 0]['BoundingBoxes']
        total_frames = kpts.shape[0]
        n_kpts = kpts.shape[2]

        cap = cv2.VideoCapture(video_path_ia)
        session_logs = []
        frame_idx = 0

        # Main interactive review loop
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame_idx = max(0, frame_idx - 1)
                continue

            viz_frame = draw_info(frame, kpts[frame_idx], bboxes[frame_idx])
            cv2.putText(viz_frame, f"Frame: {frame_idx}/{total_frames - 1}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("Review UI", viz_frame)

            key = cv2.waitKey(0) & 0xFF

            # --- VALIDATION & RENDERING ---
            if key == ord('v'):
                print("Validating and rendering corrected video...")
                sio.savemat(mat_file_corrected, mat_data)

                # Open the original blank video to overlay clean tracking data
                cap_orig = cv2.VideoCapture(orig_video_path)
                fps = cap_orig.get(cv2.CAP_PROP_FPS)
                w, h = int(cap_orig.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_orig.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out_video = cv2.VideoWriter(out_video_corrected, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

                render_idx = 0
                while cap_orig.isOpened() and render_idx < total_frames:
                    ret_orig, frame_orig = cap_orig.read()
                    if not ret_orig: break

                    colors = [COLOR_PATIENT, COLOR_CAREGIVER]
                    for p_idx in range(2):
                        box = bboxes[render_idx, p_idx]
                        points = kpts[render_idx, p_idx]

                        if not np.isnan(box).any():
                            cv2.rectangle(frame_orig, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                          colors[p_idx], 2)
                            for pt in points:
                                if not np.isnan(pt).any():
                                    cv2.circle(frame_orig, (int(pt[0]), int(pt[1])), 4, colors[p_idx], -1)

                            for pt1, pt2 in COCO_PAIRS:
                                if pt1 < n_kpts and pt2 < n_kpts:
                                    k1, k2 = points[pt1], points[pt2]
                                    if not np.isnan(k1).any() and not np.isnan(k2).any():
                                        cv2.line(frame_orig, (int(k1[0]), int(k1[1])), (int(k2[0]), int(k2[1])),
                                                 colors[p_idx], 2)

                    out_video.write(frame_orig)

                    if render_idx % 10 == 0:
                        sys.stdout.write(f"\rRendering: {render_idx}/{total_frames}")
                        sys.stdout.flush()
                    render_idx += 1

                cap_orig.release()
                out_video.release()
                print(f"\rRendering complete: {patient_name}")

                # Append session logs to the master Excel file
                if session_logs:
                    df_new = pd.DataFrame(session_logs)
                    if os.path.exists(EXCEL_LOG_PATH):
                        df_existing = pd.read_excel(EXCEL_LOG_PATH)
                        df_final = pd.concat([df_existing, df_new], ignore_index=True)
                    else:
                        df_final = df_new
                    df_final.to_excel(EXCEL_LOG_PATH, index=False)

                # Mark video as reviewed in the status tracker
                new_status = pd.DataFrame({'Video_Name': [patient_name], 'Status': ['Reviewed']})
                df_status = pd.concat([df_status, new_status], ignore_index=True)
                df_status.to_excel(EXCEL_STATUS_PATH, index=False)

                break  # Move to the next patient

            # --- EXIT ---
            elif key == ord('q'):
                print("Exiting application.")
                cap.release()
                cv2.destroyAllWindows()
                return

            # --- NAVIGATION ---
            elif key == ord('d'):
                frame_idx = min(frame_idx + 1, total_frames - 1)
            elif key == ord('a'):
                frame_idx = max(frame_idx - 1, 0)

            # --- EDITING COMMANDS ---
            elif key == ord('i'):
                # Swap identity arrays for the current frame
                kpts[frame_idx, [0, 1]] = kpts[frame_idx, [1, 0]]
                bboxes[frame_idx, [0, 1]] = bboxes[frame_idx, [1, 0]]
                log_action(session_logs, patient_name, "Swap", frame_idx, frame_idx)

            elif key == ord('x'):
                # Fill patient data with NaNs for the current frame
                kpts[frame_idx, 0] = np.nan
                bboxes[frame_idx, 0] = np.nan
                log_action(session_logs, patient_name, "Unlabel", frame_idx, frame_idx)

            elif key == ord('p'):
                try:
                    start = max(0, int(input(f"\nStart frame for swap (current={frame_idx}): ")))
                    end = min(total_frames - 1, int(input("End frame: ")))
                    if start <= end:
                        # NumPy vectorization for fast multi-frame swapping
                        kpts[start:end + 1, [0, 1]] = kpts[start:end + 1, [1, 0]]
                        bboxes[start:end + 1, [0, 1]] = bboxes[start:end + 1, [1, 0]]
                        log_action(session_logs, patient_name, "Swap", start, end)
                except ValueError:
                    print("Cancelled.")

            elif key == ord('u'):
                try:
                    start = max(0, int(input(f"\nStart frame to unlabel (current={frame_idx}): ")))
                    end = min(total_frames - 1, int(input("End frame: ")))
                    if start <= end:
                        # NumPy vectorization for fast multi-frame unlabeling
                        kpts[start:end + 1, 0] = np.nan
                        bboxes[start:end + 1, 0] = np.nan
                        log_action(session_logs, patient_name, "Unlabel", start, end)
                except ValueError:
                    print("Cancelled.")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()