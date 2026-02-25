import pyzed.sl as sl
import cv2
import numpy as np
from datetime import datetime, timedelta
import time
import csv
import os
import threading
from collections import defaultdict
import signal
import sys
import subprocess  # For system commands to prevent sleep

# --- Global variables for drawing functionality ---
defined_spaces_contours = []
# Stores all points for the current freehand stroke
current_drawing_points = [] # Renamed from current_drawing_polygon
drawing_mode = False
drawing_image_display = None

# New variables for freehand drawing
drawing_active_stroke = False # True if left mouse button is held down
last_point = (-1, -1) # Stores the last point for continuous line drawing

space_names = {}
space_counter_id = 0

# --- Global variable for tracking person's time in each space ---
person_space_lingering_times = {}
person_session_totals = {}  # Track cumulative time per person per space
running = True  # Global flag for program termination
last_successful_frame_time = time.time()  # Track last successful frame grab
frame_grab_failures = 0  # Count consecutive failures
program_start_time = time.time()  # Track program uptime

# --- Signal handler for clean exit ---
def signal_handler(sig, frame):
    global running
    print("\n\nReceived interrupt signal. Cleaning up...")
    running = False

# --- Mouse callback function for drawing ROIs ---
def mouse_callback(event, x, y, flags, param):
    global drawing_image_display, current_drawing_points, drawing_mode, drawing_active_stroke, last_point

    if not drawing_mode:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_active_stroke = True
        last_point = (x, y)
        current_drawing_points.append(last_point) # Add the starting point
        cv2.circle(drawing_image_display, last_point, 2, (0, 255, 255), -1) # Draw initial dot
        cv2.imshow("Human Tracking", drawing_image_display) # Update display immediately

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_active_stroke:
            current_point = (x, y)
            cv2.line(drawing_image_display, last_point, current_point, (0, 255, 255), 2) # Draw yellow line
            current_drawing_points.append(current_point)
            last_point = current_point
            cv2.imshow("Human Tracking", drawing_image_display)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing_active_stroke = False
        # When the mouse button is released, finalize the current freehand stroke
        # You'll need to decide how to convert these points into a polygon.
        # For simplicity, we'll treat the collected points as a polygon.
        if len(current_drawing_points) > 2: # A polygon needs at least 3 points
            # Simplify the polygon if it has too many points, or use as is
            defined_spaces_contours.append(np.array(current_drawing_points, dtype=np.int32))
            # Assign a name if successfully added
            global space_counter_id
            global space_names
            space_names[len(defined_spaces_contours) - 1] = chr(ord('A') + space_counter_id)
            space_counter_id += 1
            print(f"Area {space_names[len(defined_spaces_contours) - 1]} freehand drawn.")
        else:
            print("Warning: Freehand stroke too short, not creating an area.")
        current_drawing_points = [] # Clear points for the next stroke

def save_checkpoint_data(person_event_log, log_folder, checkpoint_id, person_session_totals):
    """Save current CSV data with checkpoint ID (can be int or string), including cumulative totals"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Handle checkpoint_id properly - if it's an integer, format with leading zeros
    if isinstance(checkpoint_id, int):
        checkpoint_str = f"{checkpoint_id:03d}"
    else:
        checkpoint_str = str(checkpoint_id)
    
    csv_file_name = os.path.join(log_folder, f"checkpoint_{checkpoint_str}_{timestamp}.csv")
    
    if person_event_log:
        # Sort the log
        sorted_log = sorted(person_event_log, key=lambda x: (x['ID'], x.get('Space ID', ''), x['Entry Time']))
        
        # Save to CSV
        with open(csv_file_name, 'w', newline='') as csvfile:
            fieldnames = ["ID", "Entry Time", "Exit Time", "Space ID", "Total Lingering Time (s)"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in sorted_log:
                writer.writerow(row)
        
        # Also save cumulative summary at checkpoint
        summary_file_name = os.path.join(log_folder, f"checkpoint_{checkpoint_str}_{timestamp}_summary.csv")
        with open(summary_file_name, 'w', newline='') as csvfile:
            fieldnames = ["ID", "Space ID", "Cumulative Time (s)", "Number of Visits", "Average Visit Duration (s)"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for pid, spaces in person_session_totals.items():
                for space_name, data in spaces.items():
                    avg_duration = data['total_time'] / data['entries'] if data['entries'] > 0 else 0
                    writer.writerow({
                        "ID": pid,
                        "Space ID": space_name,
                        "Cumulative Time (s)": f"{data['total_time']:.1f}",
                        "Number of Visits": data['entries'],
                        "Average Visit Duration (s)": f"{avg_duration:.1f}"
                    })
        
        print(f"\n--- Checkpoint {checkpoint_str} CSV saved to: {csv_file_name} ---")
        print(f"--- Checkpoint {checkpoint_str} Summary saved to: {summary_file_name} ---")
        return csv_file_name
    else:
        print(f"\n--- Checkpoint {checkpoint_str}: No data to save in CSV ---")
        return None

def prevent_sleep():
    """Prevent system from sleeping (Linux/macOS/Windows compatible)"""
    try:
        if sys.platform.startswith('linux'):
            # Use systemd-inhibit if available (Linux)
            subprocess.Popen(['systemd-inhibit', '--what=sleep:idle', '--who=ZED_Tracking', 
                            '--why=Running human tracking', 'sleep', 'infinity'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("System sleep prevention activated (systemd-inhibit)")
        elif sys.platform == 'darwin':  # macOS
            # Use caffeinate on macOS
            subprocess.Popen(['caffeinate', '-i'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("System sleep prevention activated (caffeinate)")
        elif sys.platform == 'win32':  # Windows
            # Use SetThreadExecutionState via ctypes (would need to implement)
            print("Windows sleep prevention not fully implemented - adjust power settings manually")
    except Exception as e:
        print(f"Could not prevent system sleep: {e}")

def video_recording_loop(zed, obj_runtime_param, objects, image_mat, 
                         defined_spaces_contours, space_names, 
                         person_entry_times, person_last_seen_times, 
                         person_space_lingering_times, person_event_log,
                         log_folder, person_session_totals, running_flag):
    """Main video recording loop with checkpoint management"""
    global last_successful_frame_time, frame_grab_failures, program_start_time
    
    # Get frame dimensions
    zed.retrieve_image(image_mat, sl.VIEW.LEFT)
    test_image = image_mat.get_data()
    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGBA2BGR)
    frame_height, frame_width = test_image.shape[:2]
    
    # Video recording settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = 20.0
    
    # Checkpoint timing - CHANGED TO 20 MINUTES (1200 seconds)
    CHECKPOINT_INTERVAL = 20 * 60  # 20 minutes in seconds
    start_time = time.time()
    last_checkpoint_time = start_time
    checkpoint_number = 1
    
    # Current video writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_video_path = os.path.join(log_folder, f"segment_{checkpoint_number:03d}_{timestamp}.mp4")
    video_writer = cv2.VideoWriter(current_video_path, fourcc, output_fps, (frame_width, frame_height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {current_video_path}")
        return
    
    print(f"\nLive tracking started. Recording to: {current_video_path}")
    print(f"Checkpoint interval: 20 minutes")  # Updated message
    print("Press 'q' to quit, 'r' to reset camera if frozen.")
    print(f"Program uptime: 0 minutes")
    
    frame_count = 0
    last_csv_save_time = time.time()
    last_frame_time = time.time()
    last_heartbeat_time = time.time()
    
    try:
        while running_flag[0]:  # Check the running flag
            current_time = time.time()
            
            # Calculate and display uptime every minute
            if current_time - last_heartbeat_time > 60:
                uptime_minutes = int((current_time - program_start_time) / 60)
                print(f"\n[Heartbeat] Program running for {uptime_minutes} minutes. Checkpoint: {checkpoint_number}, Frames: {frame_count}")
                last_heartbeat_time = current_time
            
            # Check if it's time for a checkpoint
            if current_time - last_checkpoint_time >= CHECKPOINT_INTERVAL:
                # Save current video segment
                video_writer.release()
                print(f"\n=== Checkpoint {checkpoint_number} reached ===")
                print(f"Video segment saved: {current_video_path}")
                
                # Save CSV checkpoint with cumulative data - pass checkpoint_number as int
                save_checkpoint_data(person_event_log, log_folder, checkpoint_number, person_session_totals)
                
                # Start new video segment
                checkpoint_number += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                current_video_path = os.path.join(log_folder, f"segment_{checkpoint_number:03d}_{timestamp}.mp4")
                video_writer = cv2.VideoWriter(current_video_path, fourcc, output_fps, (frame_width, frame_height))
                
                if not video_writer.isOpened():
                    print(f"Error: Could not open new video writer for {current_video_path}")
                    break
                
                print(f"New video segment started: {current_video_path}")
                last_checkpoint_time = current_time
            
            # Check for keyboard input without blocking (use 1ms wait)
            key = cv2.waitKey(1) & 0xFF
            
            # Handle 'r' key to reset camera if frozen
            if key == ord('r'):
                print("\nManual reset triggered. Attempting to recover camera...")
                # Try to re-grab a frame
                if zed.grab() != sl.ERROR_CODE.SUCCESS:
                    print("Camera recovery failed. Please check connection.")
                else:
                    print("Camera recovered successfully.")
                frame_grab_failures = 0
                last_successful_frame_time = current_time
            
            # Handle 'q' key to quit
            if key == ord('q'):
                print("\n'q' pressed. Stopping recording...")
                running_flag[0] = False
                break
            
            # Process frame with timeout handling
            grab_result = zed.grab()
            
            if grab_result == sl.ERROR_CODE.SUCCESS:
                # Reset failure counter on successful grab
                frame_grab_failures = 0
                last_successful_frame_time = current_time
                
                zed.retrieve_image(image_mat, sl.VIEW.LEFT)
                image = image_mat.get_data()
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

                zed.retrieve_objects(objects, obj_runtime_param)

                current_datetime = datetime.now()
                
                current_zed_ids = {obj.id for obj in objects.object_list if obj.label == sl.OBJECT_CLASS.PERSON}
                current_person_in_spaces = {pid: set() for pid in current_zed_ids}

                # --- Overlay defined spaces on the live feed ---
                for i, contour in enumerate(defined_spaces_contours):
                    if contour.size > 0 and contour.shape[0] >= 1 and contour.shape[1] == 2:
                        cv2.polylines(image, [contour.reshape((-1, 1, 2))], True, (255, 0, 0), 2) # Blue lines for defined spaces
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            cv2.putText(image, space_names[i], (cX - 15, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Process each detected person
                for obj in objects.object_list:
                    if obj.label == sl.OBJECT_CLASS.PERSON:
                        person_id = obj.id
                        
                        if person_id not in person_entry_times:
                            person_entry_times[person_id] = current_datetime
                            print(f"New person ID {person_id} detected in scene.")
                        person_last_seen_times[person_id] = current_datetime
                        
                        if obj.bounding_box_2d is not None and len(obj.bounding_box_2d) > 0:
                            x_coords = [p[0] for p in obj.bounding_box_2d]
                            y_coords = [p[1] for p in obj.bounding_box_2d]
                            
                            x1 = int(min(x_coords))
                            y1 = int(min(y_coords))
                            x2 = int(max(x_coords))
                            y2 = int(max(y_coords))
                            
                            # Calculate total cumulative time including multiple entries
                            total_cumulative_time = 0
                            if person_id in person_session_totals:
                                for space_data in person_session_totals[person_id].values():
                                    total_cumulative_time += space_data.get('total_time', 0)
                            
                            # Also add current session time
                            current_session_time = (current_datetime - person_entry_times[person_id]).total_seconds()
                            display_time = total_cumulative_time + current_session_time
                            
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            text = f"ID {person_id}: {display_time:.1f}s"
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.5
                            font_thickness = 3
                            
                            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
                            text_x = x1
                            text_y = y1 - 10
                            padding = 5
                            rect_x1 = text_x - padding
                            rect_y1 = text_y - text_height - padding
                            rect_x2 = text_x + text_width + padding
                            rect_y2 = text_y + baseline + padding
                            
                            rect_x1 = max(0, rect_x1)
                            rect_y1 = max(0, rect_y1)
                            rect_x2 = min(image.shape[1], rect_x2)
                            rect_y2 = min(image.shape[0], rect_y2)

                            cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
                            cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

                            person_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                            if person_id not in person_space_lingering_times:
                                person_space_lingering_times[person_id] = {}

                            # Check each defined space
                            for i, contour in enumerate(defined_spaces_contours):
                                space_name = space_names[i]
                                if cv2.pointPolygonTest(contour, person_center, False) >= 0:
                                    current_person_in_spaces[person_id].add(space_name)

                                    if space_name not in person_space_lingering_times[person_id]:
                                        # New entry to this space
                                        person_space_lingering_times[person_id][space_name] = {
                                            'entry_time': current_datetime,
                                            'last_seen': current_datetime,
                                            'is_active': True  # Track if currently in space
                                        }
                                        print(f"Person {person_id} entered Space {space_name}.")
                                    else:
                                        # Already in this space, update last_seen
                                        person_space_lingering_times[person_id][space_name]['last_seen'] = current_datetime
                                        person_space_lingering_times[person_id][space_name]['is_active'] = True
                
                # Improved exit detection and time accumulation
                for pid, spaces_data in list(person_space_lingering_times.items()):
                    for space_name, times in list(spaces_data.items()):
                        
                        # Check if person has left the space
                        if pid not in current_zed_ids or space_name not in current_person_in_spaces.get(pid, set()):
                            
                            # Mark as inactive if not seen in this space
                            if times.get('is_active', False):
                                time_since_last_seen_in_space = (current_datetime - times['last_seen']).total_seconds()
                                
                                # If not seen for 2 seconds, consider them exited
                                if time_since_last_seen_in_space > 2.0:
                                    times['is_active'] = False
                                    total_time_in_space = (times['last_seen'] - times['entry_time']).total_seconds()
                                    
                                    if total_time_in_space >= 1.0:
                                        # Accumulate to session totals
                                        if pid not in person_session_totals:
                                            person_session_totals[pid] = {}
                                        if space_name not in person_session_totals[pid]:
                                            person_session_totals[pid][space_name] = {'total_time': 0, 'entries': 0}
                                        
                                        person_session_totals[pid][space_name]['total_time'] += total_time_in_space
                                        person_session_totals[pid][space_name]['entries'] += 1
                                        
                                        # Log this entry/exit event
                                        person_event_log.append({
                                            "ID": pid,
                                            "Entry Time": times['entry_time'].strftime("%Y-%m-%d %H:%M:%S"),
                                            "Exit Time": times['last_seen'].strftime("%Y-%m-%d %H:%M:%S"),
                                            "Space ID": space_name,
                                            "Total Lingering Time (s)": f"{total_time_in_space:.1f}"
                                        })
                                        print(f"Person {pid} exited Space {space_name} after {total_time_in_space:.1f} seconds. "
                                              f"Cumulative: {person_session_totals[pid][space_name]['total_time']:.1f}s")
                                        
                                        # Keep the entry but mark as inactive for possible re-entry
                                        times['entry_time'] = current_datetime  # Reset for next entry
                
                # Better cleanup - don't delete data for persons who leave and might return
                for pid in list(person_entry_times.keys()):
                    if pid not in current_zed_ids:
                        # Person not seen for a while
                        time_since_last_seen = (current_datetime - person_last_seen_times.get(pid, current_datetime)).total_seconds()
                        
                        # If person has been gone for more than 10 seconds, consider them completely gone
                        if time_since_last_seen > 10.0:
                            # Finalize any active spaces
                            if pid in person_space_lingering_times:
                                for space_name, times in list(person_space_lingering_times[pid].items()):
                                    if times.get('is_active', False):
                                        total_time = (times['last_seen'] - times['entry_time']).total_seconds()
                                        if total_time >= 1.0:
                                            # Accumulate to session totals
                                            if pid not in person_session_totals:
                                                person_session_totals[pid] = {}
                                            if space_name not in person_session_totals[pid]:
                                                person_session_totals[pid][space_name] = {'total_time': 0, 'entries': 0}
                                            
                                            person_session_totals[pid][space_name]['total_time'] += total_time
                                            person_session_totals[pid][space_name]['entries'] += 1
                                            
                                            person_event_log.append({
                                                "ID": pid,
                                                "Entry Time": times['entry_time'].strftime("%Y-%m-%d %H:%M:%S"),
                                                "Exit Time": times['last_seen'].strftime("%Y-%m-%d %H:%M:%S"),
                                                "Space ID": space_name,
                                                "Total Lingering Time (s)": f"{total_time:.1f}"
                                            })
                            
                            # Clean up but keep session totals
                            if pid in person_entry_times:
                                del person_entry_times[pid]
                            if pid in person_last_seen_times:
                                del person_last_seen_times[pid]
                            if pid in person_space_lingering_times:
                                del person_space_lingering_times[pid]

                # Add checkpoint timer display
                time_until_next_checkpoint = CHECKPOINT_INTERVAL - (current_time - last_checkpoint_time)
                minutes_left = int(time_until_next_checkpoint // 60)
                seconds_left = int(time_until_next_checkpoint % 60)
                timer_text = f"Next checkpoint: {minutes_left:02d}:{seconds_left:02d} | Segment: {checkpoint_number}"
                
                # Add current time display
                current_time_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
                
                # Add uptime display
                uptime_minutes = int((current_time - program_start_time) / 60)
                uptime_text = f"Uptime: {uptime_minutes} min"
                
                # Draw timer, uptime, and current time at the top
                cv2.putText(image, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Time: {current_time_str} | {uptime_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add cumulative totals display for active persons (optional)
                y_offset = 90
                for pid in list(current_zed_ids)[:3]:  # Show up to 3 persons to avoid clutter
                    if pid in person_session_totals:
                        total_for_person = sum(data['total_time'] for data in person_session_totals[pid].values())
                        cv2.putText(image, f"ID {pid} Total: {total_for_person:.1f}s", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                        y_offset += 25

                cv2.imshow("Human Tracking", image)
                video_writer.write(image)
                
                # Save CSV periodically (every 5 minutes) to ensure data isn't lost
                if current_time - last_csv_save_time > 300:  # 5 minutes
                    # Pass a string identifier for auto-saves
                    save_checkpoint_data(person_event_log, log_folder, f"auto_{checkpoint_number}", person_session_totals)
                    last_csv_save_time = current_time
                
                frame_count += 1
                last_frame_time = current_time
                
            else:
                # Handle frame grab failure
                frame_grab_failures += 1
                time_since_last_success = current_time - last_successful_frame_time
                
                # Only print every 10 failures to avoid spam
                if frame_grab_failures % 10 == 1:
                    print(f"Warning: Frame grab failed ({frame_grab_failures} consecutive failures). Error: {grab_result}")
                
                # If no successful frame for 30 seconds, try to reset camera
                if time_since_last_success > 30 and frame_grab_failures > 10:
                    print("Camera appears frozen. Attempting to recover...")
                    # Try to disable and re-enable object detection
                    try:
                        zed.disable_object_detection()
                        time.sleep(2)
                        zed.enable_object_detection(obj_runtime_param)
                        print("Camera recovery attempted.")
                    except Exception as e:
                        print(f"Recovery failed: {e}")
                    
                    last_successful_frame_time = current_time
                    frame_grab_failures = 0
                
                # Add a status message to the display
                blank_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                cv2.putText(blank_image, f"Camera issue - {frame_grab_failures} failures", 
                           (50, frame_height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(blank_image, "Press 'r' to manually recover", 
                           (50, frame_height//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(blank_image, f"Time since last frame: {time_since_last_success:.0f}s", 
                           (50, frame_height//2 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Human Tracking", blank_image)
                
                # Don't write blank frames to video
                
                # Wait a bit before retrying (increasing backoff)
                wait_time = min(frame_grab_failures, 10)  # Max 10 seconds
                time.sleep(wait_time)

    except Exception as e:
        print(f"\nError in video recording loop: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Ensure video writer is released
        if video_writer.isOpened():
            video_writer.release()
            print(f"Final video segment saved: {current_video_path}")
        
        print("Video recording loop ended.")

def main():
    global defined_spaces_contours, current_drawing_points, drawing_mode, drawing_image_display
    global space_names, space_counter_id, person_space_lingering_times, drawing_active_stroke, last_point
    global running, last_successful_frame_time, frame_grab_failures, program_start_time

    # Record program start time
    program_start_time = time.time()

    # Set up signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create a list with running flag for thread-safe modification
    running_flag = [True]
    
    # Prevent system from sleeping
    prevent_sleep()
    print("System sleep prevention activated.")

    # Initialize ZED camera with retry logic
    max_retries = 3
    retry_count = 0
    zed = None
    
    while retry_count < max_retries and running_flag[0]:
        try:
            zed = sl.Camera()

            # Set initialization parameters
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD1200  # Native for ZED X
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
            init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
            init_params.sdk_verbose = 0
            init_params.camera_fps = 30  # Set explicit FPS
            init_params.sdk_verbose = 1  # Increase verbosity for debugging

            # Open camera
            print(f"Attempting to open camera (attempt {retry_count + 1}/{max_retries})...")
            err = zed.open(init_params)
            if err == sl.ERROR_CODE.SUCCESS:
                print("Camera opened successfully.")
                break
            else:
                print(f"Camera failed to open: {err}")
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    print("Max retries reached. Exiting.")
                    return
        except Exception as e:
            print(f"Camera initialization error: {e}")
            retry_count += 1
            if retry_count < max_retries:
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Max retries reached. Exiting.")
                return

    if zed is None or not zed.is_opened():
        print("Failed to open camera after retries.")
        return

    # *** ADJUST CAMERA SETTINGS FOR COLOR ***
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 6)
    # Set camera timeout to prevent hangs
    zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)  # Auto exposure
    zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)  # Auto white balance

    # Display camera info
    cam_info = zed.get_camera_information()
    print(f"Camera Model: {cam_info.camera_model}")
    print(f"Firmware: {cam_info.camera_configuration.firmware_version}")
    print(f"Resolution: {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")
    print(f"FPS: {cam_info.camera_configuration.fps}")

    # Enable positional tracking for object tracking
    tracking_params = sl.PositionalTrackingParameters()
    err = zed.enable_positional_tracking(tracking_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Positional tracking error: {err}")
        zed.close()
        return

    # Enable object detection
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM

    print("Enabling object detection...")
    err = zed.enable_object_detection(obj_param)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Object detection error: {err}")
        zed.disable_positional_tracking()
        zed.close()
        return

    # Create ZED objects for storing detection results
    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 60

    image_mat = sl.Mat()
    cv2.namedWindow("Human Tracking", cv2.WINDOW_NORMAL)

    person_entry_times = {}
    person_last_seen_times = {}
    person_event_log = []
    person_session_totals = {}  # Track cumulative times

    # Create log folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = os.path.join("zed_logs", f"session_{timestamp}")
    os.makedirs(log_folder, exist_ok=True)
    
    print(f"\nLog folder created: {log_folder}")

    # --- Initial frame grab for drawing boundaries with retry ---
    frame_grabbed = False
    for attempt in range(5):
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            drawing_image_display = image_mat.get_data()
            drawing_image_display = cv2.cvtColor(drawing_image_display, cv2.COLOR_RGBA2BGR)
            frame_width = drawing_image_display.shape[1]
            frame_height = drawing_image_display.shape[0]
            frame_grabbed = True
            print("Initial frame grabbed successfully.")
            break
        else:
            print(f"Waiting for initial frame (attempt {attempt+1}/5)...")
            time.sleep(1)
    
    if not frame_grabbed:
        print("Failed to grab initial frame for boundary drawing. Exiting.")
        zed.close()
        return

    print("\n--- Boundary Drawing Mode (Freehand) ---")
    print("  - Click and drag your mouse to draw a freehand area.")
    print("  - Release the mouse button to finalize the current area.")
    print("  - Press 's' to finish drawing all areas and start live tracking.")
    print("  - Press 'c' to clear all defined areas and restart drawing from scratch.")
    print("  - Press 'q' during live tracking to quit.")
    print("Areas need at least 3 points to be valid.")

    drawing_mode = True
    cv2.setMouseCallback("Human Tracking", mouse_callback)

    # Use a copy for drawing so the base image is not permanently altered during drawing
    temp_display_image = drawing_image_display.copy()
    while drawing_mode and running_flag[0]:
        display_img_copy = temp_display_image.copy()

        # Draw already defined spaces (blue lines, white label)
        for i, contour in enumerate(defined_spaces_contours):
            # Ensure the contour is not empty and has valid shape for polylines
            if contour.size > 0 and contour.shape[0] >= 1 and contour.shape[1] == 2:
                cv2.polylines(display_img_copy, [np.array(contour, dtype=np.int32).reshape((-1, 1, 2))], True, (255, 0, 0), 2)
                # Calculate centroid for text
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(display_img_copy, space_names[i], (cX - 15, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Draw the current active freehand stroke on top of the base image copy
        cv2.imshow("Human Tracking", drawing_image_display) # Show the image that the callback is drawing on
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'): # 's' key to start live tracking
            # If there's an active stroke when 's' is pressed, finalize it
            if drawing_active_stroke and len(current_drawing_points) > 2:
                defined_spaces_contours.append(np.array(current_drawing_points, dtype=np.int32))
                space_names[len(defined_spaces_contours) - 1] = chr(ord('A') + space_counter_id)
                print(f"Final area {space_names[len(defined_spaces_contours) - 1]} freehand drawn.")
            elif not defined_spaces_contours and not drawing_active_stroke:
                 print("No areas were defined. Starting tracking without any designated areas.")
            
            drawing_mode = False
            print("Finished drawing. Starting live human tracking with 20-minute checkpoints.")  # Updated message

        elif key == ord('c'): # 'c' key to clear all areas and restart drawing
            defined_spaces_contours = []
            space_names = {}
            space_counter_id = 0
            current_drawing_points = []
            drawing_active_stroke = False
            last_point = (-1, -1)
            # Reset drawing_image_display to a fresh copy of the initial frame
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image_mat, sl.VIEW.LEFT)
                drawing_image_display = image_mat.get_data()
                drawing_image_display = cv2.cvtColor(drawing_image_display, cv2.COLOR_RGBA2BGR)
            print("All defined areas cleared. Restarting drawing.")
        elif key == ord('q'): # Allow quitting during drawing mode
            print("Quit during drawing mode.")
            running_flag[0] = False
            drawing_mode = False
        
    cv2.setMouseCallback("Human Tracking", lambda *args: None) # Disable mouse callback for drawing

    # Start the main recording loop with checkpoint management if not quitting
    if running_flag[0]:
        video_recording_loop(zed, obj_runtime_param, objects, image_mat,
                            defined_spaces_contours, space_names,
                            person_entry_times, person_last_seen_times,
                            person_space_lingering_times, person_event_log,
                            log_folder, person_session_totals, running_flag)

    # Cleanup
    print("\n" + "="*50)
    print("Cleaning up resources...")
    
    print("Disabling object detection...")
    zed.disable_object_detection()
    print("Disabling positional tracking...")
    zed.disable_positional_tracking()
    print("Closing camera...")
    zed.close()
    
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Allow window to close
    
    # Final save of any lingering data - now including session totals
    current_time = datetime.now()
    
    # Process any remaining active sessions
    for pid, spaces_data in list(person_space_lingering_times.items()):
        for space_name, times in list(spaces_data.items()):
            if times.get('is_active', False):
                total_time_in_space = (current_time - times['entry_time']).total_seconds()
                if total_time_in_space >= 1.0:
                    # Add to session totals
                    if pid not in person_session_totals:
                        person_session_totals[pid] = {}
                    if space_name not in person_session_totals[pid]:
                        person_session_totals[pid][space_name] = {'total_time': 0, 'entries': 0}
                    
                    person_session_totals[pid][space_name]['total_time'] += total_time_in_space
                    person_session_totals[pid][space_name]['entries'] += 1
                    
                    person_event_log.append({
                        "ID": pid,
                        "Entry Time": times['entry_time'].strftime("%Y-%m-%d %H:%M:%S"),
                        "Exit Time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "Space ID": space_name,
                        "Total Lingering Time (s)": f"{total_time_in_space:.1f}"
                    })
                    print(f"Person {pid} (program exit) was in Space {space_name} for {total_time_in_space:.1f} seconds.")

    # Add summary of total cumulative times
    print("\n" + "="*50)
    print("CUMULATIVE TIME SUMMARY")
    print("="*50)
    for pid, spaces in person_session_totals.items():
        for space_name, data in spaces.items():
            print(f"Person {pid} spent total of {data['total_time']:.1f} seconds in Space {space_name} across {data['entries']} visits")

    # Final CSV save
    if person_event_log:
        sorted_log = sorted(person_event_log, key=lambda x: (x['ID'], x.get('Space ID', ''), x['Entry Time']))
        
        final_csv_path = os.path.join(log_folder, f"complete_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(final_csv_path, 'w', newline='') as csvfile:
            fieldnames = ["ID", "Entry Time", "Exit Time", "Space ID", "Total Lingering Time (s)"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in sorted_log:
                writer.writerow(row)
        
        # Also create a summary CSV with cumulative times
        summary_csv_path = os.path.join(log_folder, f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        with open(summary_csv_path, 'w', newline='') as csvfile:
            fieldnames = ["ID", "Space ID", "Total Time (s)", "Number of Visits", "Average Visit Duration (s)"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for pid, spaces in person_session_totals.items():
                for space_name, data in spaces.items():
                    avg_duration = data['total_time'] / data['entries'] if data['entries'] > 0 else 0
                    writer.writerow({
                        "ID": pid,
                        "Space ID": space_name,
                        "Total Time (s)": f"{data['total_time']:.1f}",
                        "Number of Visits": data['entries'],
                        "Average Visit Duration (s)": f"{avg_duration:.1f}"
                    })
        
        print(f"\nComplete tracking data saved to: {final_csv_path}")
        print(f"Summary data saved to: {summary_csv_path}")
    else:
        print("\nNo person events to log to CSV.")
    
    # Calculate total runtime
    total_runtime = time.time() - program_start_time
    runtime_minutes = int(total_runtime / 60)
    runtime_seconds = int(total_runtime % 60)
    print(f"\nTotal program runtime: {runtime_minutes} minutes {runtime_seconds} seconds")
    print("\nProgram terminated successfully.")

if __name__ == "__main__":
    main()
