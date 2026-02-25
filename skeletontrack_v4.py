import pyzed.sl as sl
import cv2
import numpy as np
from datetime import datetime, timedelta
import time
import csv
import os
import threading
from collections import defaultdict

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

def save_checkpoint_data(person_event_log, log_folder, checkpoint_number):
    """Save current CSV data with checkpoint number"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file_name = os.path.join(log_folder, f"checkpoint_{checkpoint_number:03d}_{timestamp}.csv")
    
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
        
        print(f"\n--- Checkpoint {checkpoint_number} CSV saved to: {csv_file_name} ---")
        return csv_file_name
    else:
        print(f"\n--- Checkpoint {checkpoint_number}: No data to save in CSV ---")
        return None

def video_recording_loop(zed, obj_runtime_param, objects, image_mat, 
                         defined_spaces_contours, space_names, 
                         person_entry_times, person_last_seen_times, 
                         person_space_lingering_times, person_event_log,
                         log_folder):
    """Main video recording loop with checkpoint management"""
    
    # Get frame dimensions
    zed.retrieve_image(image_mat, sl.VIEW.LEFT)
    test_image = image_mat.get_data()
    test_image = cv2.cvtColor(test_image, cv2.COLOR_RGBA2BGR)
    frame_height, frame_width = test_image.shape[:2]
    
    # Video recording settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_fps = 20.0
    
    # Checkpoint timing
    CHECKPOINT_INTERVAL = 30 * 60  # 30 minutes in seconds
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
    print(f"Checkpoint interval: 30 minutes")
    print("Press 'q' to quit.")
    
    try:
        while True:
            current_time = time.time()
            
            # Check if it's time for a checkpoint
            if current_time - last_checkpoint_time >= CHECKPOINT_INTERVAL:
                # Save current video segment
                video_writer.release()
                print(f"\n=== Checkpoint {checkpoint_number} reached ===")
                print(f"Video segment saved: {current_video_path}")
                
                # Save CSV checkpoint
                save_checkpoint_data(person_event_log, log_folder, checkpoint_number)
                
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
            
            # Process frame
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
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
                            
                            elapsed = (current_datetime - person_entry_times[person_id]).total_seconds()
                            
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            text = f"ID {person_id}: {elapsed:.1f}s"
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

                            for i, contour in enumerate(defined_spaces_contours):
                                space_name = space_names[i]
                                if cv2.pointPolygonTest(contour, person_center, False) >= 0:
                                    current_person_in_spaces[person_id].add(space_name)

                                    if space_name not in person_space_lingering_times[person_id]:
                                        person_space_lingering_times[person_id][space_name] = {
                                            'entry_time': current_datetime,
                                            'last_seen': current_datetime
                                        }
                                        print(f"Person {person_id} entered Space {space_name}.")
                                    else:
                                        person_space_lingering_times[person_id][space_name]['last_seen'] = current_datetime
                                        
                for pid, spaces_data in list(person_space_lingering_times.items()):
                    for space_name, times in list(spaces_data.items()):
                        
                        if pid not in current_zed_ids or space_name not in current_person_in_spaces.get(pid, set()):
                            
                            time_since_last_seen_in_space = (current_datetime - times['last_seen']).total_seconds()

                            if time_since_last_seen_in_space > 2.0:
                                total_time_in_space = (times['last_seen'] - times['entry_time']).total_seconds()
                                if total_time_in_space >= 1.0:
                                    print(f"Person {pid} exited Space {space_name} after {total_time_in_space:.1f} seconds.")
                                    person_event_log.append({
                                        "ID": pid,
                                        "Entry Time": times['entry_time'].strftime("%Y-%m-%d %H:%M:%S"),
                                        "Exit Time": times['last_seen'].strftime("%Y-%m-%d %H:%M:%S"),
                                        "Space ID": space_name,
                                        "Total Lingering Time (s)": f"{total_time_in_space:.1f}"
                                    })
                                del person_space_lingering_times[pid][space_name]
                    
                    if pid not in current_zed_ids:
                        if pid in person_entry_times:
                            del person_entry_times[pid]
                        if pid in person_last_seen_times:
                            del person_last_seen_times[pid]
                        if pid in person_space_lingering_times and not person_space_lingering_times[pid]:
                            del person_space_lingering_times[pid]

                # Add checkpoint timer display
                time_until_next_checkpoint = CHECKPOINT_INTERVAL - (current_time - last_checkpoint_time)
                minutes_left = int(time_until_next_checkpoint // 60)
                seconds_left = int(time_until_next_checkpoint % 60)
                timer_text = f"Next checkpoint: {minutes_left:02d}:{seconds_left:02d} | Segment: {checkpoint_number} | Time: {current_datetime} "
                cv2.putText(image, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("Human Tracking", image)
                video_writer.write(image)
                
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    # Save final checkpoint before quitting
                    video_writer.release()
                    print(f"\nFinal video segment saved: {current_video_path}")
                    save_checkpoint_data(person_event_log, log_folder, checkpoint_number)
                    break

    finally:
        # Ensure video writer is released
        if video_writer.isOpened():
            video_writer.release()
            print(f"Video saved to: {current_video_path}")

def main():
    global defined_spaces_contours, current_drawing_points, drawing_mode, drawing_image_display
    global space_names, space_counter_id, person_space_lingering_times, drawing_active_stroke, last_point

    # Initialize ZED camera
    zed = sl.Camera()

    # Set initialization parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1200  # Native for ZED X
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.sdk_verbose = 0

    # Open camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"Camera failed to open: {err}")
        return

    # *** ADJUST CAMERA SETTINGS FOR COLOR ***
    zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 6)

    # Display camera info
    cam_info = zed.get_camera_information()
    print(f"Camera Model: {cam_info.camera_model}")
    print(f"Firmware: {cam_info.camera_configuration.firmware_version}")
    print(f"Resolution: {cam_info.camera_configuration.resolution.width}x{cam_info.camera_configuration.resolution.height}")

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

    # Create log folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_folder = os.path.join("zed_logs", f"session_{timestamp}")
    os.makedirs(log_folder, exist_ok=True)
    
    print(f"\nLog folder created: {log_folder}")

    # --- Initial frame grab for drawing boundaries ---
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image_mat, sl.VIEW.LEFT)
        drawing_image_display = image_mat.get_data()
        drawing_image_display = cv2.cvtColor(drawing_image_display, cv2.COLOR_RGBA2BGR)
        frame_width = drawing_image_display.shape[1]
        frame_height = drawing_image_display.shape[0]
    else:
        print("Failed to grab initial frame for boundary drawing. Exiting.")
        zed.close()
        return

    print("\n--- Boundary Drawing Mode (Freehand) ---")
    print("  - Click and drag your mouse to draw a freehand area.")
    print("  - Release the mouse button to finalize the current area.")
    print("  - Press 's' to finish drawing all areas and start live tracking.")
    print("  - Press 'c' to clear all defined areas and restart drawing from scratch.")
    print("Areas need at least 3 points to be valid.")

    drawing_mode = True
    cv2.setMouseCallback("Human Tracking", mouse_callback)

    # Use a copy for drawing so the base image is not permanently altered during drawing
    temp_display_image = drawing_image_display.copy()
    while drawing_mode:
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
        # The mouse_callback directly modifies drawing_image_display, so we copy it to temp_display_image
        # to ensure previous strokes are persistent.
        # This part of the loop's drawing is now handled by the mouse_callback directly
        # drawing_image_display is updated by the mouse_callback.
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
            print("Finished drawing. Starting live human tracking with 30-minute checkpoints.")

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
        
    cv2.setMouseCallback("Human Tracking", lambda *args: None) # Disable mouse callback for drawing

    # Start the main recording loop with checkpoint management
    video_recording_loop(zed, obj_runtime_param, objects, image_mat,
                        defined_spaces_contours, space_names,
                        person_entry_times, person_last_seen_times,
                        person_space_lingering_times, person_event_log,
                        log_folder)

    # Cleanup
    print("\nDisabling object detection...")
    zed.disable_object_detection()
    print("Disabling positional tracking...")
    zed.disable_positional_tracking()
    print("Closing camera...")
    zed.close()
    
    cv2.destroyAllWindows()
    
    # Final save of any lingering data
    current_time = datetime.now()
    for pid, spaces_data in list(person_space_lingering_times.items()):
        for space_name, times in list(spaces_data.items()):
            total_time_in_space = (current_time - times['entry_time']).total_seconds()
            if total_time_in_space >= 1.0:
                print(f"Person {pid} (program exit) was in Space {space_name} for {total_time_in_space:.1f} seconds.")
                person_event_log.append({
                    "ID": pid,
                    "Entry Time": times['entry_time'].strftime("%Y-%m-%d %H:%M:%S"),
                    "Exit Time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Space ID": space_name,
                    "Total Lingering Time (s)": f"{total_time_in_space:.1f}"
                })

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
        print(f"\nComplete tracking data saved to: {final_csv_path}")
    else:
        print("\nNo person events to log to CSV.")

if __name__ == "__main__":
    main()
