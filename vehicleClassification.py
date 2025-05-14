import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
from norfair import Detection, Tracker
import time
import os

# Check if file exists before starting
VIDEO_PATH = r"./Test Video for Vehicle Counting Model - Indian Road.mp4"  # Use consistent path throughout code

if not os.path.isfile(VIDEO_PATH):
    raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

# Load YOLOv8 model - using larger model for better detection accuracy
model = YOLO("yolov8l.pt")  # Using larger model for better accuracy with smaller objects like bikes

# Parameters that can be tuned - improved defaults
CONFIDENCE_THRESHOLD = 0.4  # Lowered slightly to catch more potential vehicles
MIN_DETECTION_SIZE = 20  # Reduced minimum size to better capture bikes
TRACKER_DISTANCE_THRESHOLD = 30  # Reduced for tighter tracking
TRACK_MEMORY = 20  # Increased memory for smoother tracking
POSITION_HISTORY = 5  # Increased position history for better trajectory analysis

# Vehicle classes to detect and count with improved class mapping
vehicle_classes = ['bicycle', 'motorcycle', 'car', 'bus', 'truck']
# YOLO class IDs mapping (ensure we catch different terminology)
yolo_vehicle_mapping = {
    'bicycle': ['bicycle'],
    'motorcycle': ['motorcycle', 'motorbike'],
    'car': ['car'],
    'bus': ['bus'],
    'truck': ['truck']
}
# Create reverse mapping for quick lookup
yolo_to_vehicle_class = {}
for vehicle_class, yolo_classes in yolo_vehicle_mapping.items():
    for yolo_class in yolo_classes:
        yolo_to_vehicle_class[yolo_class] = vehicle_class

# Open video file first to get dimensions
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {VIDEO_PATH}. Please check the path and file.")

# Get video dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Video opened successfully: {width}x{height} at {fps} FPS")

# Define the ROI and main line based on relative positions
# These are now percentages of the frame height/width
ROI_TOP_PERCENT = 0.25  # Expanded ROI to catch vehicles earlier
ROI_BOTTOM_PERCENT = 0.95  # Expanded ROI to track vehicles longer
LINE_Y_PERCENT = 0.6  # 60% from the top

# Calculate actual pixel positions based on video dimensions
roi_top = int(height * ROI_TOP_PERCENT)
roi_bottom = int(height * ROI_BOTTOM_PERCENT)
line_y = int(height * LINE_Y_PERCENT)

# Tracker with corrected parameters that Norfair supports
tracker = Tracker(
    distance_function="euclidean", 
    distance_threshold=TRACKER_DISTANCE_THRESHOLD,
    hit_counter_max=15,  # Maximum number of frames object can be tracked without detection
    initialization_delay=3  # Wait for this many detections before creating track
)

# Counting variables
vehicle_counts = {cls: {"in": 0, "out": 0} for cls in vehicle_classes}
track_history = {}  # Store position history for each track ID
debug_crossings = []  # Store debugging info about crossings

# Add velocity tracking for better prediction
track_velocities = {}  # Store velocity information for each track ID

def estimate_vehicle_size(cls_name):
    """Return expected size ranges for different vehicle types to help with filtering"""
    if cls_name == 'bicycle' or cls_name == 'motorcycle':
        return (width * 0.01, height * 0.05)  # Smaller minimum size for two-wheelers
    elif cls_name == 'car':
        return (width * 0.02, height * 0.07)
    elif cls_name == 'bus' or cls_name == 'truck':
        return (width * 0.03, height * 0.09)
    else:
        return (width * 0.02, height * 0.07)  # Default

def calculate_velocity(positions, frames=3):
    """Calculate velocity vector based on recent positions"""
    if len(positions) < frames:
        return (0, 0)  # Not enough data
    
    # Use the most recent positions for velocity calculation
    recent = list(positions)[-frames:]
    
    # Calculate average displacement per frame
    dx = (recent[-1][0] - recent[0][0]) / (frames - 1)
    dy = (recent[-1][1] - recent[0][1]) / (frames - 1)
    
    return (dx, dy)

def is_crossing_line(prev_y, current_y, line_position, velocity_y=None):
    """
    Determine if an object is crossing the line and in which direction.
    Now uses velocity for prediction when available.
    
    - If moving UP (decreasing y), it's exiting / going OUT
    - If moving DOWN (increasing y), it's entering / going IN
    """
    # If velocity information is available, use it to help with noise
    if velocity_y is not None:
        # Only count crossing if moving in the consistent direction
        if velocity_y > 1 and prev_y < line_position and current_y >= line_position:
            return "in"  # Moving down across the line (GOING IN)
        elif velocity_y < -1 and prev_y > line_position and current_y <= line_position:
            return "out"  # Moving up across the line (GOING OUT)
    else:
        # Fall back to position-only logic
        if prev_y > line_position and current_y <= line_position:
            return "out"  # Moving up across the line (GOING OUT)
        elif prev_y < line_position and current_y >= line_position:
            return "in"  # Moving down across the line (GOING IN)
    
    return None

def map_to_vehicle_class(yolo_class):
    """Map YOLO class to our vehicle class"""
    yolo_class = yolo_class.lower()
    return yolo_to_vehicle_class.get(yolo_class, None)

# Release and reopen the capture to start from the beginning
cap.release()
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Failed to reopen video file: {VIDEO_PATH}")

# Optional: create video writer for saving the output
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

frame_count = 0
last_cleanup_time = time.time()

print("Starting video processing...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
        
        # Copy original frame for display purposes
        display_frame = frame.copy()
        
        # Double-check dimensions for each frame (in case of variable resolution videos)
        current_height, current_width = frame.shape[:2]
        if current_height != height or current_width != width:
            # Recalculate positions if dimensions changed
            height, width = current_height, current_width
            roi_top = int(height * ROI_TOP_PERCENT)
            roi_bottom = int(height * ROI_BOTTOM_PERCENT)
            line_y = int(height * LINE_Y_PERCENT)
        
        frame_count += 1
        
        # Optional: Apply preprocessing for better detection of small objects like bikes
        # This helps with detecting smaller vehicles like bicycles
        if frame.shape[0] > 1080:  # If video is high resolution
            # No preprocessing needed for high-res videos
            processed_frame = frame
        else:
            # For lower resolution videos, enhance contrast to better detect small objects
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            enhanced_lab = cv2.merge((cl, a, b))
            processed_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        detections = []
        results = model(processed_frame)[0]
        
        # Process YOLO detections
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls_id = result
            cls_id = int(cls_id)
            yolo_cls_name = model.names[cls_id]
            
            # Map the YOLO class to our vehicle class
            mapped_cls = map_to_vehicle_class(yolo_cls_name)
            if mapped_cls is None:
                continue  # Skip if not a vehicle we're interested in
            
            if conf < CONFIDENCE_THRESHOLD:
                continue
            
            # Calculate box dimensions
            w, h = x2 - x1, y2 - y1
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Get minimum size expectations based on vehicle class
            min_w, min_h = estimate_vehicle_size(mapped_cls)
            
            # Special handling for bikes - they can be smaller but should have certain aspect ratios
            is_bike_class = mapped_cls in ['bicycle', 'motorcycle']
            aspect_ratio = w/h if h > 0 else 0
            
            # Bikes typically have aspect ratios between 0.5 and 2.0
            valid_bike = is_bike_class and (0.4 < aspect_ratio < 2.5)
            
            # For non-bikes, use standard size filtering
            valid_size = (w >= min_w and h >= min_h) or valid_bike
            
            # Filter: skip detections that are too small or outside ROI
            if not valid_size or not (roi_top < cy < roi_bottom):
                continue
            
            # Higher confidence threshold for edge areas to reduce false positives
            edge_region = cy < roi_top + (height * 0.05) or cy > roi_bottom - (height * 0.05)
            if edge_region and conf < (CONFIDENCE_THRESHOLD + 0.1):
                continue
                
            detections.append(
                Detection(
                    points=np.array([cx, cy]),
                    data={
                        "cls": mapped_cls, 
                        "box": (int(x1), int(y1), int(x2), int(y2)),
                        "conf": conf,
                        "raw_cls": yolo_cls_name
                    }
                )
            )
        
        # Update tracker with new detections
        tracked_objects = tracker.update(detections=detections)
        
        for track in tracked_objects:
            # Get current position
            cx, cy = track.estimate[0].astype(int)
            track_id = track.id
            cls_name = track.last_detection.data["cls"]
            x1, y1, x2, y2 = track.last_detection.data["box"]
            
            # Initialize track history if this is a new track
            if track_id not in track_history:
                track_history[track_id] = {
                    "positions": deque(maxlen=POSITION_HISTORY),  # Store the last N positions
                    "cls": cls_name,
                    "last_update": frame_count,
                    "counted_directions": set(),  # Keep track of which directions we've already counted
                    "confidence": track.last_detection.data["conf"]
                }
            
            # Update track history
            track_history[track_id]["positions"].append((cx, cy))
            track_history[track_id]["last_update"] = frame_count
            track_history[track_id]["cls"] = cls_name  # Update class in case tracker changed it
            
            # Calculate velocity based on position history
            if len(track_history[track_id]["positions"]) >= 3:
                vx, vy = calculate_velocity(track_history[track_id]["positions"])
                track_velocities[track_id] = (vx, vy)
            
            # We need at least 2 positions to detect crossing
            if len(track_history[track_id]["positions"]) >= 2:
                # Get the current and previous positions
                pos_current = track_history[track_id]["positions"][-1]
                pos_prev = track_history[track_id]["positions"][-2]
                
                # Get velocity if available
                velocity_y = None
                if track_id in track_velocities:
                    _, velocity_y = track_velocities[track_id]
                
                # Check if the object is crossing the line
                crossing_direction = is_crossing_line(pos_prev[1], pos_current[1], line_y, velocity_y)
                
                # Only count if we haven't counted this track in this direction before
                if crossing_direction and crossing_direction not in track_history[track_id]["counted_directions"]:
                    vehicle_counts[cls_name][crossing_direction] += 1
                    track_history[track_id]["counted_directions"].add(crossing_direction)
                    
                    # Add debug info
                    debug_crossings.append({
                        "frame": frame_count,
                        "track_id": track_id,
                        "direction": crossing_direction, 
                        "cls": cls_name,
                        "position": pos_current,
                        "velocity": track_velocities.get(track_id, (0, 0))
                    })
                    
                    print(f"Frame {frame_count}: {cls_name} (ID: {track_id}) counted {crossing_direction}")
            
            # Calculate color with different hues for different vehicle types and saturations for directions
            hue = {
                'bicycle': 30,      # Orange
                'motorcycle': 60,   # Yellow
                'car': 120,         # Green
                'bus': 210,         # Blue
                'truck': 270        # Purple
            }.get(cls_name, 0)
            
            # Adjust saturation based on count status
            if "in" in track_history[track_id]["counted_directions"]:
                saturation = 255  # Full saturation for counted "in"
                value = 255
            elif "out" in track_history[track_id]["counted_directions"]:
                saturation = 255  # Full saturation for counted "out"
                value = 200
            else:
                saturation = 180  # Less saturation for not counted
                value = 255
            
            # Convert HSV to BGR for display
            color_hsv = np.uint8([[[hue, saturation, value]]])
            color = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
            
            # Draw box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            
            # Show direction in label if counted
            direction_label = ""
            if track_history[track_id]["counted_directions"]:
                direction_label = f" ({'/'.join(track_history[track_id]['counted_directions'])})"
                
            # Add velocity indicator if available
            vel_indicator = ""
            if track_id in track_velocities:
                vx, vy = track_velocities[track_id]
                vel_mag = np.sqrt(vx*vx + vy*vy)
                if vel_mag > 5:  # Only show if significant movement
                    vel_indicator = f" v:{vel_mag:.1f}"
            
            # Draw label with class, ID, and direction
            label = f"{cls_name}{direction_label}{vel_indicator}"
            cv2.putText(display_frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw the center point
            cv2.circle(display_frame, (cx, cy), 4, color, -1)
            
            # Draw tracking trail (last N positions)
            positions = list(track_history[track_id]["positions"])
            for i in range(1, len(positions)):
                # Fade the color based on age (older points are dimmer)
                alpha = 0.7 * (i / len(positions))
                trail_color = [int(c * alpha) for c in color]
                cv2.line(display_frame, positions[i-1], positions[i], trail_color, 2)
            
            # Draw velocity vector if available
            if track_id in track_velocities:
                vx, vy = track_velocities[track_id]
                if np.sqrt(vx*vx + vy*vy) > 1.0:  # Only draw if velocity is significant
                    end_x = int(cx + vx * 5)  # Scale for visibility
                    end_y = int(cy + vy * 5)
                    cv2.arrowedLine(display_frame, (cx, cy), (end_x, end_y), color, 2)
        
        # Clean up old tracks every 30 frames
        if frame_count % 30 == 0:
            old_tracks = []
            for track_id, data in track_history.items():
                if frame_count - data["last_update"] > TRACK_MEMORY:  # Remove if not seen for N frames
                    old_tracks.append(track_id)
            
            for track_id in old_tracks:
                del track_history[track_id]
                if track_id in track_velocities:
                    del track_velocities[track_id]
        
        # Draw ROI box and count line
        cv2.rectangle(display_frame, (0, roi_top), (display_frame.shape[1], roi_bottom), (255, 255, 255), 2)
        cv2.line(display_frame, (0, line_y), (display_frame.shape[1], line_y), (0, 0, 255), 2)
        
        # Add arrows to show direction logic - positioned based on relative dimensions
        arrow_x_right = width - int(width * 0.05)  # 5% from right edge
        arrow_x_left = width - int(width * 0.1)    # 10% from right edge
        arrow_length = int(height * 0.05)          # 5% of frame height
        
        # Arrow for IN (moving down)
        cv2.arrowedLine(display_frame, (arrow_x_left, line_y - arrow_length), 
                       (arrow_x_left, line_y + arrow_length), (0, 255, 0), 2, tipLength=0.3)
        cv2.putText(display_frame, "IN", (arrow_x_left - 30, line_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Arrow for OUT (moving up)
        cv2.arrowedLine(display_frame, (arrow_x_right, line_y + arrow_length), 
                       (arrow_x_right, line_y - arrow_length), (0, 0, 255), 2, tipLength=0.3)
        cv2.putText(display_frame, "OUT", (arrow_x_right + 10, line_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display counts - position based on relative dimensions
        y_offset = int(height * 0.05)  # 5% from top
        cv2.putText(display_frame, "Vehicle In/Out Count", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
        
        total_in = 0
        total_out = 0
        for i, (cls, counts) in enumerate(vehicle_counts.items(), start=1):
            text = f"{cls.capitalize()} In: {counts['in']} | Out: {counts['out']}"
            cv2.putText(display_frame, text, (20, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            total_in += counts["in"]
            total_out += counts["out"]
        
        # Display total count
        cv2.putText(display_frame, f"TOTAL In: {total_in} | Out: {total_out}", 
                    (20, y_offset + (len(vehicle_classes) + 1) * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Show frame counter for debugging - positioned based on relative dimensions
        cv2.putText(display_frame, f"Frame: {frame_count}", (width - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show FPS
        end_time = time.time()
        fps_text = f"FPS: {1 / (end_time - last_cleanup_time):.1f}"
        cv2.putText(display_frame, fps_text, (width - 150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        last_cleanup_time = end_time
        
        # Show output
        cv2.imshow("Vehicle Detection & Classification", display_frame)
        
        # Optional: write frame to output video
        # out.write(display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit key pressed. Exiting...")
            break
        elif key == ord('p'):  # Pause/play with 'p' key
            print("Video paused. Press any key to continue...")
            cv2.waitKey(0)
        elif key == ord('r'):  # Reset counts with 'r' key
            vehicle_counts = {cls: {"in": 0, "out": 0} for cls in vehicle_classes}
            track_history.clear()
            track_velocities.clear()
            debug_crossings.clear()
            print("Counts reset!")

except Exception as e:
    print(f"Error during video processing: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Clean up resources
    cap.release()
    # if 'out' in locals():
    #     out.release()  # Uncomment if using video writer
    cv2.destroyAllWindows()
    
    # Print summary of all crossings for debugging
    print("\n--- Crossing Summary ---")
    for crossing in debug_crossings:
        print(f"Frame {crossing['frame']}: {crossing['cls']} (ID: {crossing['track_id']}) {crossing['direction']}")

    print(f"\n--- Final Vehicle Counts ---")
    total_in = 0
    total_out = 0
    for cls, counts in vehicle_counts.items():
        print(f"{cls.capitalize()}: In={counts['in']}, Out={counts['out']}")
        total_in += counts["in"]
        total_out += counts["out"]
    print(f"TOTAL: In={total_in}, Out={total_out}")