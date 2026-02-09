"""
SimUAV - AirSim Real-Time UAV Tracker
Detects enemy UAV using YOLOv8 and pursues it with PID control.
Air-to-Air UAV Detection & Tracking System
"""
import airsim
import numpy as np
import cv2
import time
import math
from ultralytics import YOLO

# Configuration
MODEL_PATH = "best.pt"
HUNTER_DRONE = "Drone1"
TARGET_DRONE = "Drone2"
CONFIDENCE_THRESHOLD = 0.5
PURSUIT_SPEED = 5.0

# PID Controller Gains
KP_YAW = 0.8
KP_PITCH = 0.5
KP_THROTTLE = 0.3

# Enemy drone movement settings
ENEMY_SPEED = 3.0
ENEMY_ALTITUDE = -15.0

# Performance settings
DETECTION_INTERVAL = 3


class UAVTracker:
    def __init__(self):
        print("=" * 60)
        print("SimUAV - Air-to-Air UAV Tracker")
        print("=" * 60)
        
        # Single client for both drones
        print("Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("Connected!")
        
        # Setup hunter drone
        print("Setting up drones...")
        self.client.enableApiControl(True, HUNTER_DRONE)
        self.client.armDisarm(True, HUNTER_DRONE)
        
        # Setup enemy drone
        self.client.enableApiControl(True, TARGET_DRONE)
        self.client.armDisarm(True, TARGET_DRONE)
        
        # Load YOLO model
        print(f"Loading YOLO model: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)
        
        # Waypoints for enemy drone
        self.waypoints = [
            (30, 0, ENEMY_ALTITUDE),
            (30, 30, ENEMY_ALTITUDE - 3),
            (0, 30, ENEMY_ALTITUDE),
            (-30, 30, ENEMY_ALTITUDE + 3),
            (-30, 0, ENEMY_ALTITUDE),
            (-30, -30, ENEMY_ALTITUDE - 3),
            (0, -30, ENEMY_ALTITUDE),
            (30, -30, ENEMY_ALTITUDE + 3),
        ]
        self.current_waypoint = 0
        self.enemy_task = None
        
        # State variables
        self.target_locked = False
        self.cached_detections = []
        self.frame_counter = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.fps_count = 0
        
        print("Ready! Press Q to quit.")
        print("=" * 60)
    
    def setup_drones(self):
        print("Taking off...")
        f1 = self.client.takeoffAsync(vehicle_name=HUNTER_DRONE)
        f2 = self.client.takeoffAsync(vehicle_name=TARGET_DRONE)
        f1.join()
        f2.join()
        
        print("Moving to positions...")
        f1 = self.client.moveToZAsync(-10, 3, vehicle_name=HUNTER_DRONE)
        f2 = self.client.moveToZAsync(ENEMY_ALTITUDE, 3, vehicle_name=TARGET_DRONE)
        f1.join()
        f2.join()
        
        # Start enemy movement (non-blocking)
        wp = self.waypoints[0]
        self.enemy_task = self.client.moveToPositionAsync(
            wp[0], wp[1], wp[2], ENEMY_SPEED, vehicle_name=TARGET_DRONE
        )
        print("Drones ready!")
    
    def update_enemy(self):
        # Check if enemy reached waypoint
        if self.enemy_task is None:
            return
        
        try:
            state = self.client.getMultirotorState(vehicle_name=TARGET_DRONE)
            pos = state.kinematics_estimated.position
            wp = self.waypoints[self.current_waypoint]
            dist = math.sqrt((pos.x_val-wp[0])**2 + (pos.y_val-wp[1])**2 + (pos.z_val-wp[2])**2)
            
            if dist < 5.0:
                self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
                wp = self.waypoints[self.current_waypoint]
                self.enemy_task = self.client.moveToPositionAsync(
                    wp[0], wp[1], wp[2], ENEMY_SPEED, vehicle_name=TARGET_DRONE
                )
        except:
            pass
    
    def get_frame(self):
        try:
            resp = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
            ], vehicle_name=HUNTER_DRONE)
            if resp and resp[0].width > 0:
                img = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8)
                return img.reshape(resp[0].height, resp[0].width, 3)
        except:
            pass
        return None
    
    def detect(self, frame):
        results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(conf)
                })
        return detections
    
    def pursue(self, det, shape):
        if not det:
            return
        x1, y1, x2, y2 = det['bbox']
        h, w = shape[:2]
        cx, cy = (x1+x2)/2, (y1+y2)/2
        err_x = (cx - w/2) / (w/2)
        err_y = (cy - h/2) / (h/2)
        
        area = (x2-x1) * (y2-y1) / (w*h)
        fwd = max(0, 0.1 - area) * 10 * KP_THROTTLE
        
        try:
            state = self.client.getMultirotorState(vehicle_name=HUNTER_DRONE)
            q = state.kinematics_estimated.orientation
            yaw = math.atan2(2*(q.w_val*q.z_val + q.x_val*q.y_val), 1 - 2*(q.y_val**2 + q.z_val**2))
            
            vx = fwd * math.cos(yaw) * PURSUIT_SPEED
            vy = fwd * math.sin(yaw) * PURSUIT_SPEED
            vz = err_y * KP_PITCH * PURSUIT_SPEED * 0.5
            
            self.client.moveByVelocityAsync(
                vx, vy, vz, 0.1,
                yaw_mode=airsim.YawMode(True, -err_x * KP_YAW * 30),
                vehicle_name=HUNTER_DRONE
            )
        except:
            pass
    
    def search(self):
        try:
            self.client.moveByVelocityAsync(
                0, 0, 0, 0.2,
                yaw_mode=airsim.YawMode(True, 20),
                vehicle_name=HUNTER_DRONE
            )
        except:
            pass
    
    def draw_hud(self, frame):
        h, w = frame.shape[:2]
        cv2.line(frame, (w//2-20, h//2), (w//2+20, h//2), (0,255,0), 1)
        cv2.line(frame, (w//2, h//2-20), (w//2, h//2+20), (0,255,0), 1)
        
        for det in self.cached_detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0,255,0) if self.target_locked else (0,165,255)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"TARGET: {det['confidence']:.2f}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        status = "LOCKED" if self.target_locked else "SEARCHING"
        cv2.rectangle(frame, (10,10), (180,60), (0,0,0), -1)
        cv2.putText(frame, status, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        return frame
    
    def run(self):
        print("Starting...")
        self.setup_drones()
        
        try:
            while True:
                self.update_enemy()
                
                frame = self.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                self.frame_counter += 1
                
                if self.frame_counter % DETECTION_INTERVAL == 0:
                    self.cached_detections = self.detect(frame)
                
                if self.cached_detections:
                    self.target_locked = True
                    self.pursue(self.cached_detections[0], frame.shape)
                else:
                    self.target_locked = False
                    self.search()
                
                # FPS
                self.fps_count += 1
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.fps = self.fps_count / (now - self.last_fps_time)
                    self.fps_count = 0
                    self.last_fps_time = now
                
                cv2.imshow("simUAV Tracker", self.draw_hud(frame.copy()))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def cleanup(self):
        print("Landing...")
        try:
            f1 = self.client.landAsync(vehicle_name=HUNTER_DRONE)
            f2 = self.client.landAsync(vehicle_name=TARGET_DRONE)
            f1.join()
            f2.join()
            self.client.armDisarm(False, HUNTER_DRONE)
            self.client.armDisarm(False, TARGET_DRONE)
            self.client.enableApiControl(False, HUNTER_DRONE)
            self.client.enableApiControl(False, TARGET_DRONE)
        except:
            pass
        cv2.destroyAllWindows()
        print("Done!")


if __name__ == "__main__":
    tracker = UAVTracker()
    tracker.run()
