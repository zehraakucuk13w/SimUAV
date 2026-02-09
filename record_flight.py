"""
SimUAV - Flight Video Recorder
Records video from hunter drone camera while both drones fly.
Air-to-Air UAV Detection & Tracking System
"""
import airsim
import numpy as np
import cv2
import time
import math

# Configuration
HUNTER_DRONE = "Drone1"
TARGET_DRONE = "Drone2"
OUTPUT_VIDEO = "flight_recording.mp4"
RECORD_DURATION = 60  # seconds
FPS = 30

# Enemy drone settings
ENEMY_SPEED = 3.0
ENEMY_ALTITUDE = -15.0


def main():
    print("=" * 60)
    print("SimUAV - Flight Video Recorder")
    print("=" * 60)
    
    # Connect to AirSim
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected!")
    
    # Setup drones
    print("Setting up drones...")
    client.enableApiControl(True, HUNTER_DRONE)
    client.enableApiControl(True, TARGET_DRONE)
    client.armDisarm(True, HUNTER_DRONE)
    client.armDisarm(True, TARGET_DRONE)
    
    # Waypoints for enemy drone
    waypoints = [
        (30, 0, ENEMY_ALTITUDE),
        (30, 30, ENEMY_ALTITUDE - 3),
        (0, 30, ENEMY_ALTITUDE),
        (-30, 30, ENEMY_ALTITUDE + 3),
        (-30, 0, ENEMY_ALTITUDE),
        (-30, -30, ENEMY_ALTITUDE - 3),
        (0, -30, ENEMY_ALTITUDE),
        (30, -30, ENEMY_ALTITUDE + 3),
    ]
    current_wp = 0
    
    # Takeoff
    print("Taking off...")
    f1 = client.takeoffAsync(vehicle_name=HUNTER_DRONE)
    f2 = client.takeoffAsync(vehicle_name=TARGET_DRONE)
    f1.join()
    f2.join()
    
    # Move to positions
    print("Moving to positions...")
    f1 = client.moveToZAsync(-10, 3, vehicle_name=HUNTER_DRONE)
    f2 = client.moveToZAsync(ENEMY_ALTITUDE, 3, vehicle_name=TARGET_DRONE)
    f1.join()
    f2.join()
    
    # Start enemy movement
    wp = waypoints[current_wp]
    client.moveToPositionAsync(wp[0], wp[1], wp[2], ENEMY_SPEED, vehicle_name=TARGET_DRONE)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    print(f"Recording for {RECORD_DURATION} seconds...")
    print("Press Ctrl+C to stop early")
    
    start_time = time.time()
    frame_count = 0
    
    try:
        while time.time() - start_time < RECORD_DURATION:
            # Update enemy waypoint
            try:
                state = client.getMultirotorState(vehicle_name=TARGET_DRONE)
                pos = state.kinematics_estimated.position
                wp = waypoints[current_wp]
                dist = math.sqrt((pos.x_val-wp[0])**2 + (pos.y_val-wp[1])**2 + (pos.z_val-wp[2])**2)
                
                if dist < 5.0:
                    current_wp = (current_wp + 1) % len(waypoints)
                    wp = waypoints[current_wp]
                    client.moveToPositionAsync(wp[0], wp[1], wp[2], ENEMY_SPEED, vehicle_name=TARGET_DRONE)
            except:
                pass
            
            # Hunter rotates to search
            client.moveByVelocityAsync(0, 0, 0, 0.2,
                yaw_mode=airsim.YawMode(True, 15),
                vehicle_name=HUNTER_DRONE)
            
            # Get frame
            try:
                resp = client.simGetImages([
                    airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
                ], vehicle_name=HUNTER_DRONE)
                
                if resp and resp[0].width > 0:
                    img = np.frombuffer(resp[0].image_data_uint8, dtype=np.uint8)
                    frame = img.reshape(resp[0].height, resp[0].width, 3)
                    
                    # Initialize video writer with first frame
                    if out is None:
                        h, w = frame.shape[:2]
                        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))
                        print(f"Video size: {w}x{h}")
                    
                    out.write(frame)
                    frame_count += 1
                    
                    # Show progress
                    elapsed = time.time() - start_time
                    print(f"\rRecording: {elapsed:.1f}s / {RECORD_DURATION}s | Frames: {frame_count}", end="")
                    
                    # Display preview
                    cv2.imshow("Recording", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except:
                pass
            
            time.sleep(1/FPS)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        print("\n\nLanding drones...")
        f1 = client.landAsync(vehicle_name=HUNTER_DRONE)
        f2 = client.landAsync(vehicle_name=TARGET_DRONE)
        f1.join()
        f2.join()
        
        client.armDisarm(False, HUNTER_DRONE)
        client.armDisarm(False, TARGET_DRONE)
        client.enableApiControl(False, HUNTER_DRONE)
        client.enableApiControl(False, TARGET_DRONE)
        
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Video saved: {OUTPUT_VIDEO}")
        print(f"Total frames: {frame_count}")
        print("Done!")


if __name__ == "__main__":
    main()
