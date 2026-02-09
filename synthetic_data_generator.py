import airsim
import time
import os
import random
import numpy as np
import math
import cv2

# --- SETTINGS ---
output_dir = "dataset"
num_images = 1000

if not os.path.exists(f"{output_dir}/images"): os.makedirs(f"{output_dir}/images")
if not os.path.exists(f"{output_dir}/labels"): os.makedirs(f"{output_dir}/labels")

print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()

print("Checking system...")
try:
    state = client.getMultirotorState(vehicle_name="Drone1")
    print("Drone1 verified. configuring settings...")
    time.sleep(2) 
except Exception as e:
    print(f"!!! ERROR: Drone not found. Please press Play and wait 15 seconds.")
    exit()

client.enableApiControl(True, "Drone1")
client.enableApiControl(True, "Drone2")
client.armDisarm(True, "Drone1")
client.armDisarm(True, "Drone2")

# --- SEGMENTATION SETUP ---
client.simSetSegmentationObjectID("[\w]*", 0, True)
client.simSetSegmentationObjectID("Drone2", 42, True)

print(f"Starting data generation... ({num_images} frames)")

saved = 0
i = 0

while saved < num_images:
    try:
        t_x = random.uniform(-50, 50)
        t_y = random.uniform(-50, 50)
        t_z = random.uniform(-10, -50)
        t_yaw = random.uniform(0, 2 * math.pi)
        
        pose_target = airsim.Pose(airsim.Vector3r(t_x, t_y, t_z), airsim.to_quaternion(0, 0, t_yaw))
        client.simSetVehiclePose(pose_target, True, vehicle_name="Drone2")

        dist = random.uniform(3.0, 15.0)
        alpha = random.uniform(0, 2 * math.pi)
        beta = random.uniform(-math.pi/4, math.pi/4)

        offset_x = dist * math.cos(beta) * math.cos(alpha)
        offset_y = dist * math.cos(beta) * math.sin(alpha)
        offset_z = dist * math.sin(beta)

        h_x = t_x + offset_x
        h_y = t_y + offset_y
        h_z = t_z + offset_z

        d_x = t_x - h_x
        d_y = t_y - h_y
        d_z = t_z - h_z
        
        final_yaw = math.atan2(d_y, d_x)
        final_pitch = math.atan2(-d_z, math.sqrt(d_x*d_x + d_y*d_y))

        pose_hunter = airsim.Pose(airsim.Vector3r(h_x, h_y, h_z), airsim.to_quaternion(final_pitch, 0, final_yaw))
        client.simSetVehiclePose(pose_hunter, True, vehicle_name="Drone1")

        time.sleep(0.15) 

        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
        ], vehicle_name="Drone1")

        if len(responses) < 2: 
            continue

        img_rgb_1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb_1d.reshape(responses[0].height, responses[0].width, 3)

        img_seg_1d = np.fromstring(responses[1].image_data_uint8, dtype=np.uint8)
        img_seg = img_seg_1d.reshape(responses[1].height, responses[1].width, 3)

        gray = cv2.cvtColor(img_seg, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        label_content = ""
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            img_h, img_w = img_rgb.shape[:2]

            if w > (img_w - 5) and h > (img_h - 5):
                print(f"Warning: Invalid mask (Full Screen), skipping frame {i}.")
            elif w > 5 and h > 5:
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                nw = w / img_w
                nh = h / img_h
                label_content = f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

        if label_content:
            filename = f"img_{saved:06d}"
            cv2.imwrite(f"{output_dir}/images/{filename}.png", img_rgb)

            with open(f"{output_dir}/labels/{filename}.txt", "w") as f:
                f.write(label_content)

            saved += 1

        if i % 10 == 0:
            print(f"Saved(valid): {saved} | Processed: {i}")

        i += 1

    except Exception as e:
        print(f"Loop error: {e}")
        time.sleep(1)

print("Completed.")
client.reset()
