import time
import cv2
import numpy as np
from gpiozero import PWMOutputDevice
import pyrealsense2 as rs
from utils import average_tiles


def get_pwm_outputs():
    led_pins = [4, 14, 15, 17, 18, 27, 22, 23, 24]
    return [PWMOutputDevice(x) for x in led_pins]

def run_stereo():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    n_rows = 3
    n_cols = 3
    max_range = 3
    pwm_outputs = get_pwm_outputs()
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            tile_avgs = average_tiles(depth_image, n_rows, n_cols, max_range, depth_scale)
            for pwm, tile_avg in zip(pwm_outputs, tile_avgs):
                pwm.value = 1 - tile_avg
                
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("q"):
                break

    finally:

        # Stop streaming
        pipeline.stop()
            
        
if __name__=="__main__":
    run_stereo()