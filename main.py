import time
import cv2
import numpy as np
from gpiozero import PWMOutputDevice
import pyrealsense2 as rs
from utils import average_tiles
from dataclasses import dataclass

def get_pwm_outputs():
    led_pins = [4, 14, 15, 17, 18, 27, 22, 23, 24]
    return [PWMOutputDevice(x) for x in led_pins]

def get_sections_for_half():
    top = [0,0, 120, 320]
    middle_left = [120, 0, 360, 100]
    middle_center = [120, 100, 360, 220]
    middle_right =  [120, 220, 360, 320]
    bottom = [320, 0, 480, 320]
    return np.array([top, middle_left, middle_center, middle_right, bottom])

def get_sections(config):
    left = get_sections_for_half()
    right = [[top_left_x, top_left_y + int(config.width/2), bottom_left_x, bottom_left_y + int(config.width/2)] for
              top_left_x, top_left_y, bottom_left_x, bottom_left_y in left]
    return [left, right]

@dataclass
class RunConfig:
    width: int = 640
    height: int = 480
    max_range: int = 3
    
def run_stereo():
    # Configure depth and color streams
    run_config = RunConfig(640, 480, 3)
    pipeline = rs.pipeline()
    realsense_config = rs.config()
    realsense_config.enable_stream(rs.stream.depth, run_config.width, run_config.height, rs.format.z16, 30)
    realsense_config.enable_stream(rs.stream.color, run_config.width, run_config.height, rs.format.bgr8, 30)
    
    # Start streaming
    profile = pipeline.start(realsense_config)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    pwm_outputs = get_pwm_outputs()
    section_coords = get_sections(run_config)
    
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
            
            tile_avgs = average_tiles(depth_image, section_coords, run_config.max_range, depth_scale)
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