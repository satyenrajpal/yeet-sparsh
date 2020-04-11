import time
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2
import numpy as np

def average(frame):
    return np.average(frame)

# Average values in row major format
def average_rgb_tiles(frame, n_rows, n_cols):
    height = int(frame.shape[0] / n_rows)
    width = int(frame.shape[1] / n_cols)
    avgs = []
    
    for row in range(n_rows):
        for col in range(n_cols):
            y0 = row * height
            y1 = min(y0+ height, frame.shape[0])
            x0 = col * width
            x1 = min(x0 + width, frame.shape[1])
            avgs.append(average(frame[y0:y1, x0:x1])/255)
    return avgs


def main():
    
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    fps = FPS().start()
    size = 600
    n_rows = 3
    n_cols = 3
    
    while True:
        frame = vs.read()
        frame = cv2.resize(frame, (size, size))
        # calculate average for tiles:
        tile_avgs = average_rgb_tiles(frame, n_rows, n_cols)
        print(tile_avgs)
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        
        fps.update()

    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()

if __name__=="__main__":
    main()