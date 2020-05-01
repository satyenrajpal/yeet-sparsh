from imutils.video import VideoStream
from imutils.video import FPS
import imutils
from utils import average_tiles

def run_camera():
    
    vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    fps = FPS().start()
    size = 600
    n_rows = 3
    n_cols = 3
    pwm_outputs = get_pwm_outputs()
    
    while True:
        frame = vs.read()
        frame = cv2.resize(frame, (size, size))
        # calculate average for tiles:
        tile_avgs = average_tiles(frame, n_rows, n_cols, 255)
        
        for pwm, avg in zip(pwm_outputs, tile_avgs):
            pwm.value = avg
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        
        fps.update()

    fps.stop()
    cv2.destroyAllWindows()
    vs.stop()
