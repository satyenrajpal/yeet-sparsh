import numpy as np

def average(frame):
    return np.average(frame)

# Average tiles in row major format
def average_tiles(frame,coords, max_val, scale = 1):
    avgs = []
    for section in coords:
        for top_left_x, top_left_y, bottom_right_x, bottom_right_y in section:
            avgs.append(min(average(frame[top_left_x:bottom_right_x, top_left_y:bottom_right_y]) * scale /max_val, 1.0))
    return avgs
