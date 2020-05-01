import numpy as np

def average(frame):
    return np.average(frame)

# Average tiles in row major format
def average_tiles(frame, n_rows, n_cols, max_val, scale = 1):
    height = int(frame.shape[0] / n_rows)
    width = int(frame.shape[1] / n_cols)
    avgs = []
    
    for row in range(n_rows):
        for col in range(n_cols):
            y0 = row * height
            y1 = min(y0+ height, frame.shape[0])
            x0 = col * width
            x1 = min(x0 + width, frame.shape[1])
            avgs.append(min(average(frame[y0:y1, x0:x1]) * scale /max_val, 1.0))
    return avgs
