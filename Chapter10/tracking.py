import cv2
import numpy as np
from pylab import *

def draw_flow(im, flow, step=16):
    """ Plot optical flow at sample points
        spaced step pixel apart. """
    h, w = im.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1) # Làm phẳng tọa độ
    fx, fy = flow[y, x].T 

    # create line endpoints
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    # create image and draw
    vis = cv2.cvtColor(im, cv2.cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

# # setup video capture
# cap = cv2.VideoCapture(0)

# ret, im = cap.read()
# prev_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# while True:
#     # get grayscale image
#     ret, im = cap.read()
#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#     # compute flow
#     flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     prev_gray = gray

#     # plot the flow vectors
#     cv2.imshow('Optical flow', draw_flow(gray, flow))
#     if cv2.waitKey(10)==27:
#         break

# Using the tracker
import lktrack

imnames = ['bt.003.pgm', 'bt.002.pgm', 'bt.001.pgm', 'bt.000.pgm']

# # create tracker object
# lkt = lktrack.LKTracker(imnames)

# # detect in first frame, track in the remaining
# lkt.detect_points()
# lkt.draw()
# for i in range(len(imnames)-1):
#     lkt.track_points()
#     lkt.draw()

# tracking using the LKTracker generator
lkt = lktrack.LKTracker(imnames)
for im, ft in lkt.track():
    print('tracking %d features' % len(ft))

# plot the tracks
figure()
imshow(im)
for p in ft:
    plot(p[0], p[1], 'bo')
for t in lkt.tracks:
    plot([p[0] for p in t], [p[1] for p in t])
axis('off')
show()

