import cv2

# setup video capture
cap = cv2.VideoCapture(0)

while True:
    ret, im = cap.read()
    blur = cv2.GaussianBlur(im,(0,0),5)
    cv2.imshow('video test', blur)
    key = cv2.waitKey(10)
    if key == 27:
        break
    if key == ord(' '):
        cv2.imwrite('vid_result.jpg', im)

# reading video to NumPy arrays



