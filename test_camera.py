import cv2
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    print("index", i, "opened:", ok)
    if ok:
        ret, frame = cap.read()
        print("  read frame:", ret, "frame shape:", None if not ret else frame.shape)
    cap.release()
