"""Simple camera test to check if camera returns actual video data"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print(f"Camera opened with CAP_DSHOW: {cap.isOpened()}")

# Try different camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 200)
cap.set(cv2.CAP_PROP_CONTRAST, 100)
cap.set(cv2.CAP_PROP_SATURATION, 200)

print("Reading frames...")
for i in range(100):
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i}: Failed to read")
        break
    
    # Calculate frame statistics
    mean = np.mean(frame)
    std = np.std(frame)
    min_val = np.min(frame)
    max_val = np.max(frame)
    
    if i % 10 == 0:
        print(f"Frame {i}: mean={mean:.1f}, std={std:.1f}, min={min_val}, max={max_val}")
    
    # Save frame 50
    if i == 50:
        cv2.imwrite("snapshots/camera_frame_50.png", frame)
        print(f"Saved frame 50: mean={mean:.1f}, std={std:.1f}")
    
    # Display frame
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Done!")
cap.release()
cv2.destroyAllWindows()
