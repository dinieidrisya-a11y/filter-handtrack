"""
Interactive camera 'portal' filters using MediaPipe + OpenCV.

How to use:
 - Run: python portal_filter.py
 - Move your index finger to place the portal center.
 - Change portal size by changing distance between index finger tip and thumb tip.
 - Pinch (thumb + index touch briefly) to cycle to the next filter.
 - Press 's' to save a screenshot, 'q' to quit.
"""
import cv2
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except Exception as _e:
    raise ImportError(f"Could not import mediapipe 'solutions' API: {_e}")
    
import numpy as np
import time
import os


try:
    import optional_module  # pyright: ignore[reportMissingImports]
except ImportError:
    optional_module = None
  

# --- Filter functions ---
def apply_mono(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

def apply_dualtone(img, color1=(20, 90, 160), color2=(200, 160, 30)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    out = np.zeros_like(img, dtype=np.uint8)
    for c in range(3):
        out[..., c] = np.clip((color1[c] * (1 - gray) + color2[c] * gray), 0, 255)
    return out

def apply_pixelate(img, blocks=0.05):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, int(w*blocks)), max(1, int(h*blocks))), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def apply_invert(img):
    return 255 - img

def apply_sepia(img):
    kernel = np.array([[0.272, 0.534, 0.131],
                       [0.349, 0.686, 0.168],
                       [0.393, 0.769, 0.189]])
    sep = cv2.transform(img, kernel)
    sep = np.clip(sep, 0, 255).astype(np.uint8)
    return sep

def apply_blur(img, k=21):
    k = k if k % 2 == 1 else k+1
    return cv2.GaussianBlur(img, (k, k), 0)

def apply_thermal(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return colored

def apply_sketch(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    inv = 255.0 - gray
    blur = cv2.GaussianBlur(inv, (21,21), 0)
    sketch = np.minimum((gray * 255.0) / (blur + 1e-6), 255.0).astype(np.uint8)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)

def apply_glitch(img, frame_idx=0):
    h, w = img.shape[:2]
    out = img.copy()
    shift = max(1, int(0.01 * w))
    # split channels and shift
    b, g, r = cv2.split(img)
    b = np.roll(b, shift * ((frame_idx % 5) - 2), axis=1)
    r = np.roll(r, -shift * ((frame_idx % 7) - 3), axis=1)
    out = cv2.merge([b,g,r])
    # horizontal scanlines
    overlay = out.copy()
    for i in range(0, h, 2):
        overlay[i:i+1, :, :] = overlay[i:i+1, :, :] * 0.9
    out = cv2.addWeighted(out, 0.9, overlay, 0.1, 0)
    return out

def apply_neon(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    edges_col = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)
    neon = cv2.addWeighted(img, 0.6, edges_col, 0.9, 0)
    neon = cv2.GaussianBlur(neon, (5,5), 0)
    return neon

# Simple galaxy: color map + twinkling stars
def apply_galaxy(img, frame_idx=0):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_OCEAN)
    # add stars
    h, w = gray.shape
    stars = np.zeros((h, w), dtype=np.uint8)
    np.random.seed(frame_idx % 1000)
    num = max(10, (w*h)//50000)
    xs = np.random.randint(0, w, size=num)
    ys = np.random.randint(0, h, size=num)
    for x,y in zip(xs,ys):
        cv2.circle(stars, (x,y), radius=np.random.randint(1,3), color=255, thickness=-1)
    stars_col = cv2.cvtColor(stars, cv2.COLOR_GRAY2BGR)
    # brighten stars
    colored = cv2.add(colored, stars_col)
    colored = cv2.GaussianBlur(colored, (3,3), 0)
    return colored

FILTERS = [
    ("Mono", apply_mono),
    ("Dual-Tone", apply_dualtone),
    ("Pixelate", apply_pixelate),
    ("Invert", apply_invert),
    ("Sepia", apply_sepia),
    ("Blur", apply_blur),
    ("Thermal", apply_thermal),
    ("Sketch", apply_sketch),
    ("Glitch", apply_glitch),
    ("Neon", apply_neon),
    ("Galaxy", apply_galaxy),
]

# --- Helper functions ---
def circle_mask(h, w, center, radius, feather=20):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, int(max(1, radius)), 255, -1)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (feather|1, feather|1), 0)
    return mask

def landmarks_to_pixel(landmark, w, h):
    return int(landmark.x * w), int(landmark.y * h)

# --- Main app ---
def main():
    # Camera index: use 0 since test_camera showed index 0 works
    cam_index = 0
    # Use CAP_DSHOW which works better on Windows
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {cam_index}. Try changing cam_index variable.")
        return

    # Set camera properties - MAXIMUM brightness since camera is very dark
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 255)  # MAXIMUM
    cap.set(cv2.CAP_PROP_CONTRAST, 127)
    cap.set(cv2.CAP_PROP_SATURATION, 255)  # MAXIMUM
    cap.set(cv2.CAP_PROP_GAIN, 100)
    
    os.makedirs("snapshots", exist_ok=True)
    
    # Warm up the camera - discard first frames
    print("Warming up camera...")
    for i in range(30):
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Could not read from camera during warmup.")
            cap.release()
            return
        if i % 10 == 0:
            print(f"  Warmup frame {i}...")
    print("Camera ready!")

    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.6)

    # ...rest of your main() continues here, also indented...

    filter_idx = 0
    prev_pinch = False
    frame_idx = 0

    # FPS smoothing
    prev_time = time.time()
    fps = 0.0
    show_fps = False

    print("Controls: 'n' next, 'p' previous, 'd' toggle FPS, 's' save, 'q' quit")
    print(f"Camera resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: frame not read from camera.")
            break
        
        # AGGRESSIVE brightness enhancement for very dark camera
        # Use histogram equalization for better contrast
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # Aggressively boost brightness channel
        v = cv2.equalizeHist(v)
        v = np.uint8(np.clip(v.astype(np.float32) * 1.8, 0, 255))
        frame = cv2.merge([h, s, v])
        frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)

        frame = cv2.flip(frame, 1)  # mirror
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        portal_center = None
        portal_radius = min(w, h) // 8

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            idx_tip = hand.landmark[8]
            thumb_tip = hand.landmark[4]
            cx, cy = landmarks_to_pixel(idx_tip, w, h)
            tx, ty = landmarks_to_pixel(thumb_tip, w, h)
            portal_center = (cx, cy)
            dist = np.hypot(cx - tx, cy - ty)
            portal_radius = int(np.clip(dist * 1.8, 20, min(w, h)//2))

            # Pinch detection (advance filter) — tune this threshold if needed
            pinch = dist < 40
            if pinch and not prev_pinch:
                filter_idx = (filter_idx + 1) % len(FILTERS)
            prev_pinch = pinch

            # draw hand landmarks for feedback
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(80,80,200), thickness=1, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(80,80,50), thickness=1))
        else:
            prev_pinch = False

        # prepare filtered image
        if portal_center is not None:
            name, func = FILTERS[filter_idx]
            try:
                filtered = func(frame.copy(), frame_idx) if func.__code__.co_argcount >= 2 else func(frame.copy())
            except Exception:
                filtered = func(frame.copy())
            mask = circle_mask(h, w, portal_center, portal_radius, feather=41)
            mask3 = cv2.merge([mask, mask, mask]) / 255.0
            out = (frame.astype(np.float32) * (1.0 - mask3) + filtered.astype(np.float32) * mask3).astype(np.uint8)
        else:
            out = frame.copy()
            name = FILTERS[filter_idx][0]

        # overlay UI text
        cv2.putText(out, f"Filter: {name} ({filter_idx+1}/{len(FILTERS)})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(out, "Pinch or 'n' to next, 'p' prev, 'd' toggle FPS, 's' save, 'q' quit",
                    (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

        # FPS calculation (smoothed)
        now = time.time()
        dt = now - prev_time
        if dt > 0:
            inst_fps = 1.0 / dt
            fps = 0.9 * fps + 0.1 * inst_fps if fps > 0 else inst_fps
        prev_time = now
        if show_fps:
            cv2.putText(out, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("Portal Filters (MediaPipe + OpenCV)", out)

        # handle keys: q=quit, s=save, n=next, p=prev, d=toggle FPS
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            fname = f"snapshots/shot_{int(time.time())}.png"
            cv2.imwrite(fname, out)
            print("Saved:", fname)
        if key == ord('n'):
            filter_idx = (filter_idx + 1) % len(FILTERS)
        if key == ord('p'):
            filter_idx = (filter_idx - 1) % len(FILTERS)
        if key == ord('d'):
            show_fps = not show_fps

        frame_idx += 1

    hands.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()