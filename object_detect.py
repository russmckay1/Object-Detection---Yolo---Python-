import os
import shutil
from datetime import datetime
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import cv2
from tkinter import Tk, Button, Label, Frame
from PIL import Image, ImageTk
from ultralytics import YOLO

# ----- CONFIGURATION -----
NEW_IMAGES_DIR = "new_images"
ARCHIVE_DIR = "archive"
CROPPED_COPY_DIR = "../A2D_gauge_reader"
LATEST_FILENAME = "latest.jpg"
DISPLAY_SIZE = (500, 500)  # Display size
ALLOWED_OBJECTS = {"clock", "remote", "cell phone"}  # allowed classes

# Ensure directories exist
os.makedirs(NEW_IMAGES_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(CROPPED_COPY_DIR, exist_ok=True)

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")  # YOLOv8 nano

# ----- HELPER FUNCTIONS -----
def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    # Rename to latest.jpg in new_images
    latest_path = os.path.join(NEW_IMAGES_DIR, LATEST_FILENAME)
    cv2.imwrite(latest_path, img)

    description_text = "No Object"
    confidence_value = 0.0
    cropped_resized = cv2.resize(img, DISPLAY_SIZE)
    img_resized = cv2.resize(img, DISPLAY_SIZE)

    # Run YOLO detection
    results = yolo_model(latest_path)
    result = results[0]

    if len(result.boxes) > 0:
        # Get principal object (largest bounding box)
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        areas = [(x2-x1)*(y2-y1) for x1, y1, x2, y2 in boxes]
        max_idx = areas.index(max(areas))
        x1, y1, x2, y2 = map(int, boxes[max_idx])
        description_text = result.names[int(class_ids[max_idx])]
        confidence_value = confidences[max_idx] * 100  # percentage

        # Crop and resize principal object
        cropped = img[y1:y2, x1:x2]
        cropped_resized = cv2.resize(cropped, DISPLAY_SIZE)
        img_resized = cv2.resize(img, DISPLAY_SIZE)

        # Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Archive images
        raw_archive = os.path.join(ARCHIVE_DIR, f"latest_raw_{timestamp}.jpg")
        cropped_archive = os.path.join(ARCHIVE_DIR, f"cropped_{description_text}_{timestamp}.jpg")
        shutil.move(latest_path, raw_archive)
        cv2.imwrite(cropped_archive, cropped_resized)

        # ---- Save cropped copy only if object is allowed ----
        if description_text.lower() in ALLOWED_OBJECTS:
            annotated_crop = cropped_resized.copy()
            overlay_text = f"{description_text} {confidence_value:.1f}%"
            cv2.putText(
                annotated_crop,
                overlay_text,
                (10, 40),  # position
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,  # font scale
                (0, 255, 0),  # green text
                3,  # thickness
                cv2.LINE_AA
            )
            cropped_copy_path = os.path.join(CROPPED_COPY_DIR, "latest.jpg")
            cv2.imwrite(cropped_copy_path, annotated_crop)
            print(f"Saved {description_text} image to {cropped_copy_path}")

        # Update UI
        update_ui(img_resized, cropped_resized)

    else:
        # If no detection, just update UI with resized original
        update_ui(img_resized, cropped_resized)

    # Delete the original image from new_images
    try:
        os.remove(image_path)
        print(f"Deleted original image: {image_path}")
    except Exception as e:
        print(f"Error deleting {image_path}: {e}")

def update_ui(original_img, cropped_img):
    # Convert BGR to RGB
    orig_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    # Convert to PIL and Tkinter images
    pil_orig = Image.fromarray(orig_rgb)
    pil_crop = Image.fromarray(crop_rgb)
    tk_orig = ImageTk.PhotoImage(pil_orig)
    tk_crop = ImageTk.PhotoImage(pil_crop)

    label_orig.config(image=tk_orig)
    label_orig.image = tk_orig
    label_crop.config(image=tk_crop)
    label_crop.image = tk_crop

# ----- WATCHDOG HANDLER -----
class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(".jpg"):
            print(f"New image detected: {event.src_path}")
            process_image(event.src_path)

# ----- TKINTER UI -----
root = Tk()
root.title("Object Detection UI")

# Start UI at top-left corner
root.geometry(f"{DISPLAY_SIZE[0]*2 + 30}x{DISPLAY_SIZE[1] + 100}+0+0")

frame = Frame(root)
frame.pack(padx=10, pady=10)

label_orig = Label(frame)
label_orig.pack(side="left", padx=5)

label_crop = Label(frame)
label_crop.pack(side="right", padx=5)

exit_btn = Button(root, text="Exit", command=root.destroy)
exit_btn.pack(pady=5)

# ----- START WATCHDOG -----
observer = Observer()
observer.schedule(ImageHandler(), NEW_IMAGES_DIR, recursive=False)
observer.start()

try:
    root.mainloop()
finally:
    observer.stop()
    observer.join()
