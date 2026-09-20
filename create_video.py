import cv2
from pathlib import Path

# =====================================================
# Folder containing images
# =====================================================

image_folder = Path(
    "X-Ray-Baggage-3/test/images"
)

# =====================================================
# Output video
# =====================================================

output_video = "dataset_video.mp4"

# =====================================================
# Read Images
# =====================================================

image_files = sorted(
    image_folder.glob("*.jpg")
)

print(f"Total Images Found: {len(image_files)}")

if len(image_files) == 0:

    raise Exception("No images found!")

# =====================================================
# Read First Image
# =====================================================

first = cv2.imread(str(image_files[0]))

height, width = first.shape[:2]

# =====================================================
# Create Video Writer
# =====================================================

fps = 30
display_time = 1.5

frames_per_image = int(fps * display_time)

print(f"FPS: {fps}")
print(f"Frames per image: {frames_per_image}")
print(f"Total images: {len(image_files)}")

expected_duration = (len(image_files) * frames_per_image) / fps

print(f"Expected Duration: {expected_duration:.2f} seconds")

writer = cv2.VideoWriter(
    output_video,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# =====================================================
# Write Frames
# =====================================================

for img_path in image_files:

    frame = cv2.imread(str(img_path))

    for _ in range(frames_per_image):

        writer.write(frame)

writer.release()

print("\nVideo Created Successfully!")

print(output_video)