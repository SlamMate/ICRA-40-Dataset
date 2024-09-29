import cv2
import os
from imutils import paths
from tqdm import tqdm

# Set the paths for input and output folders
input_folder = '/media/zhangqi/DA18EBFA09C1B27D/Datasets/Dataset1/rgb/front'   # Replace with your input folder path
output_folder = '/media/zhangqi/DA18EBFA09C1B27D/Datasets/Dataset1/rgb/front_blur' # Replace with your output folder path

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the YuNet face detection model
modelFile = '/home/zhangqi/Downloads/face_detection_yunet_2023mar.onnx'  # Replace with actual path
net = cv2.FaceDetectorYN_create(
    model=modelFile,
    config='',
    input_size=(320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000,
    backend_id=cv2.dnn.DNN_BACKEND_DEFAULT,
    target_id=cv2.dnn.DNN_TARGET_CPU
)

# Process each image
for image_path in tqdm(list(paths.list_images(input_folder)), desc='Processing images'):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Unable to read image {image_path}, skipping.")
        continue

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)

    # Set input image size
    h, w, _ = img.shape
    net.setInputSize((w, h))

    # Detect faces
    faces = net.detect(img)

    # If no faces are detected, save the original image
    if faces[1] is None:
        cv2.imwrite(output_path, img)
        continue

    # Iterate over the detected faces and blur them
    for face in faces[1]:
        x1, y1, w_face, h_face = map(int, face[:4])
        x2, y2 = x1 + w_face, y1 + h_face

        # Ensure coordinates are within the image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        # Extract the face region and blur it
        face_region = img[y1:y2, x1:x2]
        face_region = cv2.GaussianBlur(face_region, (199, 199), 50)
        img[y1:y2, x1:x2] = face_region

    # Save the processed image
    cv2.imwrite(output_path, img)

print("Processing complete, blurred images have been saved to:", output_folder)

