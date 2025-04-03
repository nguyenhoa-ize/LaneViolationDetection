import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Kiểm tra GPU và sử dụng nếu có
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 model lên GPU/CPU
model = YOLO("models/my_model.pt").to(device)

def detect_vehicles(image):
    """
    Nhận diện phương tiện trong ảnh bằng YOLO, xử lý nhanh nhất có thể.
    """
    results = model(image, verbose=False)  # Tắt log để tăng tốc
    vehicle_boxes = []

    for result in results:
        if result.boxes is not None:
            boxes = result.boxes.data.cpu().numpy()  # Chuyển tensor thành numpy array
            vehicle_boxes = boxes[:, :6].astype(np.int32)  # Chỉ lấy thông tin cần thiết

    return vehicle_boxes  # [x1, y1, x2, y2, confidence, class_id]

def process_image(image_path, output_path="output.jpg"):
    """
    Xử lý nhận diện phương tiện trên ảnh và lưu kết quả.
    """
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Không thể tải ảnh!")
        return

    vehicles = detect_vehicles(image)

    for x1, y1, x2, y2, confidence, class_id in vehicles:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"ID:{class_id} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)
    print(f"✅ Ảnh đã lưu tại: {output_path}")

    cv2.imshow("Vehicle Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path, output_path="output.mp4"):
    """
    Xử lý video với nhận diện phương tiện theo thời gian thực.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Không thể mở video!")
        return

    width, height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        vehicles = detect_vehicles(frame)

        for x1, y1, x2, y2, confidence, class_id in vehicles:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{class_id} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        cv2.imshow("Vehicle Detection", frame)

        if cv2.waitKey(0):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Video đã lưu tại: {output_path}")

if __name__ == "__main__":
    # Xử lý ảnh
    process_image("data/1.jpg")

    # Xử lý video
    process_video("data/video.mp4")
