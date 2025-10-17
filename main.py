import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn_v2,
    maskrcnn_resnet50_fpn,
)
from tqdm import tqdm
from ultralytics import YOLO

from utils import COCO_LABELS, display_table, get_documented_metrics, get_yolo_metrics

# pyright: ignore
# or
# pylint: disable-all


def run_yolo_video(
    input_path="./test.mp4", output_path="output_yolo.mp4", confidence_threshold=0.5
):
    print("[YOLOv8] Loading model...")
    model = YOLO("yolov8m-seg.pt")
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in tqdm(range(frame_count), desc="YOLOv8"):
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=confidence_threshold, verbose=False)
        out.write(results[0].plot())
    cap.release()
    out.release()
    print(f"[YOLOv8] Saved → {output_path}")


def run_maskrcnn_video(
    input_path="./test.mp4", output_path="output_maskrcnn.mp4", confidence_threshold=0.5
):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights).eval()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with torch.no_grad():
        for _ in tqdm(range(frame_count), desc="Mask R-CNN"):
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = F.to_tensor(rgb_frame)
            preds = model([img_tensor])[0]

            boxes = preds["boxes"].cpu().numpy()
            masks = preds["masks"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()

            for box, mask, label_id, score in zip(boxes, masks, labels, scores):
                if score < confidence_threshold:
                    continue

                mask_2d = (mask[0] > 0.5).astype(np.uint8)
                mask_colored = np.zeros_like(frame, dtype=np.uint8)
                mask_colored[mask_2d == 1] = [0, 255, 0]
                cv2.addWeighted(frame, 1, mask_colored, 0.3, 0, frame)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                label_text = f"{COCO_LABELS[label_id-1] if label_id <= len(COCO_LABELS) else label_id}: {score:.2f}"
                cv2.putText(
                    frame,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            out.write(frame)

    cap.release()
    out.release()
    print(f"[Mask R-CNN] Saved → {output_path}")


def run_fasterrcnn_video(
    input_path="./test.mp4",
    output_path="output_fasterrcnn.mp4",
    confidence_threshold=0.5,
):
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).eval()

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    with torch.no_grad():
        for _ in tqdm(range(frame_count), desc="Faster R-CNN"):
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = F.to_tensor(rgb_frame)
            preds = model([img_tensor])[0]

            boxes = preds["boxes"].cpu().numpy()
            scores = preds["scores"].cpu().numpy()
            labels = preds["labels"].cpu().numpy()

            for box, score, label_id in zip(boxes, scores, labels):
                if score < confidence_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label_text = f"{COCO_LABELS[label_id-1] if label_id <= len(COCO_LABELS) else label_id}: {score:.2f}"
                cv2.putText(
                    frame,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                )

            out.write(frame)

    cap.release()
    out.release()
    print(f"[Faster R-CNN] Saved → {output_path}")


if __name__ == "__main__":
    yolo_metrics = get_yolo_metrics()
    other_metrics = get_documented_metrics()
    display_table(yolo_metrics, other_metrics)
    run_yolo_video()
    run_maskrcnn_video()
    run_fasterrcnn_video()
