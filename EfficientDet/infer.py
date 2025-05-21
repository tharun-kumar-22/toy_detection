
import argparse
import torch
from src.config import COCO_CLASSES, colors
import cv2
import numpy as np
import os

def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet: Video Inference"
    )
    parser.add_argument("--image_size", type=int, default=512,
                        help="The common width and height for all images")
    parser.add_argument("--cls_threshold", type=float, default=0.5,
                        help="Confidence threshold for displaying boxes")
    parser.add_argument("--pretrained_model", type=str,
                        default="trained_models/signatrix_efficientdet_coco.pth",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--input", type=str, default="test_videos/input.mp4",
                        help="Path to input video file")
    parser.add_argument("--output", type=str, default="test_videos/output.mp4",
                        help="Path to save annotated output video")
    return parser.parse_args()


def draw_predictions(image, scores, labels, boxes, threshold, classes, colors):
    """
    Draw bounding boxes and labels on the image.

    Args:
        image (np.ndarray): BGR image.
        scores (Tensor): detection scores.
        labels (Tensor): detection labels.
        boxes (Tensor): detection boxes in xyxy format.
        threshold (float): minimum score to display.
        classes (list): list of class names indexed by label.
        colors (list): list of BGR tuples for box colors.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    output = image.copy()
    for i in range(boxes.shape[0]):
        score = float(scores[i])
        if score < threshold:
            continue
        label = int(labels[i])
        xmin, ymin, xmax, ymax = boxes[i]
        # cast to int
        x1, y1, x2, y2 = map(int, (xmin, ymin, xmax, ymax))
        color = colors[label]
        # draw rectangle
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        # draw label background
        text = f"{classes[label]}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(output, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        # draw text
        cv2.putText(output, text, (x1 + 2, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    return output


def test(opt):
    # Load model
    ckpt = torch.load(opt.pretrained_model, weights_only=False)
    model = ckpt.module if hasattr(ckpt, 'module') else ckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    cap = cv2.VideoCapture(opt.input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(opt.output, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # preprocess
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = img.shape[:2]
        scale = opt.image_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (nw, nh))
        canvas = np.zeros((opt.image_size, opt.image_size, 3), dtype=np.float32)
        canvas[:nh, :nw] = img_resized
        tensor = torch.from_numpy(canvas).permute(2,0,1).unsqueeze(0).to(device)
        # inference
        with torch.no_grad():
            scores, labels, boxes = model(tensor)
            boxes /= scale
            boxes = boxes.cpu()
            scores = scores.cpu()
            labels = labels.cpu()
        # draw
        annotated = draw_predictions(frame, scores, labels, boxes, opt.cls_threshold, COCO_CLASSES, colors)
        out.write(annotated)

    cap.release()
    out.release()


if __name__ == "__main__":
    opt = get_args()
    test(opt)