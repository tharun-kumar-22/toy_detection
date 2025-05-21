# Object Detection for a Toy Manufacturer

In this project, we developed an object detection system to identify and classify toy products on a conveyor belt for a manufacturer. We tested **two different models** to compare performance:

---

## Main Folders

### 🔹 Task-1-YOLO-Important/
Contains our full pipeline using **YOLOv8**:
- Dataset preparation  
- Model training  
- Inference (on images and videos)  
- Final output video and tracking logs  

**Recommended Folder**:  
> YOLO performed the best in terms of speed and accuracy. Please check this folder for our final results and inference demos.

---

### 🔹 Task-1-FasterRCNN/
Contains an alternative implementation using **Faster R-CNN**:
- Slower but still accurate  
- Used mainly for comparison  
- Same toy dataset used

---

## Why YOLO?

- **Faster** inference time (real-time capable)
- **Better accuracy** on small toy objects
- Easier deployment and integration with apps

---

## Deliverables in Each Folder

Both folders include:
- Jupyter Notebooks for training & testing
- Scripts for running detection
- Pretrained weights
- Output videos showing detection results

---

This setup demonstrates a full evaluation pipeline, helping choose the right model for real-world deployment.
