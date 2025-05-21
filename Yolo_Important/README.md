 Object Detection for a Toy Manufacturer using YOLO

This project trains a **YOLO-based object detection model** to identify toy products on a simulated conveyor belt. It is part of a PoC (Proof of Concept) for automating the sorting process in a toy manufacturing line.

you may find missing folders here(everything included here): https://drive.google.com/drive/folders/1oWnaZq98cUe4d13vEptk78xjcL8wtxaa?usp=sharing

---

## Folder & File Descriptions

### Folders

- **`.ipynb_checkpoints/`**  
  → Auto-saved Jupyter notebook backups. *(Can be ignored)*

- **`runs/`**  
  → Output folder created by YOLO training and validation (includes detection results and metrics).

- **`Toy_Data/`**  
  → Main dataset folder. Contains labeled images used for training the object detection model.

- **`toy_model_results/`**  
  → Folder to store test predictions or result images after model inference.

---

### Important Files

- - **`app_run`** 
  → Script to run the toy detection model (e.g., using webcam or test image/video). Its a streamlit app.

- **`Inference_Video.mp4`**  
  → Video used for testing/inference to visualize toy detection in action.

- **`output_video.mp4`**  
  → Final output video with detected toy bounding boxes drawn.

- **`Task-1-Main_notebook.ipynb`**  
  → Main notebook for training, evaluating, and visualizing the toy detection model.

- **`Task-1-Main_notebook.html`**  
  → Main notebook for training, evaluating, and visualizing the toy detection model in html.

- **`toy_data_preparation.ipynb`**  
  → Preprocessing notebook to prepare and annotate the toy dataset.

- **`yolov8n.pt / yolov8s.pt`**  
  → Pretrained YOLOv8 models used as a base:
  - `n` = nano version (very fast)
  - `s` = small version (better accuracy)

---

##  Objective

Train a **YOLOv8 model** to detect and classify different types of toys on a conveyor belt. The final model can be used to automate sorting in the manufacturing process.

---

### Key Tasks Performed

1. **Dataset Preparation** (`toy_data_preparation.ipynb`)
2. **Model Training** using YOLOv8
3. **Inference Testing** on real or sample videos
4. **Results Visualization** and saving outputs
5. (Optional) Script-based detection using `app.py`

---

###  Output

- **Trained model** stored in `runs/`
- **Detection video** with predictions
- **Sample results** in `toy_model_results/`
