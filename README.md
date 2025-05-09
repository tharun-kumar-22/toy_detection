# Toy Detection using Object Detection

This project focuses on developing a robust object detection model for a toy manufacturer. The objective is to detect various toy products on a conveyor belt under different orientations, except for the underside.

## Overview

- **Goal**: Train an object detection model that reliably identifies toy objects in real-time.
- **Challenge**: Toys appear in multiple orientations (excluding the underside), requiring the model to generalize well.
- **Application**: Supports automation of sorting lines in industrial settings.

## Dataset

- Custom dataset created via self-acquisition.
- Annotated using [Label Studio](https://labelstud.io/) with bounding boxes.
- Split into training and test sets for effective evaluation.

## Model

- **Architecture**: YOLOv8, EfficienDet, FasterRCNN
- **Framework**: PyTorch
- **Training**: Conducted on annotated dataset with appropriate data augmentation.
- **Evaluation**: Accuracy assessed on a small test set and through real-time video demonstration.

## Tools Used

- Python
- YOLOv5
- OpenCV
- Label Studio
- PyTorch

## Demo

A video demonstration shows real-time detection of toys on a conveyor-like setup, with bounding boxes and class predictions.


