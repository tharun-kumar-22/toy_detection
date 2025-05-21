
from src.dataset import CocoDataset, Resizer, Normalizer
from torchvision import transforms
from pycocotools.cocoeval import COCOeval
import json
import torch
import os
import numpy as np
import pandas as pd


def print_classwise_metrics(coco_eval):
    """
    Print per-class COCO metrics in table form:
      - Images: number of images with at least one GT of that class
      - Instances: total GT boxes for that class
      - Box (P): precision at IoU=0.50
      - R: recall at IoU=0.50
      - mAP50: AP at IoU=0.50
      - mAP50–95: average AP over IoU thresholds 0.50:0.95
    """
    # Extract precision and recall arrays
    precision = coco_eval.eval.get("precision")  # shape [T, R, K, A, M]
    recall    = coco_eval.eval.get("recall")     # shape [T, K, A, M]
    cat_ids   = coco_eval.params.catIds
    T = precision.shape[0] if precision is not None else 0

    rows = []
    for idx, cat_id in enumerate(cat_ids):
        # Safe category name lookup
        cats = coco_eval.cocoGt.loadCats([cat_id])
        if cats and isinstance(cats, list):
            name = cats[0].get('name', str(cat_id))
        else:
            name = str(cat_id)

        # Count images & instances
        img_ids = coco_eval.cocoGt.getImgIds(catIds=[cat_id])
        n_imgs  = len(img_ids)
        ann_ids = coco_eval.cocoGt.getAnnIds(imgIds=img_ids, catIds=[cat_id])
        n_inst  = len(ann_ids)

        # Precision @ IoU=0.50
        if precision is not None:
            pr50 = precision[0, :, idx, 0, 0]
            box_p = float(np.mean(pr50[pr50 > -1]))
        else:
            box_p = float('nan')

        # Recall @ IoU=0.50
        if recall is not None:
            rec = float(recall[0, idx, 0, 0])
        else:
            rec = float('nan')

        # mAP over all IoUs
        ap_list = []
        for t in range(T):
            pr_t = precision[t, :, idx, 0, 0]
            valid = pr_t[pr_t > -1]
            if valid.size:
                ap_list.append(valid.mean())
        mAP = float(np.mean(ap_list)) if ap_list else float('nan')

        rows.append({
            'Class':       name,
            'Images':      n_imgs,
            'Instances':   n_inst,
            'Box (P)':     box_p,
            'R':           rec,
            'mAP50':       box_p,
            'mAP50–95':    mAP
        })

    df = pd.DataFrame(rows)
    pd.options.display.float_format = '{:0.3f}'.format
    print("\nPer-class COCO metrics:\n")
    print(df.to_markdown(index=False))
    print()


def evaluate_coco(dataset, model, threshold=0.05):
    model.eval()
    with torch.no_grad():
        results = []
        image_ids = []

        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            imgs  = data['img'].cuda().permute(2,0,1).float().unsqueeze(0)
            scores, labels, boxes = model(imgs)
            boxes /= scale

            if boxes.numel() > 0:
                boxes[:,2:] -= boxes[:,:2]
                for i in range(boxes.size(0)):
                    score = float(scores[i])
                    if score < threshold:
                        break
                    label = int(labels[i])
                    box   = boxes[i].tolist()
                    results.append({
                        'image_id':    dataset.image_ids[index],
                        'category_id': dataset.label_to_coco_label(label),
                        'score':       score,
                        'bbox':        box,
                    })

            image_ids.append(dataset.image_ids[index])
            # print(f"{index+1}/{len(dataset)}", end='\r')

        if not results:
            print("No detections to evaluate.")
            return

        res_file = f"{dataset.set_name}_bbox_results.json"
        with open(res_file, 'w') as f:
            json.dump(results, f, indent=4)

        coco_true = dataset.coco
        coco_pred = coco_true.loadRes(res_file)

        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        # coco_eval.summarize()

        # Print per-class breakdown
        print_classwise_metrics(coco_eval)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # Load checkpoint
    ckpt_path = os.path.join("trained_models", "toys_effdet.pth")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    efficientdet = ckpt.module if hasattr(ckpt, 'module') else ckpt
    efficientdet = efficientdet.to(device)

    # Build dataset
    val_path = os.path.join("data", "COCO")
    dataset_val = CocoDataset(
        val_path,
        set='val2017',
        transform=transforms.Compose([Normalizer(), Resizer()])
    )

    evaluate_coco(dataset_val, efficientdet)
