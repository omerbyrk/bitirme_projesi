from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from numpy import expand_dims
from numpy import mean
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap, compute_recall
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image


def evaluate_model(dataset, model, cfg):
  APs = list(); 
  ARs = list(); 
  F1_scores = list(); 
  count = 0
  for image_id in dataset.image_ids:
    if(count == 100):
      break
    image_id = random.randint(0, len(dataset.image_ids))
    count = count + 1;
    print(str(count) + "/" + str(len(dataset.image_ids)))
    image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
    yhat = model.detect([image, image], verbose=0)
    r = yhat[0]
    AP, precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
    AR, positive_ids = compute_recall(r["rois"], gt_bbox, iou=0.2)
    ARs.append(AR)
    F1_scores.append((2* (mean(precisions) * mean(recalls)))/(mean(precisions) + mean(recalls)))
    APs.append(AP)
  
  mAP = mean(APs)
  mAR = mean(ARs)
  return mAP, mAR, F1_scores
