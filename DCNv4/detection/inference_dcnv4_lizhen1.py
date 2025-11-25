import sys
import mmcv
import argparse
import os
from mmdet.apis import inference_detector, init_detector  # 2.x 版本常用的推理 API
import mmdet_custom
import mmcv_custom

import json
import numpy as np
from pycocotools import mask as mask_utils


def mmdet_result_to_custom(result, score_thr=0.0):
    """
    将 mmdetection 的 (bbox_results, segm_results) 转成：
    {
        "labels": [...],
        "scores": [...],
        "bboxes": [...],
        "masks": [{"size": [H, W], "counts": "..."}]
    }
    """
    # 有的模型只返回 bbox_results，有的返回 (bbox_results, segm_results)
    if isinstance(result, tuple):
        bbox_results, segm_results = result
    else:
        bbox_results, segm_results = result, None

    labels = []
    scores = []
    bboxes = []
    masks = []

    num_classes = len(bbox_results)

    for cls_id in range(num_classes):
        bboxes_per_cls = bbox_results[cls_id]  # (Ni, 5)
        if bboxes_per_cls is None or len(bboxes_per_cls) == 0:
            continue

        # 这个类别对应的 mask list（如果有）
        if segm_results is not None:
            segms_per_cls = segm_results[cls_id]
        else:
            segms_per_cls = None

        for i in range(bboxes_per_cls.shape[0]):
            x1, y1, x2, y2, score = bboxes_per_cls[i]

            # 根据需要加一个 score 阈值
            if score < score_thr:
                continue

            # label：这里用 0-based 的 cls_id，如果你想要 1-based，就用 cls_id + 1
            labels.append(cls_id)
            scores.append(float(score))
            bboxes.append([float(x1), float(y1), float(x2), float(y2)])

            if segms_per_cls is not None:
                # segm_results[cls_id][i] 是一个 (H, W) 的二值 mask（0/1）
                m = segms_per_cls[i].astype(np.uint8)
                # pycocotools 需要 Fortran contiguous
                rle = mask_utils.encode(np.asfortranarray(m))
                # rle["counts"] 是 bytes，要转成 str
                rle["counts"] = rle["counts"].decode("ascii")
                masks.append({
                    "size": list(rle["size"]),   # [H, W]
                    "counts": rle["counts"]      # RLE 字符串
                })

    return {
        "labels": labels,
        "scores": scores,
        "bboxes": bboxes,
        "masks": masks
    }


def parse_args():
    parser = argparse.ArgumentParser(description="MMDet inference")
    parser.add_argument("--input", default="", help="input file name")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_filename = args.input

    # 模型配置文件路径
    config_path = "work_dirs/mask2/mask2.py"
    # 模型权重文件路径
    checkpoint = "work_dirs/mask2/best_bbox_mAP_epoch_24.pth"

    # 输入图片路径
    # img_path = f"20147/{input_filename}"
    # 输出目录
    output_dir = f"/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/ast/results/{input_filename}/"
    os.makedirs(output_dir, exist_ok=True)

    # 初始化检测模型
    model = init_detector(config_path, checkpoint, device="cuda:0")

    # 处理批量图片
    # imgs = sorted([os.path.join(img_path, img) for img in os.listdir(img_path) if img.endswith(".png")])

    # 逐张图片推理
    for img in ["/inspire/qb-ilm/project/qproject-assement/zhangkaipeng-24043/ast/lin.cmap11.0/SB20147_component_1a.png"]:
        result = inference_detector(model, img)

        a, b = result
        # print(a.type)
        # print(b.type)
        # print(a)
        # print(b)
        output_file = os.path.join(output_dir, os.path.basename(img))
        # 保存推理结果
        model.show_result(img, result, out_file=output_file)
        with open(os.path.join(output_dir, os.path.splitext(os.path.basename(img))[0] + ".json"), "w") as f:
            json.dump(mmdet_result_to_custom(result, score_thr=0.5), f)
        break


if __name__ == "__main__":
    main()
