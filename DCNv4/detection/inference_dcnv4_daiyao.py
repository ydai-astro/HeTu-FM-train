import argparse
import os
import glob
import json
import numpy as np
import torch
from pycocotools import mask as mask_utils

from mmdet.apis import init_detector, inference_detector
import mmdet_custom
import mmcv_custom
import time

def mmdet_result_to_custom(result, score_thr=0.5):
    if isinstance(result, tuple):
        bbox_results, segm_results = result
    else:
        bbox_results, segm_results = result, None

    labels, scores, bboxes, masks = [], [], [], []

    for cls_id in range(len(bbox_results)):
        bboxes_per_cls = bbox_results[cls_id]
        if bboxes_per_cls is None or len(bboxes_per_cls) == 0:
            continue

        segms_per_cls = segm_results[cls_id] if segm_results is not None else None

        for i in range(bboxes_per_cls.shape[0]):
            x1, y1, x2, y2, score = bboxes_per_cls[i]
            if score < score_thr:
                continue

            labels.append(cls_id)
            scores.append(float(score))
            bboxes.append([float(x1), float(y1), float(x2), float(y2)])

            if segms_per_cls is not None:
                m = segms_per_cls[i].astype(np.uint8)
                rle = mask_utils.encode(np.asfortranarray(m))
                rle["counts"] = rle["counts"].decode("ascii")
                masks.append({
                    "size": list(rle["size"]),
                    "counts": rle["counts"]
                })

    return {
        "labels": labels,
        "scores": scores,
        "bboxes": bboxes,
        "masks": masks
    }


def parse_args():
    p = argparse.ArgumentParser("MMDet 2.8 batch inference test")
    p.add_argument("--input", required=True, help="SB id, e.g. 20147")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--score_thr", type=float, default=0.5)
    return p.parse_args()


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main():
    args = parse_args()
    sb = args.input

    # ===== 路径（和你之前的一致）=====
    config_path = "./configs/ast/mask_b.py"
    checkpoint = "./work_dirs/mask_b/best_bbox_mAP_epoch_52.pth"

    img_dir = f"/groups/g9600009/home/share/HeTu/racsmid/test/png/images_SB{sb}/lin.cmap11.0/"
    out_base = f"./output_dcnv4_mask_b/{sb}/"
    png_dir = os.path.join(out_base, "vis")
    json_dir = os.path.join(out_base, "pred")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    print(f"[SB {sb}] total images: {len(imgs)}")

    if len(imgs) == 0:
        print("No images found. Exit.")
        return

    torch.backends.cudnn.benchmark = True

    # ===== init model（只一次）=====
    model = init_detector(config_path, checkpoint, device="cuda:0")
    start_time = time.time()
    processed = 0


    # ===== 批量推理（核心）=====
    with torch.no_grad():
        for batch_imgs in chunks(imgs, args.batch_size):
            print(f"Running batch of {len(batch_imgs)} images")

            # 关键：list 直接喂给 inference_detector
            results = inference_detector(model, batch_imgs)

            # 期望返回 list，长度 == batch_size
            if not isinstance(results, (list, tuple)):
                raise RuntimeError("Batch inference did not return a list. This env/model may not support batch.")

            if len(results) != len(batch_imgs):
                raise RuntimeError("Result length mismatch with input batch.")

            for img_path, result in zip(batch_imgs, results):
                name = os.path.splitext(os.path.basename(img_path))[0]
                png_out = os.path.join(png_dir, name + ".png")
                json_out = os.path.join(json_dir, name + ".json")

                model.show_result(img_path, result, out_file=png_out)

                with open(json_out, "w") as f:
                    json.dump(
                        mmdet_result_to_custom(result, score_thr=args.score_thr),
                        f
                    )

                processed += 1
                if processed % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {processed} images in {elapsed:.2f} seconds ({processed/elapsed:.2f} images/sec)")

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

###python inference_dcnv4_daiyao.py --input 20147 --batch_size 16

