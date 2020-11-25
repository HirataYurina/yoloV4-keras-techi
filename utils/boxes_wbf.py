# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:boxes_wbf.py
# software: PyCharm

import numpy as np


def compute_iou(fusion_box, boxes):
    """compute box iou
    "Weighted Boxes Fusion: ensembling boxes for object detection models"
    https://arxiv.org/abs/1910.13302

    Args:
        fusion_box: (5,)
                    np.array
        boxes:      (n, 5)
                    np.array

    Returns:
        ious

    """
    fusion_box_min = fusion_box[:2]
    fusion_box_max = fusion_box[2:4]
    boxes_min = boxes[..., :2]
    boxes_max = boxes[..., 2:4]
    fusion_wh = fusion_box_max - fusion_box_min
    boxes_wh = boxes_max - boxes_min
    fusion_area = fusion_wh[0] * fusion_wh[1]  # scalar
    boxes_area = boxes_wh[..., 0] * boxes_wh[..., 1]     # (n,)

    inter_min = np.maximum(fusion_box_min, boxes_min)
    inter_max = np.minimum(fusion_box_max, boxes_max)
    inter_wh = np.maximum(0, inter_max - inter_min)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    ious = inter_area / (fusion_area + boxes_area - inter_area)

    return ious


# TODO: test weighted nms in my projects
def weighted_boxes_fusion(boxes,
                          iou_thres,
                          score_thres):
    """weighted boxes fusion for every class

    Args:
        boxes:       np.array: [[x1, y1, x2, y2, score], ..., [x1, y1, x2, y2, score]]
                     the boxes are from same class
        iou_thres:   threshold of iou
        score_thres: threshold of score

    Returns:
        results: boxes have been fused
                 [x1, y1, x2, y2, score]

    """
    L = []
    # we try to find matched boxes in the list F
    F = []

    # filter boxes
    scores = boxes[..., 4]
    valid_index = np.where(scores > score_thres)[0]  # (num_valid,)
    valid_scores = scores[valid_index]
    # sort boxes
    index_sorted = np.argsort(valid_scores)[::-1]  # (num_valid,)
    boxes_sorted = boxes[index_sorted]

    results = []

    if len(boxes_sorted) > 0:
        selected_index = []

        if len(F) == 0:
            F.append(boxes_sorted[0])
            L.append([index_sorted[0]])
            selected_index.append(index_sorted[0])

        # remain index list
        remain_index = set(index_sorted) - set(selected_index)

        while len(remain_index) > 0:
            num_fusion = len(F)
            index = np.array(list(remain_index))  # has sorted
            remain_boxes = boxes[index]
            ious = compute_iou(F[num_fusion - 1], remain_boxes)  # (n,)
            matched_index = np.where(ious > iou_thres)[0]
            if len(matched_index) > 0:
                # add matched boxes to L at the pos position of F
                matched_index = index[matched_index]
                L[num_fusion - 1].extend(matched_index)
                remain_index = remain_index - set(matched_index)
            else:
                # add matched boxes to the end of L and F
                F.append(boxes[list(remain_index)[0]])
                L.append([list(remain_index)[0]])
                remain_index = remain_index - {list(remain_index)[0]}

        # compute weighted location
        L = np.array(L, dtype=object)
        num = len(F)
        for i in range(num):
            fusion_boxes = boxes[np.array(L[i]).astype('int32')]  # (n, 5)
            scores_ = fusion_boxes[..., -1]  # (n,)
            location = fusion_boxes[..., :4]  # (n, 4)
            # score_mean = np.array([np.mean(scores_)])
            score_max = np.array([np.max(scores_)])
            weighted_location = location * scores_[:, np.newaxis]
            weighted_location_sum = np.sum(weighted_location, axis=0)
            scores_sum = np.sum(scores_)
            results_location = weighted_location_sum / scores_sum  # (4,)
            results.append(np.concatenate([results_location, score_max]))

    return np.array(results)


if __name__ == '__main__':

    # test my code
    boxes_list = [
        [0.00, 0.51, 0.81, 0.91],
        [0.10, 0.31, 0.71, 0.61],
        [0.01, 0.32, 0.83, 0.93],
        [0.02, 0.53, 0.11, 0.94],
        [0.03, 0.24, 0.12, 0.35],
    ]
    scores_list = [0.9, 0.8, 0.2, 0.4, 0.7]
    scores_list = np.reshape(scores_list, newshape=(-1, 1))
    boxes_list = np.concatenate([boxes_list, scores_list], axis=-1)

    score_thres_ = 0.3
    iou_thres_ = 0.55

    results_ = weighted_boxes_fusion(boxes_list, iou_thres_, score_thres_)
    print(results_)
