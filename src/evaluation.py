import numpy as np

def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    aa = np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)
    bb = np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)
    return intersection / np.maximum(aa[:, None] + bb[None, :] - intersection, 1e-12)

def matches(records, predictions, match_iou):
    """Descending-confidence, one-to-one matching; prefixes define thresholds."""
    if [r['id'] for r in records] != [p['id'] for p in predictions]:
        raise RuntimeError('Prediction/label IDs differ.')
    tp_scores, fp_scores, gt_matches = [], [], []
    for record, prediction in zip(records, predictions):
        boxes = np.asarray(prediction['boxes'], float).reshape(-1, 5)
        order = np.argsort(-boxes[:, 4], kind='stable')
        boxes = boxes[order]
        overlaps = iou_matrix(boxes[:, :4], record['boxes_xyxy'])
        used = set()
        scores = [None] * len(record['boxes_xyxy'])
        for row, box in enumerate(boxes):
            candidates = [j for j in range(len(scores)) if j not in used and overlaps[row, j] >= match_iou]
            if candidates:
                chosen = max(candidates, key=lambda j: overlaps[row, j])
                used.add(chosen)
                scores[chosen] = float(box[4])
                tp_scores.append(float(box[4]))
            else:
                fp_scores.append(float(box[4]))
        gt_matches.append({'id': record['id'], 'scores': scores})
    return np.asarray(tp_scores), np.asarray(fp_scores), gt_matches

def counts(tp_scores, fp_scores, objects, images, threshold):
    tp = int(np.count_nonzero(tp_scores >= threshold))
    fp = int(np.count_nonzero(fp_scores >= threshold))
    return {'TP': tp, 'FP': fp, 'FN': objects - tp,
            'recall': tp / objects if objects else None,
            'precision': tp / (tp + fp) if tp + fp else None,
            'FP_per_image': fp / images, 'images': images, 'objects': objects}

def condition_recall(records, gt_matches, cutoffs, threshold):
    values = {name: {'objects': 0, 'TP': 0} for name in
              ['small', 'low_contrast', 'near_product_edge_proxy', 'marker_overlap', 'marker_no_overlap']}
    for record, matched in zip(records, gt_matches):
        for obj, score in zip(record['objects'], matched['scores']):
            active = []
            if obj['area_px2'] <= cutoffs['small_area_px2_max'] + 1e-6:
                active.append('small')
            if obj['cnr_valid'] and obj['cnr_proxy'] <= cutoffs['low_contrast_cnr_max']:
                active.append('low_contrast')
            if obj['product_edge_distance_proxy_px'] <= cutoffs['near_product_edge_proxy_px_max']:
                active.append('near_product_edge_proxy')
            active.append('marker_overlap' if obj['marker_overlap_fraction'] > 0 else 'marker_no_overlap')
            for name in active:
                values[name]['objects'] += 1
                values[name]['TP'] += int(score is not None and score >= threshold)
    for row in values.values():
        row['FN'] = row['objects'] - row['TP']
        row['recall'] = row['TP'] / row['objects'] if row['objects'] else None
    return values
