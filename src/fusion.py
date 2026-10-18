"""One-to-one correspondence and conservative coordinate blending, no GT."""
from .evaluation import iou_matrix
import numpy as np


def blend(a_boxes, partner_boxes, spec, alpha):
    a = np.asarray(a_boxes, float).reshape(-1, 5)
    b = np.asarray(partner_boxes, float).reshape(-1, 5)
    result = a.copy()
    overlaps = iou_matrix(a[:, :4], b[:, :4])
    used, audit = set(), []
    for pi in np.argsort(-a[:, 4], kind='stable'):
        aw = a[pi, 2:4] - a[pi, :2]
        candidates = []
        if a[pi, 4] >= spec['minimum_A_score'] and min(aw) > 0:
            for qi in range(len(b)):
                if qi in used or b[qi, 4] < spec['minimum_partner_score'] or overlaps[pi, qi] < spec['minimum_pair_iou']:
                    continue
                ratio = (b[qi, 2:4] - b[qi, :2]) / aw
                shift = np.abs((b[qi, :2] + b[qi, 2:4] - a[pi, :2] - a[pi, 2:4]) / (2 * aw))
                if np.any(ratio < spec['size_ratio_range'][0]) or np.any(ratio > spec['size_ratio_range'][1]) or max(shift) > spec['maximum_center_shift_fraction']:
                    continue
                candidates.append(qi)
        qi = max(candidates, key=lambda q: (overlaps[pi, q], b[q, 4])) if candidates else None
        if qi is not None:
            used.add(qi)
            result[pi, :4] = a[pi, :4] + alpha * (b[qi, :4] - a[pi, :4])
        audit.append({'prediction_index': int(pi), 'A_confidence': float(a[pi, 4]),
                      'partner_prediction_index': qi, 'pair_iou': float(overlaps[pi, qi]) if qi is not None else None,
                      'changed': bool(np.any(result[pi, :4] != a[pi, :4]))})
    return result, audit


def self_checks(spec):
    a = [[10, 10, 20, 20, .9], [10, 10, 20, 20, .4]]
    b = [[11, 10, 21, 20, .8]]
    c, audit = blend(a, b, spec, .5)
    if not np.array_equal(c[0, :4], [10.5, 10, 20.5, 20]) or not np.array_equal(c[1], a[1]):
        raise RuntimeError('Correspondence/duplicate handling failed')
    if not np.array_equal(c[:, 4], np.asarray(a)[:, 4]):
        raise RuntimeError('Scores changed')
    identity, _ = blend(a, [], spec, .5)
    if not np.array_equal(identity, a):
        raise RuntimeError('Missing partner fallback failed')
    far, _ = blend(a, [[30, 30, 40, 40, .9]], spec, .5)
    if not np.array_equal(far, a):
        raise RuntimeError('Wrong-object rejection failed')
    empty, _ = blend([], b, spec, .5)
    if empty.shape != (0, 5):
        raise RuntimeError('Empty A handling failed')
    return {'coordinate_blend': True, 'one_to_one': True, 'score_count_preserved': True, 'fallback': True, 'empty': True}
