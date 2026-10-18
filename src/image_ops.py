import cv2
import numpy as np
import time

def clahe_image(gray, settings):
    # Image only; labels, object positions, and condition metadata are never used.
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError('CLAHE expects native-resolution uint8 grayscale.')
    return cv2.createCLAHE(clipLimit=settings['clipLimit'],
                           tileGridSize=tuple(settings['tileGridSize'])).apply(gray)

def infer(model, records, grayscale, config):
    import torch
    settings, training = config['evaluation'], config['train']
    common = dict(imgsz=training['imgsz'], device=training['device'],
                  conf=settings['confidence_floor'], iou=settings['nms_iou'],
                  max_det=settings['max_det'], rect=False, augment=False,
                  verbose=False, half=False, save=False)
    arrays = [np.repeat(gray[:, :, None], 3, axis=2) for gray in grayscale]
    model.predict(source=arrays[:1], **common)
    if training['device'] != 'cpu': torch.cuda.synchronize(training['device'])
    start = time.perf_counter()
    predictions = []
    for offset in range(0, len(records), training['batch']):
        result = model.predict(source=arrays[offset:offset + training['batch']], **common)
        for record, row in zip(records[offset:offset + training['batch']], result):
            if row.orig_shape != (record['height'], record['width']):
                raise RuntimeError('Prediction coordinates were not restored to native resolution.')
            boxes = np.c_[row.boxes.xyxy.cpu().numpy(), row.boxes.conf.cpu().numpy()]
            predictions.append({'id': record['id'], 'boxes': boxes.tolist()})
    if training['device'] != 'cpu': torch.cuda.synchronize(training['device'])
    elapsed = time.perf_counter() - start
    if model.model.training:
        raise RuntimeError('Detector unexpectedly in training mode.')
    return predictions, {'seconds': elapsed, 'mean_ms_per_image': elapsed * 1000 / len(records),
                         'batch_size': training['batch'],
                         'scope': 'warm, loaded images: input transform + GPU forward + postprocess + CPU transfer; excludes CLAHE and file IO'}
