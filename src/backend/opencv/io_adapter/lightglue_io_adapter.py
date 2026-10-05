import cv2 as cv
import numpy as np
from src.backend.io_adapter_base import IOAdapter
from src.backend.opencv.io_adapter.io_adapter import OpenCVIOAdapter


@IOAdapter.register("lightglue_opencv")
class LightGlueOpenCVIOAdapter(OpenCVIOAdapter):
    def preprocess(self, inputs):
        features1, features2 = inputs.get('features1'), inputs.get('features2')
        kpts0 = features1.get('kp')
        kpts1 = features2.get('kp')
        desc0 = features1.get('des')
        desc1 = features2.get('des')

        if kpts0.ndim == 2:
            kpts0 = np.expand_dims(kpts0, axis=0)
        if kpts1.ndim == 2:
            kpts1 = np.expand_dims(kpts1, axis=0)
        if desc0.ndim == 2:
            desc0 = np.expand_dims(desc0, axis=0)
        if desc1.ndim == 2:
            desc1 = np.expand_dims(desc1, axis=0)

        input_layers = {
            'kpts0': kpts0.astype(np.float32),
            'kpts1': kpts1.astype(np.float32),
            'desc0': desc0.astype(np.float32),
            'desc1': desc1.astype(np.float32)
        }
        for name, value in input_layers.items():
            self._model.setInput(value, name=name)

    def postprocess(self, outputs):
        matches = np.squeeze(outputs[0], axis=0)
        scores = np.squeeze(outputs[1], axis=0)
        result = {'matches': matches, 'scores': scores}
        return result
