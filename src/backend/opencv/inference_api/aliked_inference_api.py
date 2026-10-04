from src.backend.inference_api_base import InferenceAPI
from src.backend.opencv.inference_api.inference_api import OpenCVInferenceAPI

@InferenceAPI.register("aliked_opencv")
class AlikedOpenCVInferenceAPI(OpenCVInferenceAPI):
    def run(self, img):
        try:
            output_names = ["keypoints", "descriptors", "scores"]
            outputs = self._model.forward(output_names)
            return {'kp': outputs[0], 'des': outputs[1], 'sc': outputs[2]}

        except Exception as e:
            self._logger.error(f"Aliked inference error: {e}")
            return {'kp': (), 'des': ()}
