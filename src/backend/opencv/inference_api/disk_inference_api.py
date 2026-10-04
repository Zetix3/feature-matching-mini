from src.backend.inference_api_base import InferenceAPI
from src.backend.opencv.inference_api.inference_api import OpenCVInferenceAPI

@InferenceAPI.register("disk_opencv")
class DiskOpenCVInferenceAPI(OpenCVInferenceAPI):
    def run(self, img):
        try:
            output_names = ["keypoints", "scores", "descriptors"]
            outputs = self._model.forward(output_names)
            return {'kp': outputs[0], 'des': outputs[1], 'sc': outputs[2]}

        except Exception as e:
            self._logger.error(f"Disk inference error: {e}")
            return {'kp': (), 'des': ()}
