from src.backend.inference_api_base import InferenceAPI


@InferenceAPI.register("opencv")
@InferenceAPI.register("aliked_opencv")
@InferenceAPI.register("disk_opencv")
class OpenCVInferenceAPI(InferenceAPI):
    _OUTPUT_NAMES = {
        'aliked_opencv': ["keypoints", "descriptors", "scores"],
        'disk_opencv': ["keypoints", "descriptors", "scores"]
    }

    def __init__(self, logger, model_name, model, config=None):
        if config is None:
            config = {}
        super().__init__(logger, model_name, model, config)

        self._model_name = model_name
        self.output_names = self._get_out_names(model_name)
        self._mode = config.get('mode', 'simple')

    def _get_out_names(self, model_name):
        output_names = self._OUTPUT_NAMES.get(model_name)
        return output_names

    def run(self, inputs):
        try:
            outputs = self._model.forward(self.output_names)
            return outputs

        except Exception as e:
            self._logger.error(f"{self._model_name} inference error: {e}")
