from src.backend.inference_api_base import InferenceAPI


@InferenceAPI.register("onnx")
@InferenceAPI.register("aliked_onnx")
@InferenceAPI.register("disk_onnx")
@InferenceAPI.register("lightglue_onnx")
class OnnxInferenceAPI(InferenceAPI):
    def __init__(self, logger, model_name, model, config=None):
        if config is None:
            config = {}
        super().__init__(logger, model_name, model, config)

        self._model_name = model_name
        self.output_names = [output_node.name for output_node in self._model_wrapper.get_outputs()]

    def run(self, inputs=None):
        try:
            outputs = self._model_wrapper.run(self.output_names, inputs)
            return outputs

        except Exception as e:
            self._logger.error(f"{self._model_name} inference error: {e}")
