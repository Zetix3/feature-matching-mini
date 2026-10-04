from src.backend.io_adapter_base import IOAdapter

@IOAdapter.register("opencv")
class OpenCVIOAdapter(IOAdapter):
    def __init__(self, model_name, logger=None, config=None):
        if config is None:
            config = {}

        super().__init__(model_name, logger, config)
