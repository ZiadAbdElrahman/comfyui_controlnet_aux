from ..utils import common_annotator_call, create_node_input_types
import comfy.model_management as model_management

class DensePose_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return create_node_input_types(
            model=(["densepose_r50_fpn_dl.torchscript", "densepose_r101_fpn_dl.torchscript"], {"default": "densepose_r50_fpn_dl.torchscript"}),
            cmap=(["Viridis (MagicAnimate)", "Parula (CivitAI)"], {"default": "Viridis (MagicAnimate)"}),
            cache_model = (["enable", "disable"], {"default": "enable"}),

        )

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    model = None

    CATEGORY = "ControlNet Preprocessors/Faces and Poses Estimators"

    def execute(self, image, model, cmap, cache_model, resolution=512):
        cache_model = cache_model == "enable"

        from controlnet_aux.densepose import DenseposeDetector
        if self.model is None:
            self.model = DenseposeDetector \
                        .from_pretrained(filename=model) \
                        .to(model_management.get_torch_device())
        output = common_annotator_call(self.model, image, cmap="viridis" if "Viridis" in cmap else "parula", resolution=resolution) 
        
        if not cache_model:
            self.model = None
            
        return (output, )


NODE_CLASS_MAPPINGS = {
    "DensePosePreprocessor": DensePose_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DensePosePreprocessor": "DensePose Estimator"
}
