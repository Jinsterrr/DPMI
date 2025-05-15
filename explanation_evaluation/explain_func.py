import numpy as np
import torch
from typing import Optional, Union
import warnings
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric

from captum.attr import (
    GradientShap,
    IntegratedGradients,
    InputXGradient,
    Saliency,
    Occlusion,
    FeatureAblation,
    LayerGradCam,
    DeepLift,
    DeepLiftShap,
    GuidedGradCam,
    Deconvolution,
    FeaturePermutation,
    Lime,
    KernelShap,
    LRP,
    LayerConductance,
    LayerActivation,
    InternalInfluence,
    LayerGradientXActivation,
)

from explain_methods import FovEx
from explain_methods.FovEx.classIdx import CLS2IDX
from explain_methods.GradCamPlusScore import GradCamPlusScore


def explain(model: torch.nn.Module, 
            inputs: TensorOrTupleOfTensorsGeneric, 
            targets: TargetType,
            method: str = "Saliency",
            device: Optional[str] = None,
            **kwargs) -> np.ndarray:
    

    reduce_axes = kwargs.get("reduce_axes", None)   # 是否需要在特定轴上求和
    if reduce_axes is None:
        def f_reduce_axes(a):
            return a
    else:
        def f_reduce_axes(a):
            return a.sum(dim=reduce_axes)
    
    xai_lib_kwargs = kwargs.get("xai_lib_kwargs", {}) # 调用可解释性方法实例需要传入的参数

    # 具体的实现
    if method in ["GradientShap", "DeepLift", "DeepLiftShap"]:
        baselines = (
            kwargs["baseline"] if "baseline" in kwargs else torch.zeros_like(inputs)
        )
        attr_func = eval(method)
        explanation = f_reduce_axes(
            attr_func(model, **xai_lib_kwargs).attribute(
                inputs=inputs,
                target=targets,
                baselines=baselines,
            )
        )

    elif method == "IntegratedGradients":
        baselines = (
            kwargs["baseline"] if "baseline" in kwargs else torch.zeros_like(inputs)
        )
        attr_func = eval(method)
        explanation = f_reduce_axes(
            attr_func(model, **xai_lib_kwargs).attribute(
                inputs=inputs,
                target=targets,
                baselines=baselines,
                n_steps=10,
                method="riemann_trapezoid",
            )
        )

    elif method in [
        "InputXGradient",
        "Saliency",
        "FeatureAblation",
        "Deconvolution",
        "FeaturePermutation",
        "Lime",
        "KernelShap",
        "LRP",
    ]:
        attr_func = eval(method)
        explanation = f_reduce_axes(
            attr_func(model, **xai_lib_kwargs).attribute(inputs=inputs, target=targets)
        )

    elif method == "Gradient":
        explanation = f_reduce_axes(
            Saliency(model, **xai_lib_kwargs).attribute(
                inputs=inputs, target=targets, abs=False
            )
        )

    elif method == "Occlusion":
        window_shape = kwargs.get("window", (1, *([4] * (inputs.ndim - 2))))
        explanation = f_reduce_axes(
            Occlusion(model).attribute(
                inputs=inputs,
                target=targets,
                sliding_window_shapes=window_shape,
            )
        )

    elif method in [
        "LayerGradCam",
        "GuidedGradCam",
        "LayerConductance",
        "LayerActivation",
        "InternalInfluence",
        "LayerGradientXActivation",
        # "GradCamPlusScore",
    ]:
        # 指定层
        if "gc_layer" in kwargs:
            xai_lib_kwargs["layer"] = kwargs["gc_layer"]

        if "layer" not in xai_lib_kwargs:
            raise ValueError(
                "Specify a convolutional layer name as 'gc_layer' to run GradCam."
            )

        if isinstance(xai_lib_kwargs["layer"], str):
            xai_lib_kwargs["layer"] = eval(xai_lib_kwargs["layer"])

        attr_func = eval(method)

        if method != "LayerActivation":
            explanation = attr_func(model, **xai_lib_kwargs).attribute(
                inputs=inputs, target=targets
            )
        else:
            explanation = attr_func(model, **xai_lib_kwargs).attribute(inputs=inputs)

        if "interpolate" in kwargs:
            if isinstance(kwargs["interpolate"], tuple):
                if "interpolate_mode" in kwargs:
                    explanation = LayerGradCam.interpolate(
                        explanation,
                        kwargs["interpolate"],
                        interpolate_mode=kwargs["interpolate_mode"],
                    )
                else:
                    explanation = LayerGradCam.interpolate(
                        explanation, kwargs["interpolate"]
                    )
        else:
            if explanation.shape[-1] != inputs.shape[-1]:
                warnings.warn(
                    "Quantus requires GradCam attribution and input to correspond in "
                    "last dimensions, but got shapes {} and {}\n "
                    "Pass 'interpolate' argument to explanation function get matching dimensions.".format(
                        explanation.shape, inputs.shape
                    ),
                    category=UserWarning,
                )

        explanation = f_reduce_axes(explanation)

    elif method == "FovEx":
        # 初始化FovEx
        SEED = 42
        # FovEx Paramters
        lr = 0.1
        blur_sigma = 10 
        forgetting = 0.1
        image_size = 32
        scanpath_length = 5
        blur_filter_size = 11
        foveation_sigma = 0.05
        random_restart = True
        optimization_steps = 10
        heatmap_sigma=0.1

        heatmap_forgetting = scanpath_length*[1.0]
        normalize_heatmap=True
        criterion = torch.nn.CrossEntropyLoss(reduction="none")
        def target_function(x, y):
            return y
        fovex = FovEx.FovExWrapper(downstream_model=model,
                    criterion=criterion,
                    target_function=target_function,
                    image_size=image_size,
                    foveation_sigma=foveation_sigma,
                    blur_filter_size=blur_filter_size,
                    blur_sigma=blur_sigma,
                    forgetting=forgetting,
                    foveation_aggregation=1,
                    heatmap_sigma=heatmap_sigma,
                    heatmap_forgetting=heatmap_forgetting,
                    device=device
                    )

        # 生成解释

        # explanation, _, _, _ = fovex.generate_explanation(
        #     inputs.to(device),
        #     targets.to(device),
        #     scanpath_length,
        #     optimization_steps,
        #     lr,
        #     random_restart,
        #     normalize_heatmap
        # )
        def batch_fovex_generate_explanation(inputs, targets, scanpath_length, optimization_steps, lr, random_restart, normalize_heatmap, device):
            batch_size = inputs.size(0)
            all_explanations = []

            for i in range(batch_size):
                single_input = inputs[i].unsqueeze(0)  # [1, 3, 32, 32]
                single_target = targets[i].unsqueeze(0)  # [1]

                explanation, _, _, _ = fovex.generate_explanation(
                    single_input.to(device),
                    single_target.to(device),
                    scanpath_length,
                    optimization_steps,
                    lr,
                    random_restart,
                    normalize_heatmap
                )

                all_explanations.append(explanation)

            # 合并所有解释
            combined_explanation = torch.cat(all_explanations, dim=0)

            return combined_explanation
        
        explanation = batch_fovex_generate_explanation(
        inputs.to(device),
        targets.to(device),
        scanpath_length,
        optimization_steps,
        lr,
        random_restart,
        normalize_heatmap,
        device
        )
        
    elif method == "GradCamPlusScore":
        if "gc_layer" in kwargs:
            target_layer = kwargs["gc_layer"]
        else:
            raise ValueError(
                "Specify a convolutional layer name as 'gc_layer' to run GradCamPlusScore."
            )

        if isinstance(target_layer, str):
            target_layer = eval(target_layer)

        gcps = GradCamPlusScore(model, target_layer)
        explanation = gcps.attribute(inputs=inputs, target=targets)

        if "interpolate" in kwargs:
            if isinstance(kwargs["interpolate"], tuple):
                # if "interpolate_mode" in kwargs:
                #     explanation = gcps.interpolate_cam(
                #         explanation,
                #         kwargs["interpolate"],
                #         mode=kwargs["interpolate_mode"],
                #     )
                # else:
                explanation = gcps.interpolate_cam(
                    explanation, kwargs["interpolate"]
                )
        else:
            if explanation.shape[-2:] != inputs.shape[-2:]:
                warnings.warn(
                    "GradCamPlusScore attribution and input have different spatial dimensions: "
                    f"{explanation.shape[-2:]} vs {inputs.shape[-2:]}\n"
                    "Pass 'interpolate' argument to explanation function to match dimensions.",
                    category=UserWarning,
                )

        explanation = f_reduce_axes(explanation)
    # elif method == "Control Var. Sobel Filter":
    #     explanation = torch.zeros(size=inputs.shape)

    #     for i in range(len(explanation)):
    #         explanation[i] = torch.Tensor(
    #             np.clip(scipy.ndimage.sobel(inputs[i].cpu().numpy()), 0, 1)
    #         )
    #     explanation = explanation.mean(**reduce_axes)

    # elif method == "Control Var. Random Uniform":
    #     explanation = torch.rand(size=(inputs.shape[0], *inputs.shape[2:]))

    # elif method == "Control Var. Constant":
    #     assert (
    #         "constant_value" in kwargs
    #     ), "Specify a 'constant_value' e.g., 0.0 or 'black' for pixel replacement."

    #     explanation = torch.zeros(size=inputs.shape)

    #     # Update the tensor with values per input x.
    #     for i in range(explanation.shape[0]):
    #         constant_value = get_baseline_value(
    #             value=kwargs["constant_value"], arr=inputs[i], return_shape=(1,)
    #         )[0]
    #         explanation[i] = torch.Tensor().new_full(
    #             size=explanation[0].shape, fill_value=constant_value
    #         )

    #     explanation = explanation.mean(**reduce_axes)

    else:
        raise KeyError(
            f"The selected {method} XAI method is not in the list of supported built-in Quantus XAI methods for Captum. "
            f"Please choose an XAI method that has already been implemented."
        )

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            explanation = explanation.cpu().detach().numpy()
        else:
            explanation = explanation.cpu().numpy()

    return explanation
