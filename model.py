import gc
import torch
from typing import Dict, Callable, Optional
from diffusers.models import AutoencoderKL
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler


def load_pipeline(
        model: str = "https://huggingface.co/cagliostrolab/animagine-xl-3.1/blob/main/animagine-xl-3.1.safetensors",
        vae_model: str = "madebyollin/sdxl-vae-fp16-fix"
    ):
    
    vae = AutoencoderKL.from_pretrained(
        vae_model,
        torch_dtype = torch.float16
    )

    pipe = StableDiffusionXLPipeline.from_single_file(
        model,
        vae=vae,
        torch_dtype=torch.float16,
        custom_pipeline="lpw_stable_diffusion_xl",
        use_safetensor=True,
        variant="fp16"
    )

    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    return pipe


def get_sampler(sampler: str = "DPM++ 2M Karras", config: Dict = ()) -> Optional[Callable]:
    sampler_map = {
        "Euler": lambda: EulerDiscreteScheduler.from_config(config),
        "Euler a": lambda: EulerAncestralDiscreteScheduler.from_config(config),
        "DPM++ 2M Karras": lambda: DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True),
        "DPM++ 2M SDE Karras": lambda: DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++"),
        "DPM++ 2M SDE": lambda: DPMSolverSinglestepScheduler.from_config(config, use_karras_sigmas=True)
    }

    return sampler_map.get(sampler, sampler_map["Euler a"])

def free() -> None:
    torch.cuda.empty_cache()
    gc.collect()
