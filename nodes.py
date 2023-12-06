import torch

import torch.nn.functional as F
from comfy_extras.nodes_post_processing import gaussian_kernel

class LogSigmas:
    """For testing, simply prints the input sigmas"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "sigmas": ("SIGMAS",),
                              }}

    FUNCTION = "log_sigmas"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    CATEGORY = "_for_testing"

    def log_sigmas(self, sigmas):
        print(sigmas)
        return ()

#Blur functions shameless stolen borrowed comfy_extras/nodes_post_processing
#with slight modifications for latent dimensions

def gaussian_blur(latents, kernel, radius=20):
    padded_latents = F.pad(latents, [radius]*4, 'reflect')
    blurred = F.conv2d(padded_latents, kernel, padding=(radius*2+1) // 2, groups=4)
    return blurred[:, :, radius:-radius, radius:-radius]

class SpliceLatents:
    """Performs a fast approximate splice of 2 latents by bluring.
    Intended to eventually automatically calculate blur strength from sigmas"""
    @classmethod
    def INPUT_TYPES(s):
        #These numbers are likely flawed
        return {"required": {"mult": ("FLOAT", {"default": 1.0, "precision": 3,
                                                "step": 0.1, "round": .001}),
                             "size": ("INT", {"default": 4, "min": 1, "step": 1}),
                             "wetness": ("FLOAT", {"default": 1.0, "max": 1,
                                                   "min": 0, "precision": 3,
                                                   "step": 0.1, "round": .01})},
                "optional": {"lower": ("LATENT",),
                             "upper": ("LATENT",)}}

    FUNCTION = "splice_latents"
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "latent/advanced"

    def splice_latents(s, mult, size, wetness=1.0, lower=None, upper=None):
        if lower is None and upper is None:
            raise "lower and upper can't both be none"
        if lower is None:
            lower = torch.zeros_like(upper['samples'])
        else:
            lower = lower['samples']
        if upper is None:
            upper = torch.zeros_like(lower)
        else:
            upper = upper['samples']
        radius = size
        kernel = gaussian_kernel(radius * 2 + 1, mult, device=lower.device).repeat(4,1,1).unsqueeze(1)

        lower_b = gaussian_blur(lower, kernel, radius)
        upper_b = gaussian_blur(upper, kernel, radius)
        upper_e = upper - upper_b
        lower_out = lower_b * wetness + lower * (1 - wetness)
        upper_out = upper_e * wetness + upper * (1 - wetness)

        return ({"samples": lower_out + upper_out},)

class SpliceDenoised:
    """A convenience node to splice latents when both noised and denoised outputs exist"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "noised_latent" : ("LATENT",),
            "denoised_latent" : ("LATENT",),
            "donor_latent" : ("LATENT",),
            }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "splice_denoised"
    CATEGORY = "_for_testing"

    def splice_denoised(self, noised_latent, denoised_latent, donor_latent):
        samples = noised_latent['samples'] - denoised_latent['samples'] + donor_latent['samples']
        return ({"samples": samples},)

NODE_CLASS_MAPPINGS = {
    "LogSigmas": LogSigmas,
    "SpliceLatents": SpliceLatents,
    "SpliceDenoised": SpliceDenoised
}
NODE_DISPLAY_NAME_MAPPINGS = {}
