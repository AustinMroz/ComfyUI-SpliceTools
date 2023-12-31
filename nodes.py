import torch

import torch.nn.functional as F
from comfy.sample import prepare_mask
from comfy.k_diffusion import sampling

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

class RerangeSigmas:
    """Given a set of input sigmas, produce a new set of sigmas that cover the same range"""
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"sigmas": ("SIGMAS",),
                             "steps": ("INT", {"default": 10, "min": 1})}}
    FUNCTION = "rerange_sigmas"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"
    def rerange_sigmas(self, sigmas, steps):
        assert(len(sigmas)>1)
        (s_max, s_min) = (sigmas[0], sigmas[-1])
        full_denoise = False
        if s_min == 0:
            assert(len(sigmas)>2)
            full_denoise = True
            s_min = sigmas[-2]
        else:
            steps+=1

        #TODO: Implement scheduling method with a more uniform distribution
        sigmas = sampling.get_sigmas_exponential(n=steps, sigma_min=s_min, sigma_max=s_max)
        if not full_denoise:
            sigmas = sigmas[:-1]
        return (sigmas,)


#Blur functions shameless borrowed from comfy_extras/nodes_post_processing
#with slight modifications for latent dimensions
def gaussian_kernel(sigma, size=5):
    maxl = size // 2
    x, y = torch.meshgrid(torch.linspace(-maxl, maxl, size), torch.linspace(-maxl, maxl, size), indexing="ij")
    d = (x * x + y * y) / (2 * sigma * sigma)
    mask =  torch.exp(-d) / (2 * torch.pi * sigma * sigma)
    return mask / mask.sum()




def gaussian_blur(latents, kernel, radius=5):
    padded_latents = F.pad(latents, [radius]*4, 'reflect')
    blurred = F.conv2d(padded_latents, kernel, padding=(radius*2+1) // 2, groups=4)
    return blurred[:, :, radius:-radius, radius:-radius]

class SpliceLatents:
    """Performs a fast approximate splice of 2 latents by bluring."""
    @classmethod
    def INPUT_TYPES(s):
        #These numbers are likely flawed
        return {"required": {"sigmas": ("SIGMAS",),
                             "radius": ("INT", {"default": 4, "min": 1, "step": 1}),
                             "wetness": ("FLOAT", {"default": 1.0, "max": 1,
                                                   "min": 0, "precision": 3,
                                                   "step": 0.1, "round": .01}),
                             "texture_override": (["None", "Upper", "Lower"],)},
                "optional": {"lower": ("LATENT",),
                             "upper": ("LATENT",)}}

    FUNCTION = "splice_latents"
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "latent/advanced"

    def splice_latents(s, sigmas, radius, texture_override, wetness=1.0, lower=None, upper=None):
        #TODO: Find solution to prevent errors when nodes are muted in workflow
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
        length = radius * 2 + 1
        #for 1.5, channel 3 is texture. Its feasible to
        #create a kernel that blurs channels [0,1,3] and zeros 2
        #this conceptually delegates sub-pixel texture to upper always,
        #but is less viable for mixed configuration
        #further experimentation is needed
        mask = gaussian_kernel(sigmas[-1], length)
        kernel = torch.stack((mask, mask, mask, mask)).unsqueeze(1)

        lower_b = gaussian_blur(lower, kernel, radius)
        upper_b = gaussian_blur(upper, kernel, radius)
        upper_e = upper - upper_b
        lower_out = lower_b * wetness + lower * (1 - wetness)
        upper_out = upper_e * wetness + upper * (1 - wetness)
        out = lower_out + upper_out
        if texture_override == "Upper":
            out[:,2] = upper[:,2]
        elif texture_override == "Lower":
            out[:,2] = lower[:,2]

        return ({"samples": out},)

class TemporalSplice:
    """Areas of low movement are passed from lower"""
    @classmethod
    def INPUT_TYPES(s):
        #These numbers are likely flawed
        return {"required": {"sigma": ("FLOAT", {"default": 1.0, "step": .01, "min": 0}),
                             "wetness": ("FLOAT", {"default": 1.0, "max": 1,
                                                   "min": 0, "precision": 3,
                                                   "step": 0.1, "round": .01}),},
                "optional": {"lower": ("LATENT",),
                             "upper": ("LATENT",)}}
    FUNCTION = "temporal_splice"
    RETURN_TYPES = ("LATENT",)
    CATEGORY = "latent/advanced"
    def temporal_splice(s, sigma, wetness, lower=None, upper=None):
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
        #Ensure odd with no overlap
        length = max(lower.shape[0], upper.shape[0])
        radius = (length + 1) // 2
        t = torch.linspace(-radius, radius, 2 * radius + 1)
        d = torch.exp(-t*t/(2*sigma*sigma))
        mask = d / d.sum()
        kernel = torch.stack([mask] * 4).unsqueeze(1)
        def temporal_blur(tensor, kernel, radius):
            #latent is B C H W, but HW C B is desired
            tensor = tensor.permute((3,2,1,0))
            shape = tensor.shape
            tensor = tensor.reshape((shape[0]*shape[1],shape[2],shape[3]))
            tensor = F.pad(tensor, [radius]*2, 'circular')
            tensor = F.conv1d(tensor, kernel, padding=(radius*2+1) // 2, groups=4)
            tensor = tensor[:,:,radius:-radius]

            #Test code to force full blur
            #tensor = tensor.mean(dim=3).unsqueeze(3).repeat(1,1,1,shape[3])

            tensor = tensor.reshape(shape)
            tensor = tensor.permute((3,2,1,0))
            return tensor
        if len(lower) == 1:
            lower_b = lower
        else:
            if len(lower) < length //2:
                lower = lower.repeat(2,1,1,1)
            lower_b = temporal_blur(lower, kernel, radius)
        if len(upper) == 1:
            upper_b = upper
        else:
            if len(upper) < radius:
                upper = upper.repeat(2,1,1,1)
            upper_b = temporal_blur(upper, kernel, radius)
        upper_e = upper - upper_b
        lower_out = lower_b * wetness + lower * (1 - wetness)
        upper_out = upper_e * wetness + upper * (1 - wetness)
        out = lower_out + upper_out
        #TODO: copy other items (mask,batch) from inputs? (also splice_latents)
        return ({"samples": out},)

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
    CATEGORY = "latent/advanced"

    def splice_denoised(self, noised_latent, denoised_latent, donor_latent):
        #Partial mask support for donor latent
        donor_samples = donor_latent['samples']
        if 'noise_mask' in donor_latent:
            mask = prepare_mask(donor_latent['noise_mask'], donor_samples.shape, 'cpu')
            donor_samples = donor_samples * (1 - mask)
            donor_samples = donor_samples + denoised_latent['samples'] * mask
        samples = noised_latent['samples'] - denoised_latent['samples'] + donor_samples
        out = noised_latent.copy()
        out['samples'] = samples
        return (out,)

NODE_CLASS_MAPPINGS = {
    "LogSigmas": LogSigmas,
    "SpliceLatents": SpliceLatents,
    "SpliceDenoised": SpliceDenoised,
    "TemporalSplice": TemporalSplice,
    "RerangeSigmas": RerangeSigmas
}
NODE_DISPLAY_NAME_MAPPINGS = {}
