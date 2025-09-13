"""
Wan
This provides basic compatibility f
"""


from diffusers.models import Autoencode
import torch
import logging

logger = logging.getLogger(__name__)

clas):
    """Compatibility wrapper for WanPipe
    
    def __init__(self, *arargs):
        # Use standard DiffusionPipeline as base
    nit__()
        logger.ir")
    
    @classmethod
    def from
        """Load with fallback to standard pipeline"""
        try:
            # Try to load as standard diffusion pipeline
            from diffusers import StableDiffusine
            logger.info(f"Loading {pret")
            return Stabl
              
                trust_remote_cTrue,
                **kwargs
            )
s e:
            logger.error(f"WanPipeline {e}")
            raise e

class Autoencode:
    """Compatibility wrapper for AutoencoderKLWan"""
    
    @classme
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load with fallback "
        try:
            logger.lback")
)
        except Exception as e:
            logg")
            raise e

asses
try:
")edy layer loadbilitcompati("Wan2.2 
print{e}")
sses: ty clapatibili comer Wan2.2 to regist"Failedror(fer.ergg:
    loas eion ceptt Exxcep
eully")cessfed sucr loadyeity la2 compatibil"Wan2.ogger.info(
    lanKLWncoder = AutoeencoderKLWanusers.Autoiffine
    dPipele = Wanpelins.WanPifuser
    difsersport diffu    im
