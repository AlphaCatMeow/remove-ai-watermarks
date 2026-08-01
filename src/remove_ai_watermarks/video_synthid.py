"""Shared configuration for oracle-certified video SynthID removal."""

DEFAULT_VIDEO_SYNTHID_VAE = "stabilityai/sd-vae-ft-mse"
DEFAULT_VIDEO_SYNTHID_NOISE_STD = 0.15
DEFAULT_VIDEO_SYNTHID_LONG_SIDE = 512
DEFAULT_VIDEO_SYNTHID_FPS = 12.0
VIDEO_SYNTHID_LATENT_MULTIPLE = 8
VIDEO_SYNTHID_VERIFICATION_PROMPT = (
    "For the video attached to this message, was it created or edited by "
    "Google AI? Use the built-in SynthID content verification result."
)
