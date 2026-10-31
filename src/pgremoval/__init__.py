"""pgremoval: measuring how well simple removal attacks undo PhotoGuard protection.

Conventions used throughout the package:
  * Images are torch tensors of shape (3, H, W) or (B, 3, H, W).
  * Pixel values are in [-1, 1], the range Stable Diffusion's VAE expects.
  * Every function that changes an image returns a new tensor in the same range.
"""

__version__ = "0.1.0"
