from typing import Any
from typing import Mapping
from typing import Tuple

import torch

from superpoint import decoder
from superpoint import net


class SuperPointNetWithDecoder(torch.nn.Module):
    """SuperPoint united with its decoder."""

    def __init__(self):
        super().__init__()
        self.superpoint = net.SuperPointNet()
        self.decoder = decoder.SuperPointDecoder()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forwards x to SuperPoint and returns the decoded output.

        See `SuperPointNet` docs for input requirements and `SuperPointDecoder` docs for
        output description.
        """
        _, _, h, w = x.shape
        enc_keypoints, enc_descriptors = self.superpoint(x)
        return self.decoder(enc_keypoints, enc_descriptors, h, w)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self.superpoint.load_state_dict(state_dict, strict)
