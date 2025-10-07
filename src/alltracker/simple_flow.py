from .alltracker import Net as AllTrackerNet
import torch


class AllTrackerFlow(AllTrackerNet):
    def __init__(self):
        super(AllTrackerFlow, self).__init__(seqlen=16)

    def forward(self, ref, frames):
        # Input: images [B, T, 3, H, W]
        B, T, C, H, W = frames.shape
        self.update_seqlen(T)
        device = frames.device

        # --- Normalize the images like in the forward() function ---
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=device).reshape(1, 1, 3, 1, 1).to(frames.dtype)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=device).reshape(1, 1, 3, 1, 1).to(frames.dtype)
        frames = (frames - mean) / std
        ref = (ref - mean) / std

        # Extract fmap for reference frame (e.g. frame 0)
        fmap_anchor = self.get_fmaps(ref.flatten(0, 1), B, T, sw=None, is_training=False)
        fmap_anchor = fmap_anchor.unflatten(0, (B, T))  # shape: [B, C, H, W]

        # Extract fmaps for target window (e.g. frames 1 to N)
        fmaps2 = self.get_fmaps(frames.flatten(0, 1), B, T, sw=None, is_training=False)
        fmaps2 = fmaps2.unflatten(0, (B, T))  # shape: [B, N, C, H8, W8]

        # --- Prepare zero-initialized visconfs and flows ---
        visconfs8 = torch.zeros((B * T, 2, H // 8, W // 8), dtype=frames.dtype, device=device)
        flows8 = torch.zeros((B * T, 2, H // 8, W // 8), dtype=frames.dtype, device=device)

        # --- Call forward_window() ---
        flow_predictions, _, flows8, visconfs8, _ = self.forward_window(
            fmap_anchor, fmaps2, visconfs8, flows8=flows8, flowfeat=None, iters=5, sw=None, is_training=False
        )

        return flow_predictions
