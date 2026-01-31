import torch
import torch.nn as nn
from pathlib import Path
from omegaconf import OmegaConf

from src.stage2.raedit_teacher import RAEDiTTeacher  # or RAEDiTGuidance from RAE
# plus any RAE imports you need (RAE stage1, etc.)

class RAEDiTMeshupGuidance(nn.Module):
    def __init__(self, cfg, device='cuda', t_range=[0.02, 0.98]):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # 1) load your RAE stage1 + DiTDH-XL + stats via the code you already tested
        #    essentially copy the logic from test_rae_dit_guidance.py / raedit_teacher.py
        self.teacher = RAEDiTTeacher(...)

    @torch.no_grad()
    def encode_text_2(self, prompt, negative_prompt=None, batch_size=1, return_mask=False):
        """
        MeshUp expects this interface.
        For now, we can map prompts → ImageNet class ids manually and just return an integer tensor.
        """
        # e.g. for your first experiments: assume cfg.imagenet_class_id in config
        class_id = torch.full((batch_size,), self.cfg.class_id, dtype=torch.long, device=self.device)
        return class_id  # later you can do a real mapping or CLIP-based classifier

    def SDS(self, rgb, text_embeds, rgb_as_latents=False, **kwargs):
        """
        rgb: [B,3,H,W] - same as DeepFloydGuidance
        text_embeds: here will just be class_ids we return in encode_text_2
        Returns dict with keys 'loss_sds', 'grad', 'grad_norm'
        """
        # This will internally:
        #   rgb → DINO encoder → z0
        #   sample t
        #   call DiT teacher to get SDS-style loss and gradient wrt rgb
        return self.teacher.SDS_from_rgb(rgb, class_ids=text_embeds)
