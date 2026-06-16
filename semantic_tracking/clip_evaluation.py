
import torch
import clip
from PIL import Image

class CLIPEvaluator:
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def compute_clip_similarity(self, images, text_prompt):
        images_preprocessed = torch.stack([self.preprocess(Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))) for img in images]).to(self.device)
        text = clip.tokenize([text_prompt]).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(images_preprocessed)
            text_features = self.model.encode_text(text)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ text_features.T).mean()
        return similarity.item()
