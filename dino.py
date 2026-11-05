"""
Общий модуль DINOv2 для Assignment 2 (используется в 18, 19 и в app.py).

DINOv2 (Oquab et al., 2023, Meta AI) - Vision Transformer, обученный самообучением на 142 млн картинок.
Берём маленькую версию ViT-S/14 (22 млн параметров), чтобы работало на CPU.
Эмбеддинг кадра = [CLS-токен (384) | среднее по патч-токенам (384)] = 768 чисел,
как в линейной оценке из статьи DINOv2.
Карта внимания = внимание CLS-токена к патчам 16x16 в последнем слое, среднее по головам.

Нужно: pip install torch transformers
"""
import numpy as np

MODEL_ID = "facebook/dinov2-small"
SIZE = 224                      # 224 / 14 = 16x16 патчей
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
EMB_DIM = 768


class Dino:
    def __init__(self, device=None, threads=None):
        import torch
        from transformers import AutoModel
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # eager: только так модель отдаёт матрицы внимания (для тепловых карт)
        self.model = AutoModel.from_pretrained(MODEL_ID, attn_implementation="eager").to(self.device).eval()

    @staticmethod
    def prep(rgb_list):
        """rgb_list: картинки RGB uint8 любого размера -> тензор-массив [B,3,224,224]."""
        import cv2
        out = []
        for im in rgb_list:
            if im.shape[0] != SIZE or im.shape[1] != SIZE:
                im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            x = (im.astype(np.float32) / 255.0 - MEAN) / STD
            out.append(x.transpose(2, 0, 1))
        return np.stack(out).astype(np.float32)

    def embed(self, rgb_list, with_attention=False):
        """-> эмбеддинги [B,768] и (если нужно) карты внимания [B,16,16] в диапазоне 0..1."""
        torch = self.torch
        x = torch.from_numpy(self.prep(rgb_list)).to(self.device)
        with torch.inference_mode():
            out = self.model(pixel_values=x, output_attentions=with_attention)
        h = out.last_hidden_state                       # [B, 1+256, 384]
        emb = torch.cat([out.pooler_output, h[:, 1:].mean(1)], dim=1).float().cpu().numpy()
        if not with_attention:
            return emb, None
        att = out.attentions[-1][:, :, 0, 1:].mean(1)   # CLS -> патчи, среднее по головам: [B,256]
        n = int(round(att.shape[1] ** 0.5))
        att = att.reshape(-1, n, n).float().cpu().numpy()
        att = (att - att.min(axis=(1, 2), keepdims=True)) / (np.ptp(att, axis=(1, 2), keepdims=True) + 1e-9)
        return emb, att


def overlay(rgb, att, alpha=0.45):
    """Наложить карту внимания (16x16, 0..1) на картинку RGB."""
    import cv2
    h, w = rgb.shape[:2]
    a = cv2.resize(att.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    a = np.clip(a, 0, 1)
    heat = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return (rgb * (1 - alpha) + heat * alpha).astype(np.uint8)
