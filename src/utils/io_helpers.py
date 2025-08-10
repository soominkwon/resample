import os
from PIL import Image


def _to_uint8(x):
    x = (x.clamp(-1, 1) + 1) * 0.5
    x = (x * 255.0 + 0.5).clamp(0, 255).byte()
    return x


def save_tensor_as_image(x, out_dir, denorm=True):
    os.makedirs(out_dir, exist_ok=True)
    x_u8 = _to_uint8(x)
    B = x_u8.shape[0]
    for i in range(B):
        img = x_u8[i].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img).save(os.path.join(out_dir, f"{i:06d}.png"))
