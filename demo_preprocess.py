"""Vẽ hình các bước tiền xử lý cho mọi ảnh trong sample_images/ -> outputs/preprocess_steps/"""
from data_utils import ROOT
from preprocess import preprocess_image, save_steps_figure

src = ROOT / "sample_images"
out = ROOT / "outputs" / "preprocess_steps"
out.mkdir(parents=True, exist_ok=True)
for p in sorted(src.iterdir()):
    if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        _, steps = preprocess_image(p, return_steps=True)
        save_steps_figure(steps, out / f"{p.stem}_steps.png", p.name)
        print("saved", p.stem)
