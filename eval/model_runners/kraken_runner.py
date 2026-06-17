import os

from PIL import ImageOps

from .common import CACHE_DIR


def build_kraken_model(model_path):
    from huggingface_hub import hf_hub_download
    from kraken.lib import models

    if os.path.exists(model_path):
        resolved_path = model_path
        print(f"Loading Kraken model from local path: {resolved_path}")
    else:
        print(f"Local Kraken path not found, downloading from Hub: {model_path}")
        resolved_path = hf_hub_download(
            repo_id=model_path,
            filename="real_plus_synth_200_best_250326.mlmodel",
            cache_dir=CACHE_DIR,
        )
        print(f"Downloaded Kraken model to: {resolved_path}")

    return models.load_any(resolved_path)


def run_kraken_inference(kraken_model, base_img):
    from kraken import rpred
    from kraken.containers import Segmentation, BaselineLine

    padded_img = ImageOps.expand(base_img, border=10, fill="white")
    w, h = padded_img.size
    baseline_y = int(h * 0.75)

    line_struct = BaselineLine(
        id="line_0",
        baseline=[(1, baseline_y), (w - 1, baseline_y)],
        boundary=[(1, 1), (w - 1, 1), (w - 1, h - 1), (1, h - 1), (1, 1)],
        text=None,
        base_dir="R",
    )

    seg_container = Segmentation(
        type="baselines",
        imagename="",
        text_direction="horizontal-rl",
        script_detection=False,
        lines=[line_struct],
        regions={},
        line_orders=[],
    )

    pred_iterator = rpred.rpred(kraken_model, padded_img, seg_container, bidi_reordering="L")
    preds = [record.prediction.strip() for record in pred_iterator if getattr(record, "prediction", None)]
    return " ".join(preds).strip()
