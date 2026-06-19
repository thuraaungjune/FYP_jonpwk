import os
import io
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import scipy.ndimage as ndimage
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
from tqdm import tqdm  

# DATASET AUGMENTATION / ATTACK FUNCTIONS

def apply_gaussian_noise(image, intensity=0.15):
    img_array = np.array(image).astype(float) / 255.0
    noise = np.random.normal(0, intensity, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 1) * 255
    return Image.fromarray(noisy_img.astype(np.uint8))

def apply_shot_noise(image, intensity=15):
    img_array = np.array(image).astype(float)
    noisy_img = np.random.poisson(img_array * intensity) / intensity
    return Image.fromarray(np.clip(noisy_img, 0, 255).astype(np.uint8))

def apply_impulse_noise(image, amount=0.07):
    img_array = np.array(image).copy()
    h, w, c = img_array.shape
    s_vs_p = 0.5
    num_salt = np.ceil(amount * img_array.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape[:2]]
    img_array[tuple(coords)] = 255
    num_pepper = np.ceil(amount * img_array.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape[:2]]
    img_array[tuple(coords)] = 0
    return Image.fromarray(img_array)

def apply_motion_blur(image, degree=12, angle=45):
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree
    img_array = np.array(image)
    blurred = cv2.filter2D(img_array, -1, motion_blur_kernel)
    return Image.fromarray(blurred)

def apply_defocus_blur(image, radius=4):
    img_array = np.array(image)
    kernel_size = radius * 2 + 1
    blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), 0)
    return Image.fromarray(blurred)

def apply_rotation_skew(image, angle=8):
    return image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor="white")

def apply_perspective_distortion(image, factor=0.12):
    w, h = image.size
    offsets = np.random.uniform(-factor * w, factor * w, size=(4, 2))
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = src + offsets.astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    img_array = np.array(image)
    distorted = cv2.warpPerspective(img_array, M, (w, h), borderValue=(255, 255, 255))
    return Image.fromarray(distorted)

def apply_baseline_drift(image, amplitude=7, frequency=0.05):
    img_array = np.array(image)
    h, w, c = img_array.shape
    new_img = np.full_like(img_array, 255)
    for x in range(w):
        y_offset = int(amplitude * np.sin(2 * np.pi * frequency * x / 10))
        for y in range(h):
            new_y = y + y_offset
            if 0 <= new_y < h:
                new_img[new_y, x] = img_array[y, x]
    return Image.fromarray(new_img)

def apply_over_exposure(image, factor=2.2):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def apply_under_exposure(image, factor=0.3):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

def apply_gradient_illumination(image):
    img_array = np.array(image).astype(float)
    h, w, c = img_array.shape
    gradient = np.linspace(1.0, 0.2, w)
    gradient = np.tile(gradient, (h, 1))
    gradient = np.stack([gradient]*3, axis=-1)
    illuminated = img_array * gradient
    return Image.fromarray(np.clip(illuminated, 0, 255).astype(np.uint8))

def apply_jpeg_artifacts(image, quality=10):
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality)
    output.seek(0)
    return Image.open(output)

def apply_ink_bleed(image, intensity=0.5):
    img_array = np.array(image).astype(float)
    flipped = np.array(image.transpose(Image.FLIP_LEFT_RIGHT)).astype(float)
    bleed = ndimage.gaussian_filter(flipped, sigma=2)
    blended = img_array * (1 - intensity * 0.2) + bleed * (intensity * 0.1)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

def apply_under_segmentation(image, neighbor_image, overlap_px=10):
    w, h = image.size
    neighbor_strip = neighbor_image.crop((0, 0, w, overlap_px))
    combined = Image.new("RGB", (w, h + overlap_px), (255, 255, 255))
    combined.paste(image, (0, 0))
    combined.paste(neighbor_strip, (0, h))
    return combined

def attack_bbox_shift(image, shift_percent=0.12):
    w, h = image.size
    shift_px = int(h * shift_percent)
    return image.crop((0, shift_px, w, h))

def attack_space_injection(image, num_patches=5):
    img_array = np.array(image).copy()
    h, w, _ = img_array.shape
    for _ in range(num_patches):
        x = np.random.randint(0, w-10)
        img_array[:, x:x+5, :] = np.random.randint(0, 255, (h, 5, 3))
    return Image.fromarray(img_array)

def attack_spatial_warp(image, alpha=12, sigma=3):
    image_array = np.array(image)
    shape = image_array.shape
    dx = ndimage.gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), sigma) * alpha
    dy = ndimage.gaussian_filter((np.random.rand(*shape[:2]) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    warped = np.zeros_like(image_array)
    for i in range(3):
        warped[:,:,i] = ndimage.map_coordinates(image_array[:,:,i], indices, order=1).reshape(shape[:2])
    return Image.fromarray(warped)

def attack_fgsm_wide(image, epsilon=0.09):
    img_array = np.array(image).astype(float) / 255.0
    noise = np.sign(np.random.normal(0, 1, img_array.shape))
    adversarial_array = img_array + (epsilon * noise)
    adversarial_array = np.clip(adversarial_array, 0, 1) * 255
    return Image.fromarray(adversarial_array.astype(np.uint8))

def to_pil_image(image_value):
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, str) and os.path.exists(image_value):
        return Image.open(image_value).convert("RGB")
    if isinstance(image_value, dict):
        if image_value.get("path"):
            return Image.open(image_value["path"]).convert("RGB")
        if image_value.get("bytes") is not None:
            return Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")
    if isinstance(image_value, np.ndarray):
        if image_value.ndim == 2:
            return Image.fromarray(image_value).convert("RGB")
        return Image.fromarray(image_value.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image_value)}")


def build_attack_registry():
    return {
        "gaussian_noise": apply_gaussian_noise,
        "motion_blur": apply_motion_blur,
        "rotation_skew": apply_rotation_skew,
        "perspective_distortion": apply_perspective_distortion,
        "under_exposure": apply_under_exposure,
        "ink_bleed": apply_ink_bleed,
        "spatial_warp": attack_spatial_warp,
        "bbox_shift": attack_bbox_shift,
        "space_injection": attack_space_injection,
        "fgsm_wide": attack_fgsm_wide,
    }

# DUAL EXPORT PROCESSING PIPELINE

def dual_export_to_paddle(input_folder, output_folder, splits, selected_attacks):
    """
    Reads extensionless Parquet raw splits, and builds TWO separate 
    PaddleOCR datasets: one clean, and one augmented.
    """
    full_registry = build_attack_registry()
    attack_registry = {k: v for k, v in full_registry.items() if k in selected_attacks}
    
    # Establish dual root output structures
    clean_root = os.path.join(output_folder, "clean")
    aug_root = os.path.join(output_folder, "augmented")
    
    # Vocabulary set to construct global character configuration dictionary
    jawi_character_set = set()

    for split in splits:
        split_path = os.path.join(input_folder, split)
        
        if not os.path.exists(split_path):
            print(f"Skipping split '{split}', file not found at: {split_path}")
            continue

        print(f"\n=== Processing Split: {split} ===")
        try:
            df = pd.read_parquet(split_path)
        except Exception as e:
            print(f"[Error] Failed to read split file '{split}' directly: {str(e)}")
            continue

        df = df[["Image", "Text", "Identifier"]].copy()
        images = [to_pil_image(v) for v in df["Image"].tolist()]
        texts = df["Text"].tolist()
        identifiers = df["Identifier"].tolist()

        # Define specific directories for CLEAN pipeline
        clean_img_dir = os.path.join(clean_root, "rec", split)
        os.makedirs(clean_img_dir, exist_ok=True)
        clean_gt_path = os.path.join(clean_root, "rec", f"rec_gt_{split}.txt")

        # Define specific directories for AUGMENTED pipeline
        aug_img_dir = os.path.join(aug_root, "rec", split)
        os.makedirs(aug_img_dir, exist_ok=True)
        aug_gt_path = os.path.join(aug_root, "rec", f"rec_gt_{split}.txt")

        progress_bar = tqdm(
            zip(images, texts, identifiers), 
            total=len(images), 
            desc=f"└─ Processing {split}", 
            unit="img"
        )
        
        # Open both mapping manifest tracking logs concurrently
        with open(clean_gt_path, "w", encoding="utf-8") as clean_gt, \
             open(aug_gt_path, "w", encoding="utf-8") as aug_gt:
             
            for idx, (image, text, identifier) in enumerate(progress_bar):
                neighbor_image = images[(idx + 1) % len(images)] if len(images) > 1 else image
                clean_text = str(text).replace("\t", " ").replace("\n", " ")

                # Collect text alphabet maps for dictionary compilation
                if split == "train":
                    for char in clean_text:
                        jawi_character_set.add(char)

                # -----------------------------------------------------
                # PIPELINE 1: Write to Clean Output Tree
                # -----------------------------------------------------
                clean_filename = f"{identifier}_clean.jpg"
                clean_save_path = os.path.join(clean_img_dir, clean_filename)
                image.save(clean_save_path, "JPEG")
                
                # Write to clean manifest using exact singular tab separator (\t)
                clean_gt.write(f"{split}/{clean_filename}\t{clean_text}\n")

                # -----------------------------------------------------
                # PIPELINE 2: Write to Augmented Output Tree
                # -----------------------------------------------------
                # First, save baseline clean copy inside augmented folder too
                aug_gt.write(f"{split}/{clean_filename}\t{clean_text}\n")
                image.save(os.path.join(aug_img_dir, clean_filename), "JPEG")

                # Loop through custom geometric transformations/noises
                for attack_name, attack_fn in attack_registry.items():
                    try:
                        if attack_name == "under_segmentation":
                            attacked_image = attack_fn(image, neighbor_image)
                        else:
                            attacked_image = attack_fn(image)

                        aug_filename = f"{identifier}_{attack_name}.jpg"
                        aug_save_path = os.path.join(aug_img_dir, aug_filename)
                        
                        # Write altered variant out to augmented structures
                        attacked_image.convert("RGB").save(aug_save_path, "JPEG")
                        aug_gt.write(f"{split}/{aug_filename}\t{clean_text}\n")

                    except Exception as e:
                        print(f"\n[Warning] Attack '{attack_name}' failed on ID {identifier}: {str(e)}")
                        continue

    # Automatically write unified character vocab dictionaries inside both outputs
    if jawi_character_set:
        sorted_chars = sorted(list(jawi_character_set))
        for target_root in [clean_root, aug_root]:
            dict_path = os.path.join(target_root, "jawi_dict.txt")
            with open(dict_path, "w", encoding="utf-8") as dict_file:
                for char in sorted_chars:
                    if char.strip():
                        dict_file.write(f"{char}\n")

    print(f"\nDual Generation Complete!")
    print(f" -> Clean Paddle OCR Dataset: {clean_root}")
    print(f" -> Augmented Paddle OCR Dataset: {aug_root}")


# COMMAND LINE INTERACTION ENGINE

if __name__ == "__main__":
    default_attacks = list(build_attack_registry().keys())

    parser = argparse.ArgumentParser(description="Split raw monolithic parquet data into Clean and Augmented PaddleOCR datasets.")
    
    parser.add_argument("--input", type=str, default="FYP_jonpwk/Jawi-OCR-data-v4",
                        help="Path to unaugmented source folder containing monolithic splits")
    parser.add_argument("--output", type=str, default="../paddle_jawi_outputs",
                        help="Root path where clean/ and augmented/ datasets will be saved")
    parser.add_argument("--splits", nargs="+", default=["train", "test", "validation"],
                        help="Dataset splits to extract (default: train, test, validation)")
    parser.add_argument("--attacks", nargs="+", default=default_attacks, choices=default_attacks,
                        help="Target attack/augmentation variations to apply onto the augmented output")

    args = parser.parse_args()

    print("\n--- Dual Export Configuration Summary ---")
    print(f" Source Location      : {args.input}")
    print(f" Output Parent Base   : {args.output}")
    print(f" Target Process Splits: {args.splits}")
    print(f" Variations Computed  : {args.attacks}\n")

    dual_export_to_paddle(
        input_folder=args.input,
        output_folder=args.output,
        splits=args.splits,
        selected_attacks=args.attacks
    )
