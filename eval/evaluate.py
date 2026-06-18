import os
import io
import json
import argparse
import random
import re

import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from sacrebleu.metrics import CHRF
from jiwer import wer
from datasets import load_from_disk

from model_runners import (
    clear_model_load_cache,
    calculate_cer_components,
    clean_ocr_text,
    build_kraken_model,
    build_deepseek_model,
    build_gemma_model,
    build_qwen_model,
    prepare_gemma_inputs,
    prepare_standard_inputs,
    generate_gemma_text,
    generate_standard_text,
    run_kraken_inference,
    run_deepseek_inference,
    run_deepseek_batch,
)
from model_runners.common import CACHE_DIR

# Regex pattern to match standard Arabic Harakat, Tanween, Sukun, and Tatweel/Kashida lines
DIACRITICS_PATTERN = re.compile(r'[\u064b-\u0652\u0640\u0670]')

def remove_diacritics(text: str) -> str:
    """Helper to explicitly strip diacritics for the second metric pathway."""
    if not text:
        return ""
    return DIACRITICS_PATTERN.sub('', text)

# =====================================================================
# Main Execution Runner
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Zero-Shot VLM and Kraken Robustness Evaluation")
    parser.add_argument("--model", type=str, required=True, help="HF model hub identifier or local path")
    parser.add_argument("--prompt", type=str, required=True, help="Path to text file containing execution prompt context")
    parser.add_argument("--dataset_path", type=str, default="/home/thura/data/Jawi-OCR-data-v4-aug2/test_part_1", help="Path to local Hugging Face dataset folder split")
    parser.add_argument("--clean", action="store_true", help="If enabled, isolates and evaluates only rows matching attack_type == 'clean'")
    parser.add_argument("--sample", type=int, default=None, help="Number of records to randomly subset for testing")
    parser.add_argument("--kraken", action="store_true", help="Set this flag if evaluating Kraken natively")
    parser.add_argument("--deepseek", action="store_true", help="Enforces direct initialization blocks for DeepSeek-OCR")
    
    # Mini-batch configuration option
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size slice (Recommended: 16-64 for A40)")

    # Explicit Architecture Selection Flags
    parser.add_argument("--jawi_qwen_v1", action="store_true", help="Explicit override routing for custom Jawi-OCR-Qwen-v1")
    parser.add_argument("--jawi_qwen_v2", action="store_true", help="Explicit override routing for custom Jawi-OCR-Qwen-v2")
    args = parser.parse_args()

    clean_model_name = args.model.split("/")[-1]
    subset_name = os.path.basename(args.dataset_path)
    
    suffix = f"_{subset_name}"
    if args.clean: suffix += "_clean"
    if args.sample: suffix += f"_sample{args.sample}"
    if args.batch_size > 1: suffix += f"_b{args.batch_size}"
        
    os.makedirs("results", exist_ok=True)
    csv_output_path = f"results/{clean_model_name}{suffix}.csv"
    json_output_path = f"results/{clean_model_name}{suffix}.json"

    if not os.path.exists(args.prompt): raise FileNotFoundError(f"Instruction prompt file not found at: {args.prompt}")
    with open(args.prompt, "r", encoding="utf-8") as f: user_prompt_text = f.read().strip()

    print(f"Loading dataset partition from HuggingFace Local Storage: {args.dataset_path}")
    if not os.path.exists(args.dataset_path): raise FileNotFoundError(f"Dataset directory target not located at: {args.dataset_path}")
    
    loaded_dict = load_from_disk(args.dataset_path)
    test_dataset = loaded_dict["test"]

    if args.clean:
        print("Filter active: subsetting dataset to 'clean' attack types only.")
        test_dataset = test_dataset.filter(lambda row: row["attack_type"] == "clean")
        print(f"Subset completed. Base clean rows available: {len(test_dataset):,}")

    if args.sample is not None and args.sample <= len(test_dataset):
        print(f"Compiling random deterministic subset slice. Selecting exactly {args.sample} records...")
        random.seed(42)
        indices = random.sample(range(len(test_dataset)), args.sample)
        test_dataset = test_dataset.select(indices)
        print(f"Subsampling successful. Array size locked to: {len(test_dataset)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device configuration set to: {device}")
    
    model_lower = args.model.lower()
    # Only true for actual Gemma checkpoints
    is_gemma_family = "gemma" in model_lower
    # Treat Sea-Lion (Qwen SEA-LION) as part of the Qwen family
    is_qwen_family = ("qwen" in model_lower) or ("sea-lion" in model_lower)
    is_deepseek_ocr = args.deepseek or "deepseek" in model_lower

    processor = None
    model = None
    kraken_model = None
    deepseek_tokenizer = None
    deepseek_output_dir = None
    deepseek_prompt = None

    if args.kraken:
        print("Kraken Engine Mode activated...")
        try:
            kraken_model = build_kraken_model(args.model)
            print("Loaded Kraken recognition components successfully.")
        except Exception as kraken_init_err:
            print(f"CRITICAL: Failed to load Kraken infrastructure: {kraken_init_err}")
            kraken_model = None

    elif is_deepseek_ocr:
        print("DeepSeek OCR Engine Mode activated...")
        deepseek_tokenizer, model = build_deepseek_model(args.model)
        if "cuda" in device:
            model = model.cuda().to(torch.bfloat16)
        else:
            model = model.to(device)

        deepseek_output_dir = os.path.join(CACHE_DIR, "deepseek_eval_outputs")
        os.makedirs(deepseek_output_dir, exist_ok=True)
        deepseek_prompt = f"<image>\n{user_prompt_text}"

    elif is_gemma_family:
        print("Gemma-3 Engine Mode activated...")
        processor, model = build_gemma_model(args.model, device)

    elif is_qwen_family:
        print(f"Initializing Qwen-family Engine on Target Device: {device}")
        if args.jawi_qwen_v1 or args.jawi_qwen_v2:
            print(f"Configuring explicit Jawi-Qwen structural loaders...")
            processor, model = build_qwen_model(args.model, model_lower, device, jawi_qwen_v1=args.jawi_qwen_v1, jawi_qwen_v2=args.jawi_qwen_v2)
        else:
            processor, model = build_qwen_model(args.model, model_lower, device)

        try:
            model.eval()
        except Exception:
            pass

    else:
        print(f"Initializing Transformers Engine on Target Device: {device}")
        # Fallback: attempt to load via Qwen runner as general handler
        if args.jawi_qwen_v1 or args.jawi_qwen_v2:
            print(f"Configuring explicit Jawi-Qwen structural loaders...")
            processor, model = build_qwen_model(args.model, model_lower, device, jawi_qwen_v1=args.jawi_qwen_v1, jawi_qwen_v2=args.jawi_qwen_v2)
        else:
            processor, model = build_qwen_model(args.model, model_lower, device)
        try:
            model.eval()
        except Exception:
            pass

    if model is not None:
        try:
            model.eval()
        except Exception:
            pass

    chrf_metric = CHRF()
    results_records = []

    if "cuda" in device:
        print("Pinning persistent memory footprint block to lock process ID tracking...")
        _persistent_anchor = torch.zeros((1024, 1024), device=device)

    dataset_records = [row for row in test_dataset]
    total_records = len(dataset_records)
    
    print(f"\nCommencing Robust Batched Inference Pipeline Slices (Batch Size: {args.batch_size})")
    
    for i in tqdm(range(0, total_records, args.batch_size), desc="Processing Batches"):
        batch_slice = dataset_records[i : i + args.batch_size]
        
        batch_imgs = []
        for row in batch_slice:
            base_img = row["Image"]
            if not isinstance(base_img, Image.Image):
                base_img = Image.open(io.BytesIO(base_img["bytes"])).convert("RGB")
            else:
                base_img = base_img.convert("RGB")
            batch_imgs.append(base_img)

        batch_predictions = []
        
        # Safe device target identification (safeguards pipeline tracking)
        target_device = model.device if hasattr(model, "device") else device

        # --- ENGINE ROUTING LOGIC ---
        if args.kraken:
            for img in batch_imgs:
                if kraken_model is not None:
                    try: pred = run_kraken_inference(kraken_model, img)
                    except Exception: pred = ""
                else: pred = ""
                batch_predictions.append(pred)
                
        elif is_deepseek_ocr and model is not None and deepseek_tokenizer is not None:
            try:
                identifiers = [row["Identifier"] for row in batch_slice]
                batch_preds = run_deepseek_batch(
                    model, deepseek_tokenizer, batch_imgs, identifiers, deepseek_prompt, deepseek_output_dir
                )
                batch_predictions.extend(batch_preds)
            except Exception as e:
                print(f"[DeepSeek Batch Failure] {e}")
                for sub_idx, img in enumerate(batch_imgs):
                    row_meta = batch_slice[sub_idx]
                    try:
                        pred = run_deepseek_inference(
                            model, deepseek_tokenizer, img, row_meta["Identifier"], deepseek_prompt, deepseek_output_dir
                        )
                    except Exception:
                        pred = ""
                    batch_predictions.append(pred)
                
        elif is_gemma_family:
            try:
                inputs = prepare_gemma_inputs(processor, model, batch_imgs, user_prompt_text)
                batch_preds = generate_gemma_text(processor, model, inputs)
                batch_predictions.extend(batch_preds)
            except Exception as e:
                print(f"[Gemma Batch Failure] {e}")
                for img in batch_imgs:
                    try:
                        inputs = prepare_gemma_inputs(processor, model, img, user_prompt_text)
                        pred = generate_gemma_text(processor, model, inputs)
                    except Exception:
                        pred = ""
                    batch_predictions.append(pred)
                
        else:
            try:
                if "qwen" in model_lower or "qari" in model_lower:
                    from qwen_vl_utils import process_vision_info
                    
                    batch_inputs_structs = []
                    for img in batch_imgs:
                        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_prompt_text}]}]
                        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        image_inputs, video_inputs = process_vision_info(messages)
                        batch_inputs_structs.append((text_prompt, image_inputs, video_inputs))
                        
                    text_prompts = [x[0] for x in batch_inputs_structs]
                    all_images = [x[1][0] for x in batch_inputs_structs if x[1]]
                    
                    inputs = processor(text=text_prompts, images=all_images, padding="longest", return_tensors="pt").to(target_device)
                    
                    if hasattr(model, "dtype"):
                        inputs = {k: v.to(model.dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}
                        
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs, 
                            max_new_tokens=128, 
                            do_sample=False,
                            pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else None,
                            eos_token_id=processor.tokenizer.eos_token_id if hasattr(processor, "tokenizer") else None
                        )
                        
                    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
                    
                    if args.jawi_qwen_v1 or args.jawi_qwen_v2:
                        batch_preds = processor.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
                    else:
                        batch_preds = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        
                    batch_predictions.extend([p.strip() for p in batch_preds])
                else:
                    for img in batch_imgs:
                        inputs = prepare_standard_inputs(processor, model, img, user_prompt_text, model_lower)
                        pred_list = generate_standard_text(processor, model, inputs, jawi_qwen_v1=args.jawi_qwen_v1, jawi_qwen_v2=args.jawi_qwen_v2)
                        batch_predictions.append(pred_list[0] if isinstance(pred_list, list) else pred_list)
            except Exception as batch_vlm_err:
                print(f"\n[Batch VLM Failure Catch] Index window context starting at {i}: {batch_vlm_err}")
                for img in batch_imgs:
                    try:
                        inputs = prepare_standard_inputs(processor, model, img, user_prompt_text, model_lower)
                        pred_list = generate_standard_text(processor, model, inputs, jawi_qwen_v1=args.jawi_qwen_v1, jawi_qwen_v2=args.jawi_qwen_v2)
                        batch_predictions.append(pred_list[0] if isinstance(pred_list, list) else pred_list)
                    except Exception:
                        batch_predictions.append("")

        if len(batch_predictions) < len(batch_slice):
            batch_predictions += [""] * (len(batch_slice) - len(batch_predictions))

        # --- EVALUATE METRICS FOR CURRENT BATCH SLICE ---
        for sub_idx, row in enumerate(batch_slice):
            ground_truth = str(row["Text"]).strip()
            attack_type = row["attack_type"]
            identifier = row["Identifier"]
            raw_prediction = batch_predictions[sub_idx]

            clean_prediction = clean_ocr_text(raw_prediction)
            cleaned_diacritics_prediction = remove_diacritics(clean_prediction)
            
            gt_no_spaces = ground_truth.replace(" ", "")
            gt_no_diacritics = remove_diacritics(ground_truth)
            gt_no_diacritics_no_spaces = gt_no_diacritics.replace(" ", "")

            try: wer_with = wer(ground_truth, clean_prediction) if len(ground_truth) > 0 else 1.0
            except Exception: wer_with = 1.0

            pred_with_no_spaces = clean_prediction.replace(" ", "")
            cer_data_with = calculate_cer_components(gt_no_spaces, pred_with_no_spaces)
            
            try: chrf_with = chrf_metric.sentence_score(clean_prediction, [ground_truth]).score / 100.0
            except Exception: chrf_with = 0.0

            try: wer_without = wer(gt_no_diacritics, cleaned_diacritics_prediction) if len(gt_no_diacritics) > 0 else 1.0
            except Exception: wer_without = 1.0

            pred_without_no_spaces = cleaned_diacritics_prediction.replace(" ", "")
            cer_data_without = calculate_cer_components(gt_no_diacritics_no_spaces, pred_without_no_spaces)
            
            try: chrf_without = chrf_metric.sentence_score(cleaned_diacritics_prediction, [gt_no_diacritics]).score / 100.0
            except Exception: chrf_without = 0.0

            results_records.append({
                "Identifier": identifier, 
                "attack_type": attack_type, 
                "original_text": ground_truth, 
                "raw_prediction": raw_prediction, 
                "clean_prediction": clean_prediction, 
                "cleaned_diacritics_prediction": cleaned_diacritics_prediction,
                
                "wer_with_diacritics": wer_with, 
                "cer_with_diacritics": cer_data_with["cer"], 
                "substitute_cer_with_diacritics": cer_data_with["sub_rate"], 
                "delete_cer_with_diacritics": cer_data_with["del_rate"], 
                "insertion_cer_with_diacritics": cer_data_with["ins_rate"], 
                "chrF_with_diacritics": chrf_with,
                
                "wer_without_diacritics": wer_without, 
                "cer_without_diacritics": cer_data_without["cer"], 
                "substitute_cer_without_diacritics": cer_data_without["sub_rate"], 
                "delete_cer_without_diacritics": cer_data_without["del_rate"], 
                "insertion_cer_without_diacritics": cer_data_without["ins_rate"], 
                "chrF_without_diacritics": chrf_without
            })

    if not results_records: return
    df_results = pd.DataFrame(results_records)
    
    csv_parent = os.path.dirname(csv_output_path)
    if csv_parent: os.makedirs(csv_parent, exist_ok=True)
    json_parent = os.path.dirname(json_output_path)
    if json_parent: os.makedirs(json_parent, exist_ok=True)

    df_results.to_csv(csv_output_path, index=False, encoding="utf-8")
    
    # Compile Summary Metric Structures
    agg_json_metrics = {}
    grouped = df_results.groupby("attack_type")
    
    for attack_name, group in grouped:
        agg_json_metrics[attack_name] = {
            "sample_count": int(len(group)),
            "with_diacritics": {
                "avg_wer": float(group["wer_with_diacritics"].mean()), 
                "avg_cer": float(group["cer_with_diacritics"].mean()), 
                "avg_substitute_cer": float(group["substitute_cer_with_diacritics"].mean()), 
                "avg_delete_cer": float(group["delete_cer_with_diacritics"].mean()), 
                "avg_insertion_cer": float(group["insertion_cer_with_diacritics"].mean()), 
                "avg_chrF": float(group["chrF_with_diacritics"].mean())
            },
            "without_diacritics": {
                "avg_wer": float(group["wer_without_diacritics"].mean()), 
                "avg_cer": float(group["cer_without_diacritics"].mean()), 
                "avg_substitute_cer": float(group["substitute_cer_without_diacritics"].mean()), 
                "avg_delete_cer": float(group["delete_cer_without_diacritics"].mean()), 
                "avg_insertion_cer": float(group["insertion_cer_without_diacritics"].mean()), 
                "avg_chrF": float(group["chrF_without_diacritics"].mean())
            }
        }
        
    agg_json_metrics["GLOBAL_TOTAL_AVERAGE"] = {
        "sample_count": int(len(df_results)),
        "with_diacritics": {
            "avg_wer": float(df_results["wer_with_diacritics"].mean()), 
            "avg_cer": float(df_results["cer_with_diacritics"].mean()), 
            "avg_substitute_cer": float(df_results["substitute_cer_with_diacritics"].mean()), 
            "avg_delete_cer": float(df_results["delete_cer_with_diacritics"].mean()), 
            "avg_insertion_cer": float(df_results["insertion_cer_with_diacritics"].mean()), 
            "avg_chrF": float(df_results["chrF_with_diacritics"].mean())
        },
        "without_diacritics": {
            "avg_wer": float(df_results["wer_without_diacritics"].mean()), 
            "avg_cer": float(df_results["cer_without_diacritics"].mean()), 
            "avg_substitute_cer": float(df_results["substitute_cer_without_diacritics"].mean()), 
            "avg_delete_cer": float(df_results["delete_cer_without_diacritics"].mean()), 
            "avg_insertion_cer": float(df_results["insertion_cer_without_diacritics"].mean()), 
            "avg_chrF": float(df_results["chrF_without_diacritics"].mean())
        }
    }

    with open(json_output_path, "w", encoding="utf-8") as json_file: 
        json.dump(agg_json_metrics, json_file, indent=4, ensure_ascii=False)
        
    print(f"Successfully compiled batched comparative logs for: {clean_model_name} on {subset_name}")

if __name__ == "__main__":
    main()
