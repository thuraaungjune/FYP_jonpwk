# import contextlib
# import io
# import torch

# from .common import CACHE_DIR, extract_deepseek_prediction, sanitize_filename

# device = "cuda" if torch.cuda.is_available() else "cpu"

# def build_deepseek_model(model_name):
#     from transformers import AutoModel, AutoTokenizer

#     tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=CACHE_DIR)
    
#     try:
#         # If we are strictly on CPU, don't even try flash_attention_2 or auto device mapping
#         if device == "cpu":
#             raise RuntimeError("CPU mode forced; skipping Flash-Attention initialization path.")
            
#         model = AutoModel.from_pretrained(
#             model_name,
#             _attn_implementation="flash_attention_2",
#             trust_remote_code=True,
#             use_safetensors=True,
#             cache_dir=CACHE_DIR,
#             device_map="auto",
#         )
#     except Exception as deepseek_load_err:
#         print(f"Flash-attention load failed, retrying with default attention: {deepseek_load_err}")
        
#         # FIXED: Explicitly dictate device mapping configuration or throw to CPU layout
#         model = AutoModel.from_pretrained(
#             model_name,
#             trust_remote_code=True,
#             use_safetensors=True,
#             cache_dir=CACHE_DIR,
#             device_map="auto" if device == "cuda" else None, # Prevents automated GPU probing when device == cpu
#         )

#     # FIXED: Force the model weights entirely to CPU memory spaces if CUDA is broken/unsupported
#     if device == "cpu":
#         print("Explicitly casting DeepSeek model structure to CPU storage map...")
#         model = model.to("cpu")

#     return tokenizer, model.eval()


# def run_deepseek_inference(model, tokenizer, base_img, identifier, prompt_text, output_dir):
#     image_path = f"{output_dir}/{sanitize_filename(identifier)}.jpg"
#     base_img.save(image_path)

#     capture_buffer = io.StringIO()
#     with contextlib.redirect_stdout(capture_buffer):
#         deepseek_result = model.infer(
#             tokenizer,
#             prompt=prompt_text,
#             image_file=image_path,
#             output_path=output_dir,
#             base_size=1024,
#             image_size=640,
#             crop_mode=True,
#             save_results=True,
#             test_compress=True,
#         )

#     stdout_captured = capture_buffer.getvalue()
#     generated_text = extract_deepseek_prediction(deepseek_result, output_dir)
#     if not generated_text and stdout_captured:
#         generated_text = extract_deepseek_prediction(stdout_captured, output_dir)
#     return generated_text

import contextlib
import io
import torch

from .common import CACHE_DIR, extract_deepseek_prediction, sanitize_filename

device = "cuda" if torch.cuda.is_available() else "cpu"

def build_deepseek_model(model_name):
    from transformers import AutoModel, AutoTokenizer

    print(f"Initializing DeepSeek-OCR v2 Components from target path: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=CACHE_DIR)
    # prefer left-padding for decoder-only tokenizers to avoid generation warnings
    try:
        is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False) if 'model' in locals() else False
        is_decoder = getattr(model.config, "is_decoder", False) if 'model' in locals() else False
        if not is_encoder_decoder and is_decoder:
            try:
                tokenizer.padding_side = "left"
            except Exception:
                pass
    except Exception:
        pass
    
    try:
        if device == "cpu":
            raise RuntimeError("CPU mode forced; skipping Flash-Attention initialization path.")
            
        model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
            cache_dir=CACHE_DIR,
            device_map="auto",
        )
        # DeepSeek-OCR-2 highly recommends evaluating weights in bfloat16 precision maps
        model = model.to(torch.bfloat16)
        
    except Exception as deepseek_load_err:
        print(f"Flash-attention or bfloat16 initialization failed, retrying with fallback pipeline: {deepseek_load_err}")
        
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
            cache_dir=CACHE_DIR,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cuda":
            model = model.to(torch.bfloat16)

    if device == "cpu":
        print("Explicitly casting DeepSeek model structure to CPU storage map...")
        model = model.to("cpu")

    # Ensure tokenizer padding side for decoder-only models
    try:
        is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False)
        is_decoder = getattr(model.config, "is_decoder", False)
        if not is_encoder_decoder and is_decoder:
            try:
                tokenizer.padding_side = "left"
            except Exception:
                pass
    except Exception:
        pass

    return tokenizer, model.eval()


def run_deepseek_inference(model, tokenizer, base_img, identifier, prompt_text, output_dir):
    image_path = f"{output_dir}/{sanitize_filename(identifier)}.jpg"
    base_img.save(image_path)

    capture_buffer = io.StringIO()
    with contextlib.redirect_stdout(capture_buffer):
        # Updated parameters to support DeepSeek-OCR-2 defaults
        deepseek_result = model.infer(
            tokenizer,
            prompt=prompt_text,
            image_file=image_path,
            output_path=output_dir,
            base_size=1024,
            image_size=768,      # Updated from 640 to 768 for v2 standard compliance
            crop_mode=True,
            save_results=True,
        )

    stdout_captured = capture_buffer.getvalue()
    generated_text = extract_deepseek_prediction(deepseek_result, output_dir)
    if not generated_text and stdout_captured:
        generated_text = extract_deepseek_prediction(stdout_captured, output_dir)
    return generated_text


def run_deepseek_batch(model, tokenizer, images, identifiers, prompt_text, output_dir, max_workers=4):
    """
    Run DeepSeek inference over a list of PIL images in parallel using a thread
    pool. Returns a list of extracted strings in the same order as inputs.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not isinstance(images, (list, tuple)):
        images = [images]
    if not isinstance(identifiers, (list, tuple)):
        identifiers = [identifiers]

    results = [""] * len(images)
    def _call(idx, img, ident):
        try:
            return idx, run_deepseek_inference(model, tokenizer, img, ident, prompt_text, output_dir)
        except Exception:
            return idx, ""

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_call, i, im, ident) for i, (im, ident) in enumerate(zip(images, identifiers))]
        for fut in as_completed(futures):
            idx, text = fut.result()
            results[idx] = text

    return results
