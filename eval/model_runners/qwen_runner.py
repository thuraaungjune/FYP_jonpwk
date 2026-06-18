import torch
import os

from .common import CACHE_DIR

def build_qwen_model(model_name, model_lower, device, jawi_qwen_v1=False, jawi_qwen_v2=False):
    from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2VLForConditionalGeneration

    # Set up optimal settings for GPU vs CPU up front
    dev_map = "auto" if device == "cuda" else None
    data_type = torch.bfloat16 if device == "cuda" else torch.float32

    if jawi_qwen_v2:
        print("Loading Jawi-OCR-Qwen-v2 model and processor...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=data_type,
            device_map=dev_map,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", cache_dir=CACHE_DIR)
    elif jawi_qwen_v1:
        print("Loading Jawi-OCR-Qwen-v1 model and processor...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=data_type,
            cache_dir=CACHE_DIR,
            device_map=dev_map,
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", cache_dir=CACHE_DIR)
    else:
        # Dynamic fallback: Try AutoModel first. If it omits model_type or is an adapter, 
        # intercept and gracefully load it with its corresponding base weights.
        from transformers import Qwen2VLConfig

        print(f"Loading processor and model from: {model_name}")
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)

        try:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=data_type,
                trust_remote_code=True,
                cache_dir=CACHE_DIR,
                device_map=dev_map,
            )
        except Exception as e:
            try:
                print(f"AutoModel load failed ({e}). Checking for PEFT adapter or explicit configuration path...")
                
                # Check repo details to identify if it is a standalone weight file or a PEFT adapter
                is_adapter = False
                try:
                    from huggingface_hub import list_repo_files
                    repo_files = list_repo_files(model_name)
                    if "adapter_config.json" in repo_files:
                        is_adapter = True
                except Exception:
                    if "qari" in model_name.lower():
                        is_adapter = True

                if is_adapter:
                    print("Detected LoRA Adapter configuration. Resolving via PeftModel wrapper...")
                    from peft import PeftModel
                    
                    base_model_id = "Qwen/Qwen2-VL-2B-Instruct"
                    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        base_model_id,
                        torch_dtype=data_type,
                        device_map=dev_map,
                        cache_dir=CACHE_DIR,
                    )
                    model = PeftModel.from_pretrained(base_model, model_name, cache_dir=CACHE_DIR)
                else:
                    cfg = Qwen2VLConfig.from_pretrained(model_name, cache_dir=CACHE_DIR)
                    if not hasattr(cfg, "model_type") or not cfg.model_type:
                        cfg.model_type = "qwen2_vl"

                    model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        config=cfg,
                        torch_dtype=data_type,
                        device_map=dev_map,
                        trust_remote_code=True,
                        cache_dir=CACHE_DIR,
                    )
            except Exception as e2:
                print(f"Fallback Qwen2VL loading logic completely failed: {e2}")
                raise

    # If the loaded model is decoder-only (no encoder), prefer left-padding for correct generation
    try:
        is_encoder_decoder = getattr(model.config, "is_encoder_decoder", False)
        is_decoder = getattr(model.config, "is_decoder", False)
        if not is_encoder_decoder and is_decoder and hasattr(processor, "tokenizer"):
            try:
                processor.tokenizer.padding_side = "left"
            except Exception:
                pass
    except Exception:
        pass

    return processor, model


def prepare_standard_inputs(processor, model, base_img, user_prompt_text, model_lower):
    """
    Accepts a single PIL image or a list of PIL images and returns a batched inputs dict
    ready to pass to `model.generate`. Always returns tensors on `model.device`.
    """
    single = False
    if not isinstance(base_img, (list, tuple)):
        base_imgs = [base_img]
        single = True
    else:
        base_imgs = list(base_img)

    # Build text prompts for each image
    text_prompts = []
    for img in base_imgs:
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_prompt_text}]}]
        text_prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    # Safe retrieval of the underlying target device even if wrapped in DataParallel
    target_device = model.module.device if hasattr(model, "module") else model.device

    # Qwen branch: rely on processor batching of text+images
    if "qwen" in model_lower or "qari" in model_lower:
        try:
            inputs = processor(text=text_prompts, images=base_imgs, padding="longest", return_tensors="pt")
        except Exception:
            inputs_list = [processor(text=t, images=[img], return_tensors="pt") for t, img in zip(text_prompts, base_imgs)]
            inputs = processor(text=text_prompts, images=base_imgs, padding="longest", return_tensors="pt")

        # Move tensors to correct underlying device safely
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
        return inputs if not single else {k: v[0:1] for k, v in inputs.items()}

    # Non-Qwen standard path: provide batched `images` + `text`
    inputs = processor(images=base_imgs, text=text_prompts, padding="longest", return_tensors="pt")
    return inputs.to(target_device) if not single else {k: v[0:1].to(target_device) for k, v in inputs.items()}


def generate_standard_text(processor, model, inputs, jawi_qwen_v1=False, jawi_qwen_v2=False):
    """
    Accepts a batched `inputs` dict (or a single-item batch) and returns a list of
    decoded prediction strings (one per batch element).
    """
    if hasattr(model, "dtype"):
        inputs = {k: v.to(model.dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}

    pad_token_id = None
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "pad_token_id"):
        pad_token_id = processor.tokenizer.pad_token_id

    if "attention_mask" in inputs:
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    elif "input_ids" in inputs and pad_token_id is not None:
        input_lens = (inputs["input_ids"] != pad_token_id).sum(dim=1).tolist()
    else:
        batch_size = inputs[next(iter(inputs))].shape[0]
        input_lens = [inputs["input_ids"].shape[1]] * batch_size if "input_ids" in inputs else [0] * batch_size

    with torch.no_grad():
        generate_kwargs = {
            "max_new_tokens": 128,
            "do_sample": False,
            "pad_token_id": pad_token_id,
            "eos_token_id": getattr(processor.tokenizer, "eos_token_id", None) if hasattr(processor, "tokenizer") else None,
        }
        generated_ids = model.generate(**inputs, **generate_kwargs)

    if generated_ids.dim() == 1:
        generated_ids = generated_ids.unsqueeze(0)

    decoded = []
    for i in range(generated_ids.shape[0]):
        start_idx = int(input_lens[i]) if i < len(input_lens) else 0
        new_ids = generated_ids[i, start_idx:]
        if jawi_qwen_v1 or jawi_qwen_v2:
            text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        else:
            try:
                text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
            except Exception:
                text = processor.batch_decode(new_ids.unsqueeze(0), skip_special_tokens=True)[0]
        decoded.append(text.strip())

    return decoded


def build_qari_model(model_name, device=None):
    """Load the Qari OCR Qwen-VL model and processor via direct Peft wrappers."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel

    dev_map = "auto"
    base_model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        device_map=dev_map,
        cache_dir=CACHE_DIR,
    )
    model = PeftModel.from_pretrained(base_model, model_name, cache_dir=CACHE_DIR)
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)

    try:
        model.eval()
    except Exception:
        pass

    # If multiple GPUs are visible, wrap with DataParallel
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    except Exception:
        pass

    return processor, model


def prepare_qari_inputs(processor, model, image_src, user_prompt_text, max_new_tokens=2000):
    """Prepare inputs for Qari-style models. Accepts filepath or PIL Image."""
    from qwen_vl_utils import process_vision_info

    if isinstance(image_src, str) and os.path.exists(image_src):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_src}"},
                {"type": "text", "text": user_prompt_text},
            ],
        }]
    else:
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image_src}, {"type": "text", "text": user_prompt_text}],
        }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    
    # Safe retrieval of device for explicit Qari pipelines
    target_device = model.module.device if hasattr(model, "module") else model.device
    try:
        inputs = {k: v.to(target_device) for k, v in inputs.items()}
    except Exception:
        pass

    return inputs


def generate_qari_text(processor, model, inputs, max_new_tokens=2000):
    """Generate text for the Qari/Qwen2VL model and return decoded string(s)."""
    if hasattr(model, "dtype"):
        inputs = {k: v.to(model.dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}

    pad_token_id = None
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "pad_token_id"):
        pad_token_id = processor.tokenizer.pad_token_id

    if "attention_mask" in inputs:
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    elif "input_ids" in inputs and pad_token_id is not None:
        input_lens = (inputs["input_ids"] != pad_token_id).sum(dim=1).tolist()
    else:
        batch_size = inputs[next(iter(inputs))].shape[0]
        input_lens = [inputs["input_ids"].shape[1]] * batch_size if "input_ids" in inputs else [0] * batch_size

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    if generated_ids.dim() == 1:
        generated_ids = generated_ids.unsqueeze(0)

    decoded = []
    for i in range(generated_ids.shape[0]):
        start_idx = int(input_lens[i]) if i < len(input_lens) else 0
        new_ids = generated_ids[i, start_idx:]
        try:
            text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        except Exception:
            text = processor.batch_decode(new_ids.unsqueeze(0), skip_special_tokens=True)[0]
        decoded.append(text.strip())

    return decoded
