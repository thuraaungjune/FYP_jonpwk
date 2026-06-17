import torch

from .common import CACHE_DIR

def build_qwen_model(model_name, model_lower, device, jawi_qwen_v1=False, jawi_qwen_v2=False):
    from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen2VLForConditionalGeneration

    # Set up optimal settings for GPU vs CPU up front
    dev_map = "cuda" if device == "cuda" else None
    data_type = torch.bfloat16 if device == "cuda" else torch.float32

    if jawi_qwen_v2:
        print("Loading Jawi-OCR-Qwen-v2 model and processor...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=data_type,         # 🚨 FIX: Added to prevent CPU execution
            device_map=dev_map,           # 🚨 FIX: Added to force GPU allocation
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
        # Dynamic fallback: Use the model_name itself to pull the matched processor
        print(f"Loading processor and model from: {model_name}")
        processor = AutoProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)

        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=data_type,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            device_map=dev_map,
        )

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
    messages_list = []
    for img in base_imgs:
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": user_prompt_text}]}]
        messages_list.append(messages)
        # prefer string prompts for batch tokenization
        text_prompts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    # Qwen branch: rely on processor batching of text+images
    if "qwen" in model_lower:
        try:
            inputs = processor(text=text_prompts, images=base_imgs, padding="longest", return_tensors="pt")
        except Exception:
            # Fallback: process items individually then collate on CPU
            inputs_list = [processor(text=t, images=[img], return_tensors="pt") for t, img in zip(text_prompts, base_imgs)]
            # naive collate: call processor again with lists
            inputs = processor(text=text_prompts, images=base_imgs, padding="longest", return_tensors="pt")

        # move tensors to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        return inputs if not single else {k: v[0:1] for k, v in inputs.items()}

    # Non-Qwen standard path: provide batched `images` + `text`
    inputs = processor(images=base_imgs, text=text_prompts, padding="longest", return_tensors="pt")
    return inputs.to(model.device) if not single else {k: v[0:1].to(model.device) for k, v in inputs.items()}


def generate_standard_text(processor, model, inputs, jawi_qwen_v1=False, jawi_qwen_v2=False):
    """
    Accepts a batched `inputs` dict (or a single-item batch) and returns a list of
    decoded prediction strings (one per batch element). Backwards-compatible when
    the caller expects a single prediction: the caller can take the first element.
    """
    if hasattr(model, "dtype"):
        inputs = {k: v.to(model.dtype) if torch.is_floating_point(v) else v for k, v in inputs.items()}

    # compute per-sample input lengths using attention mask when possible
    pad_token_id = None
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "pad_token_id"):
        pad_token_id = processor.tokenizer.pad_token_id

    if "attention_mask" in inputs:
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    elif "input_ids" in inputs and pad_token_id is not None:
        input_lens = (inputs["input_ids"] != pad_token_id).sum(dim=1).tolist()
    else:
        # fallback: assume full-length inputs
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

    # Ensure generated_ids is (batch, seq)
    if generated_ids.dim() == 1:
        generated_ids = generated_ids.unsqueeze(0)

    decoded = []
    for i in range(generated_ids.shape[0]):
        start_idx = int(input_lens[i]) if i < len(input_lens) else 0
        new_ids = generated_ids[i, start_idx:]
        if jawi_qwen_v1 or jawi_qwen_v2:
            text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
        else:
            # prefer processor-level batch decode when available
            try:
                text = processor.tokenizer.decode(new_ids, skip_special_tokens=True)
            except Exception:
                text = processor.batch_decode(new_ids.unsqueeze(0), skip_special_tokens=True)[0]
        decoded.append(text.strip())

    return decoded
