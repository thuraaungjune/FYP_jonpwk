import torch

from .common import CACHE_DIR

def build_gemma_model(model_name, device):
    import torch 
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    print("Loading Gemma-3 model with its native processor...")

    if device == "cuda":
        dev_map = "cuda"
        data_type = torch.bfloat16  
    else:
        dev_map = None
        data_type = torch.float32

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_name,
        device_map=dev_map,
        torch_dtype=data_type,
        cache_dir=CACHE_DIR,
    ).eval()

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=CACHE_DIR)
    processor.tokenizer.padding_side = "left"
    return processor, model



def prepare_gemma_inputs(processor, model, base_img, user_prompt_text):
    """
    Accepts a single PIL image or a list of PIL images and returns batched tensors
    ready for generation. Returns tensors on `model.device` with appropriate dtype.
    """
    single = False
    if not isinstance(base_img, (list, tuple)):
        base_imgs = [base_img]
        single = True
    else:
        base_imgs = list(base_img)

    text_prompts = []
    for img in base_imgs:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": f"{user_prompt_text}\nTranscription:"}]},
        ]
        text_prompts.append(processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))

    inputs = processor(text=text_prompts, images=base_imgs, padding="longest", return_tensors="pt")
    dtype = torch.bfloat16 if model.device.type == "cuda" else torch.float32
    inputs = {k: v.to(model.device, dtype=dtype) if torch.is_floating_point(v) else v.to(model.device) for k, v in inputs.items()}
    return inputs if not single else {k: v[0:1] for k, v in inputs.items()}


def generate_gemma_text(processor, model, inputs):
    """
    Accepts batched `inputs` and returns a list of decoded strings (one per batch
    element). If inputs represent a single element, a one-item list is returned.
    """
    # compute input lengths
    if "attention_mask" in inputs:
        input_lens = inputs["attention_mask"].sum(dim=1).tolist()
    else:
        input_lens = [inputs["input_ids"].shape[1]] * inputs["input_ids"].shape[0]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    if generation.dim() == 1:
        generation = generation.unsqueeze(0)

    outputs = []
    for i in range(generation.shape[0]):
        start = int(input_lens[i])
        new_ids = generation[i, start:]
        try:
            text = processor.decode(new_ids, skip_special_tokens=True)
        except Exception:
            text = processor.batch_decode(new_ids.unsqueeze(0), skip_special_tokens=True)[0]
        outputs.append(text.strip())

    return outputs
