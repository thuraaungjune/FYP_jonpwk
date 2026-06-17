import gc
import os
import re
import torch

CACHE_DIR = "/home/thura/data/hf_cache"

# ALLOWED_OCR_CHARS Whitelist
ALLOWED_OCR_CHARS = {
    # Whitespace and permitted punctuation
    " ", '؟', '،', '⹁', 'ـ', '–', 
    
    # Core letters & Hamza variants
    'ء', 'أ', 'ئ', 'ا', 'ب', 'ة', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز',
    'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ل', 'م', 'ن', 'ه', 'و', 'ى', 'ي', 
    
    # Custom Jawi extensions
    'پ', 'چ', 'ڠ', 'ڤ', 'ک', 'ڬ', 'ڽ', 'ݢ', 'ۏ',
    
    # Permitted numerals
    '٢', '٤',

    # Arabic Harakat / Diacritics (Crucial so vowelized tokens clean down cleanly)
    '\u064b', # Fathatayn
    '\u064c', # Dammatayn
    '\u064d', # Kasratayn
    '\u064e', # Fatha
    '\u064f', # Damma
    '\u0650', # Kasra
    '\u0651', # Shadda
    '\u0652', # Sukun
    '\u0670', # Dagger Alif
}

# Standard Jawi/Arabic diacritic characters Unicode block definition
# \u064b-\u0652 : Tanween, Fatha, Damma, Kasra, Shadda, Sukun
# \u0640       : Tatweel / Kashida (text elongation character line)
# \u0670       : Alif Khanzariya (superscript Alef)
DIACRITICS_PATTERN = re.compile(r'[\u064b-\u0652\u0640\u0670]')

def clean_ocr_text(text, allowed_chars=None):
    """
    Cleans raw VLM output text by isolating whitelisted Jawi characters
    and stripping away Arabic diacritics/harakat to standardize evaluation.
    """
    if text is None:
        return ""

    # Import inside function if ALLOWED_OCR_CHARS is declared globally elsewhere in the file
    global ALLOWED_OCR_CHARS
    if allowed_chars is None:
        allowed_chars = ALLOWED_OCR_CHARS

    # Standardize spaces
    cleaned = str(text).replace("\u00a0", " ")
    
    # 1. Strip away all Arabic diacritics/harakat/tatweel lines first
    cleaned = DIACRITICS_PATTERN.sub('', cleaned)
    
    # 2. Filter remaining characters by whitelist (keeps core Jawi script, removes English/Markdown)
    cleaned = "".join(ch for ch in cleaned if ch in allowed_chars)
    
    # 3. Collapse duplicate whitespace leaks into single spaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    
    return cleaned

def clear_model_load_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def sanitize_filename(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))

def clean_ocr_text(text, allowed_chars=None):
    if text is None:
        return ""
    if allowed_chars is None:
        allowed_chars = ALLOWED_OCR_CHARS

    cleaned = str(text).replace("\u00a0", " ")
    # Strip out any character not explicitly listed in our Jawi array
    cleaned = "".join(ch for ch in cleaned if ch in allowed_chars)
    # Collapse consecutive spaces down to a single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def extract_deepseek_prediction(result, output_path):
    res = ""
    if isinstance(result, str):
        res = result
    elif isinstance(result, dict):
        for key in ("text", "output", "result", "prediction", "predicted_text"):
            value = result.get(key)
            if isinstance(value, str):
                res = value
                break
    elif isinstance(result, (list, tuple)) and len(result) > 0:
        if isinstance(result[0], str):
            res = result[0]

    if not res and os.path.isdir(output_path):
        candidate_files = []
        for root, _, files in os.walk(output_path):
            for file_name in files:
                if file_name.lower().endswith((".txt", ".md")):
                    candidate_files.append(os.path.join(root, file_name))
        candidate_files.sort(key=os.path.getmtime, reverse=True)
        for file_path in candidate_files:
            try:
                with open(file_path, "r", encoding="utf-8") as file_handle:
                    content = file_handle.read().strip()
                    if content and "output texts tokens" not in content:
                        res = content
                        break
            except Exception:
                continue

    if res:
        res = re.sub(r'<\|ref\|>.*?<\|/ref\|>', '', res)
        res = re.sub(r'<\|det\|>.*?<\|/det\|>', '', res)
        res = re.sub(r'(?i)BASE:\s*torch\.Size.*', '', res)
        res = re.sub(r'(?i)NO\s+PATCHES', '', res)
        res = re.sub(r'={3,}.*?={3,}', '', res)
        res = re.sub(r'-{3,}.*?-{3,}', '', res)
        res = re.sub(r'image size:.*|valid image tokens:.*|output texts tokens \(valid\):.*|compression ratio:.*|===============save results:===============|====================+', '', res, flags=re.MULTILINE)
        lines = [line.strip() for line in res.split('\n') if line.strip()]
        clean_lines = [
            line for line in lines
            if not any(x in line.lower() for x in ["black and white", "jawi script", "torch.size", "image of"])
        ]
        
        raw_combined = " ".join(clean_lines).strip()
        # Explicitly invoke cleaning wrapper before passing back data
        return clean_ocr_text(raw_combined)
    return ""

def calculate_cer_components(reference, hypothesis):
    ref_len = len(reference)
    if ref_len == 0:
        return {"cer": 1.0 if len(hypothesis) > 0 else 0.0, "sub_rate": 0.0, "del_rate": 0.0, "ins_rate": 1.0 if len(hypothesis) > 0 else 0.0}

    dp = [[0] * (len(hypothesis) + 1) for _ in range(ref_len + 1)]
    ops = [[None] * (len(hypothesis) + 1) for _ in range(ref_len + 1)]

    for i in range(ref_len + 1):
        dp[i][0] = i
        ops[i][0] = 'D'
    for j in range(len(hypothesis) + 1):
        dp[0][j] = j
        ops[0][j] = 'I'

    for i in range(1, ref_len + 1):
        for j in range(1, len(hypothesis) + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = 'M'
            else:
                sub_cost = dp[i - 1][j - 1] + 1
                del_cost = dp[i - 1][j] + 1
                ins_cost = dp[i][j - 1] + 1
                min_cost = min(sub_cost, del_cost, ins_cost)
                dp[i][j] = min_cost
                if min_cost == sub_cost: ops[i][j] = 'S'
                elif min_cost == del_cost: ops[i][j] = 'D'
                else: ops[i][j] = 'I'

    subs, dels, inss = 0, 0, 0
    i, j = ref_len, len(hypothesis)
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == 'M':
            i -= 1; j -= 1
        elif op == 'S':
            subs += 1; i -= 1; j -= 1
        elif op == 'D':
            dels += 1; i -= 1
        elif op == 'I':
            inss += 1; j -= 1

    return {"cer": float(dp[ref_len][len(hypothesis)]) / ref_len, "sub_rate": float(subs) / ref_len, "del_rate": float(dels) / ref_len, "ins_rate": float(inss) / ref_len}
