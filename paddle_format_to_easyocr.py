import os
import re
import pandas as pd

def convert_paddle_to_easyocr(paddle_manifest_path, easyocr_output_dir):
    """
    Converts a PaddleOCR tab-separated manifest file to an EasyOCR labels.csv structure.
    
    Args:
        paddle_manifest_path (str): Path to your 'rec_gt_train.txt' or validation file.
        easyocr_output_dir (str): Folder where the new labels.csv should be saved.
    """
    print(f"🔄 Processing: {paddle_manifest_path}")
    
    if not os.path.exists(paddle_manifest_path):
        print(f"❌ Error: Could not find PaddleOCR file at {paddle_manifest_path}")
        return

    # Create the output directory if it doesn't exist yet
    os.makedirs(easyocr_output_dir, exist_ok=True)
    
    data_records = []
    
    with open(paddle_manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            # PaddleOCR uses strict tabs (\t) to split path and label
            if '\t' not in line:
                print(f"⚠️ Warning line {line_num}: No tab character found. Skipping line.")
                continue
                
            parts = line.split('\t')
            relative_img_path = parts[0]
            # Handle cases where multiple tabs might exist or text has trailing blocks
            text_label = parts[1] 
            
            # EasyOCR expects just the filename inside its subfolder, NOT 'train/img.jpg'
            # Extracts 'img_001.png' from 'train/rec/img_001.png'
            filename = os.path.basename(relative_img_path)
            
            # Your regex parser 'sep=^([^,]+),' splits at the first comma. 
            # If your filename contains a comma, it will break your EasyOCR training loader.
            if ',' in filename:
                print(f"⚠️ Warning: Filename '{filename}' contains a comma! Replacing with underscore to prevent parser failure.")
                filename = filename.replace(',', '_')
                
            data_records.append({
                'filename': filename,
                'words': text_label
            })
            
    # Convert into a structured DataFrame to guarantee valid CSV quoting architectures
    df = pd.DataFrame(data_records)
    
    # Write out the CSV file explicitly matching your expected parameters
    output_file_path = os.path.join(easyocr_output_dir, 'labels.csv')
    
    # We enforce keep_default_na=False compatibility by keeping empty strings empty
    df.to_csv(output_file_path, index=False, encoding='utf-8', sep=',')
    
    print(f"✅ Success! Saved {len(df)} records to: {output_file_path}\n")


if __name__ == "__main__":
    # Define your source mappings (Modify these if your paths are located somewhere else)
    PADDLE_BASE = "/dest/thura/code/paddle_jawi_outputs"
    EASYOCR_BASE = "/dest/thura/code/jawi_easyocr_data"
    
    # 1. Convert Augmented Training Pipeline Split
    convert_paddle_to_easyocr(
        paddle_manifest_path=os.path.join(PADDLE_BASE, "augmented/rec/rec_gt_train.txt"),
        easyocr_output_dir=os.path.join(EASYOCR_BASE, "train/augmented")
    )
    
    # 2. Convert Clean Baseline Training Pipeline Split
    convert_paddle_to_easyocr(
        paddle_manifest_path=os.path.join(PADDLE_BASE, "clean/rec/rec_gt_train.txt"),
        easyocr_output_dir=os.path.join(EASYOCR_BASE, "train/clean")
    )
    
    # 3. Convert Validation Pipeline Split
    convert_paddle_to_easyocr(
        paddle_manifest_path=os.path.join(PADDLE_BASE, "augmented/rec/rec_gt_validation.txt"),
        easyocr_output_dir=os.path.join(EASYOCR_BASE, "valid/validation")
    )
    
    print("🎉 All dataset manifests have been completely reformatted for EasyOCR!")
