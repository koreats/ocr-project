def load_corrections(filepath):
    """Loads OCR correction rules from a file."""
    correction_dict = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) == 2:
                        error, correction = parts
                        correction_dict[error.strip()] = correction.strip()
    except FileNotFoundError:
        pass
    return correction_dict

def post_process_text(ocr_result, wrapper, correction_dict, config):
    """Formats and corrects the raw OCR result."""
    paragraphs = []
    current_paragraph = ""
    for item in ocr_result:
        line = item[1].strip()
        current_paragraph += line + " "
        if line.endswith(('.', '?', '!', ':')):
            paragraphs.append(current_paragraph.strip())
            current_paragraph = ""
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())
    
    corrected_paragraphs = []
    for p in paragraphs:
        temp_p = p
        for error, correction in correction_dict.items():
            temp_p = temp_p.replace(error, correction)
        corrected_paragraphs.append(temp_p)
        
    wrapper.width = config.get('text_wrap_width', 70)
    wrapped_paragraphs = [wrapper.fill(p) for p in corrected_paragraphs]
    return "\n\n".join(wrapped_paragraphs)
