"""
Prompt definitions for SAGE crop disease visual diagnosis.
Defines canonical system and user prompts used consistently across training, validation, testing, and inference.
"""

SYSTEM_PROMPT = "You are an expert plant pathologist AI assistant specializing in agricultural visual diagnosis."
USER_PROMPT = "Given this crop image, identify the crop disease. Respond with only the disease name from the official vocabulary."

def get_canonical_prompt() -> str:
    """Returns the standardized user prompt string."""
    return USER_PROMPT

def get_system_prompt() -> str:
    """Returns the standardized system prompt string."""
    return SYSTEM_PROMPT

def build_conversation(image, disease_label: str = None, include_crop: bool = False, crop_name: str = None):
    """
    Builds standard Qwen2.5-VL conversation messages.
    
    Args:
        image: PIL Image or path to image
        disease_label: Optional target disease label (for training)
        include_crop: Whether to include crop context in user prompt
        crop_name: Crop name if include_crop is True
    
    Returns:
        List of message dicts formatted for Qwen2.5-VL processor
    """
    text_prompt = USER_PROMPT
    if include_crop and crop_name:
        text_prompt = f"Given this {crop_name} crop image, identify the crop disease. Respond with only the disease name from the official vocabulary."
        
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text_prompt}
            ]
        }
    ]
    
    if disease_label is not None:
        messages.append({
            "role": "assistant",
            "content": disease_label
        })
        
    return messages
