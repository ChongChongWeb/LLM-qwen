import torch

# Set max_split_size_mb for potential memory fragmentation reduction

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download

# Download the model
model_dir = snapshot_download("qwen/Qwen2-VL-2B-Instruct")

# Try lower precision if needed, potentially impacting accuracy
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype=torch.float16, device_map="auto"
)

# Processor with potentially reduced visual token range
# Adjust min/max_pixels based on your needs
processor = AutoProcessor.from_pretrained(
    model_dir, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
)

# Messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "gxl.jpg"},
            {"type": "text", "text": "这个女人美不美？."},
        ],
    }
]
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)