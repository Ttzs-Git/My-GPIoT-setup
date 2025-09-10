from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from peft import PeftModel
import torch

def data_transform(message_list):
    system_message = message_list[0]["content"]
    user_message = message_list[1]["content"]

    output = "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n{}[/INST]".format(
        system_message, user_message.strip()
    )

    return output


base_model_dir = "meta-llama/Llama-2-13b-chat-hf"
lora_model_dir = "GPIoT_Code_Generation/checkpoint-13000"

tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
model = AutoModelForCausalLM.from_pretrained(
    base_model_dir,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(model, lora_model_dir)
# model = model.to("cuda")
model.eval()

prompt = [
    {
        "role": "system",
        "content": "You are a professional and skillful Python programmer, especially in the field of communication, signal processing, and machine learning. According to the user instruction, you need to generate one single Python function with detailed comments and documentation. The documentation should be in the Markdown format.",
    },
    {
        "role": "user",
        "content": "**Target**\nDefine a Python function to create a simple augmentation pipeline for image processing and provide detailed code comments.\n\n**Input Specifications**\n- `image_path` (str): The file path to the input image.\n\n**Output specifications**\nThe function does not explicitly return any value but visualizes the original and augmented images using matplotlib.",
    },
]

input_text = data_transform(prompt)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
input_ids = inputs["input_ids"]
generation_config = GenerationConfig(
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id

)

output = model.generate(
    input_ids,
    generation_config=generation_config,  # ← 关键：使用 generation_config
)
output_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(output_text)
