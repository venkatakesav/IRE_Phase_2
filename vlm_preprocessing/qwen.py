import os
import json
import re
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm

custom_cache_dir = "/scratch/rud"

# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=custom_cache_dir,
    attn_implementation="flash_attention_2"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", cache_dir=custom_cache_dir)

# Dataset directory
dataset_dir = "dataset"

# Regex pattern to extract product ID (number before ".json")
product_id_pattern = re.compile(r"(\d+)\.json$")

# Recursive traversal of the dataset directory
for root, _, files in os.walk(dataset_dir):
    for filename in files:
        if filename.endswith(".json"):
            # Extract product ID using regex
            match = product_id_pattern.match(filename)
            print(filename)
            if not match:
                print(f"Skipping file {filename} (no product ID found)")
                continue
            product_id = match.group(1)
            print("Processing product ID:", product_id)

            # Load JSON data
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)

                # Extract title and item_specifics
                title = data.get("title", "")
                item_specifics = data.get("item_specifics", {})
                image_path = root + "/imgs/"+data.get("main_image","") +".jpg"
                

                product_template = f"""                 
                Given the product title, metadata, and the provided image, generate a detailed and informative description. The description should include:
                - The product's title as a key identifier.
                - A detailed explanation of the condition, ensuring clarity about its state and packaging.
                - Mention any specific attributes like unique identifiers, brand, material, or notable features [only those mentioned in the metadata; do not infer new details].
                - Use the image to validate and enrich the description, ensuring alignment between the textual details and the visual representation.
                - Maintain a professional tone suitable for a product listing or database query.

                You will be given:
                - `title` and `item_specifics` (metadata)
                - An image associated with the product.

                Ensure that the description incorporates all specifics mentioned in the `item_specifics` and aligns with the provided image. Do not infer or add information that is not present in the `item_specifics` or visible in the image.

                If either `title` or `item_specifics` is missing, or if the image does not add additional clarifying details, focus only on the provided data. Your role is to transform the JSON data into a well-structured, grammatically correct product description.

                The product details are:
                {{
                    "title": "{title}",
                    "item_specifics": {json.dumps(item_specifics)}
                }}
                """

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": product_template},
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

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                # Save the output to a new JSON file named after the product ID
                output_filename = f"{product_id}.txt"
                output_file_path = os.path.join(root, output_filename)
                with open(output_file_path, 'w') as output_file:
                    json.dump({"description": output_text}, output_file)