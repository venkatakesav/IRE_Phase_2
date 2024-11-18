import os
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModel

# Initialize processor and model
def initialize_model_and_processor(model_name):
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return processor, model

# Get paths for a specific folder hierarchy
def get_image_paths(base_path, target_category):
    image_paths = []
    try:
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if not os.path.isdir(folder_path):
                continue

            for sub_folder in os.listdir(folder_path):
                if sub_folder != target_category:
                    continue

                sub_folder_path = os.path.join(folder_path, sub_folder)
                for sub_sub_folder in os.listdir(sub_folder_path):
                    sub_sub_folder_path = os.path.join(sub_folder_path, sub_sub_folder)
                    for product in os.listdir(sub_sub_folder_path):
                        product_path = os.path.join(sub_sub_folder_path, product, "imgs")
                        if os.path.exists(product_path):
                            for image_file in os.listdir(product_path):
                                image_paths.append(os.path.join(product_path, image_file))
    except Exception as e:
        print(f"Error while traversing directory structure: {e}")
    return image_paths

# Process a single image
def process_image(image_path, texts, processor, model):
    try:
        print(f"Processing image: {image_path}")
        image = Image.open(image_path)
        inputs = processor(text=texts, images=image, return_tensors="pt", padding="max_length")

        for key, value in inputs.items():
            print(f"{key}: {value.shape}")

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()

            for i, text in enumerate(texts):
                print(f"{probs[0][i]:.1%} probability that image is '{text}'")

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

# Main function
def main(base_path, target_category, model_name):
    processor, model = initialize_model_and_processor(model_name)
    image_paths = get_image_paths(base_path, target_category)
    if not image_paths:
        print("No images found to process.")
        return

    texts = [f"This is a photo of {target_category}."]
    for image_path in image_paths:
        process_image(image_path, texts, processor, model)

# Parameters
BASE_PATH = "/workspace/Retrieval_Experiment/random"
TARGET_CATEGORY = "Cars"
MODEL_NAME = "google/siglip-so400m-patch14-384"

if __name__ == "__main__":
    main(BASE_PATH, TARGET_CATEGORY, MODEL_NAME)
