import os
from PIL import Image
import torch
from transformers import BlipProcessor, BlipModel

def initialize_model_and_processor(model_name):
    """
    Initialize the BLIP processor and model.

    Args:
        model_name (str): The name of the pre-trained BLIP model.

    Returns:
        processor (BlipProcessor): The initialized BLIP processor.
        model (BlipModel): The initialized BLIP model.
    """
    try:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipModel.from_pretrained(model_name)
        model.eval()  # Set model to evaluation mode
        return processor, model
    except Exception as e:
        print(f"Error initializing model and processor: {e}")
        raise

def get_image_paths(base_path, target_category):
    """
    Traverse the directory structure to collect image file paths for the target category.

    Args:
        base_path (str): The base directory containing the images.
        target_category (str): The category to filter images (e.g., "Cars").

    Returns:
        list: A list of image file paths.
    """
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
                        if os.path.exists(product_path) and os.path.isdir(product_path):
                            for image_file in os.listdir(product_path):
                                image_full_path = os.path.join(product_path, image_file)
                                if is_image_file(image_full_path):
                                    image_paths.append(image_full_path)
    except Exception as e:
        print(f"Error while traversing directory structure: {e}")
    return image_paths

def is_image_file(file_path):
    """
    Check if a file is an image based on its extension.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file is an image, False otherwise.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    _, ext = os.path.splitext(file_path)
    return ext.lower() in image_extensions

def generate_texts(target_category):
    """
    Generate descriptive texts for the target category.

    Args:
        target_category (str): The category name.

    Returns:
        list: A list containing descriptive text strings.
    """
    return [f"This is a photo of a {target_category}."]

def process_image(image_path, texts, processor, model):
    """
    Process a single image using the BLIP model and print the results.

    Args:
        image_path (str): The path to the image file.
        texts (list): A list of text descriptions.
        processor (BlipProcessor): The BLIP processor.
        model (BlipModel): The BLIP model.
    """
    try:
        print(f"\nProcessing Image: {image_path}")
        image = Image.open(image_path).convert("RGB")

        inputs = processor(text=texts, images=image, return_tensors="pt", padding="max_length")

        for key, value in inputs.items():
            print(f"{key}: {value.shape}")

        with torch.no_grad():
            outputs = model(**inputs)
            # Assuming the model outputs logits or similar scores
            # Adjust based on the actual BLIP model's output structure
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for i, text in enumerate(texts):
                probability = probs[0][i] if probs.ndim > 1 else probs[0]
                print(f"{probability:.1%} probability that the image is '{text}'")
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

def main(base_path, target_category, model_name):
    """
    The main function to orchestrate image processing.

    Args:
        base_path (str): The base directory containing images.
        target_category (str): The category to filter images.
        model_name (str): The name of the pre-trained BLIP model.
    """
    try:
        processor, model = initialize_model_and_processor(model_name)
    except Exception:
        print("Failed to initialize the BLIP model and processor.")
        return

    image_paths = get_image_paths(base_path, target_category)
    if not image_paths:
        print("No images found to process.")
        return

    texts = generate_texts(target_category)
    for image_path in image_paths:
        process_image(image_path, texts, processor, model)

if __name__ == "__main__":
    # Parameters
    BASE_PATH = "/workspace/Retrieval_Experiment/random"
    TARGET_CATEGORY = "Cars"
    MODEL_NAME = "Salesforce/blip-image-captioning-base"  # Example BLIP model; replace with desired model

    main(BASE_PATH, TARGET_CATEGORY, MODEL_NAME)
