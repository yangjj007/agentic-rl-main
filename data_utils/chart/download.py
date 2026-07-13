from datasets import load_dataset
import json
import os
from PIL import Image

from data_utils.paths import CHARTQA_DIR, CHARTQA_IMAGES_DIR, CHARTQA_JSON_DIR


def save_chartqa_with_absolute_paths(base_output_dir=None):
    """
    Load the HuggingFaceM4/ChartQA dataset.
    For each PIL image, generate a unique filename, save the image locally,
    and create a JSON metadata file that contains absolute image paths.
    """
    if base_output_dir is None:
        base_output_dir = CHARTQA_DIR

    base_dir_abs = os.path.abspath(base_output_dir)
    image_output_dir = os.path.join(base_dir_abs, "images")
    json_output_dir = os.path.join(base_dir_abs, "json")

    os.makedirs(image_output_dir, exist_ok=True)
    os.makedirs(json_output_dir, exist_ok=True)

    print(f"Images will be saved to (absolute path): {image_output_dir}")
    print(f"JSON will be saved to (absolute path): {json_output_dir}")

    print("Loading HuggingFaceM4/ChartQA dataset (this may take some time)...")
    try:
        dataset = load_dataset("HuggingFaceM4/ChartQA")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Dataset splits: {list(dataset.keys())}")

    for split in dataset.keys():
        print(f"\n--- Processing split {split} ---")

        metadata_list = []
        total_count = len(dataset[split])

        for i, example in enumerate(dataset[split]):
            question = example["query"]
            answer = example["label"][0]
            pil_image = example["image"]

            generated_filename = f"{split}_{i:06d}.png"
            image_save_path = os.path.join(image_output_dir, generated_filename)

            if not os.path.exists(image_save_path):
                try:
                    pil_image.save(image_save_path)
                except Exception as e:
                    print(f"Failed to save image {image_save_path}: {e}")
                    continue

            metadata_list.append({
                "question": question,
                "question_wo_prompt": question,
                "answer": answer,
                "image": image_save_path,
                "human_or_machine": example.get("human_or_machine", 0)
            })

            if (i + 1) % 500 == 0 or (i + 1) == total_count:
                print(f"  Processed {split} split: {i + 1} / {total_count}")

        json_filename = os.path.join(json_output_dir, f"{split}.json")
        print(f"Saving {len(metadata_list)} metadata entries to {json_filename}...")

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)

    print(f"\n--- Processing completed! ---")
    print(f"All image files have been saved in: '{image_output_dir}'")
    print(f"All JSON files have been saved in: '{json_output_dir}'")


if __name__ == "__main__":
    Image.MAX_IMAGE_PIXELS = None
    save_chartqa_with_absolute_paths()
