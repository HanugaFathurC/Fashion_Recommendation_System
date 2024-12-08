import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch
from datasets import Dataset, Features, Value, Image
from PIL import Image as pil_image


if __name__ == "__main__":
    #Definition dataset path
    data_folder = "data"
    metadata_path = os.path.join(data_folder, "fashion.csv")

    #Load metadata
    df = pd.read_csv(metadata_path)

    #Combine relevant columns
    df['text'] = df.apply(lambda row: f"Gender: {row['Gender']}, Category: {row['Category']},"
                                      f" SubCategory: {row['SubCategory']}, ProductType: "
                                      f"{row['ProductType']}, Colour: {row['Colour']},"
                                      f" Usage: {row['Usage']}, ProductTitle: {row['ProductTitle']}",
                          axis=1)

    #Image
    df['image'] = df.apply(lambda row: f"{data_folder}/{row['Category']}/{row['ProductId']}.jpg", axis=1)

    print(df[['text', 'image']].head())

    # Function to preprocess images
    def preprocess_images(df, image_column='image', image_size=(224, 224)):
        processed_images = []
        for image_path in tqdm(df[image_column]):
            try:
                image = pil_image.open(image_path).convert('RGB')
                image = image.resize(image_size)
                processed_images.append(np.array(image))
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
        return np.array(processed_images)

    # Preprocess images
    data = preprocess_images(df)

    # Save processed data
    df.to_csv("processed_metadata.csv", index=False)
    np.save("processed_data.npy", df)

    print(f"Processed data shape: {df.shape}")

    # Define the features of the dataset
    features = Features({
        'text': Value('string'),
        'image': Image()
    })

    # Create the dataset
    dataset = Dataset.from_pandas(df[['text', 'image']], features=features)

    # Load the model and processor
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)

    # Set device (CPU or GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"Model loaded on device: {device}")

    def compute_embeddings_in_batches(texts=None, images=None, batch_size=16):
        """
        Compute embeddings for texts and/or images using the CLIP model.

        Args:
            texts (list of str): List of text inputs to encode.
            images (list of PIL.Image.Image): List of image inputs to encode.
            batch_size (int): Batch size for processing inputs.

        Returns:
            tuple: (text_embeddings, image_embeddings)
        """
        text_emb = img_emb = None

        if texts:
            text_emb = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                text_inputs = processor(
                    text=batch_texts,
                    padding=True,
                    return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    batch_emb = model.get_text_features(**text_inputs)
                    text_emb.append(batch_emb.cpu().numpy())
            text_emb = np.vstack(text_emb)

        if images:
            img_emb = []
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                image_inputs = processor(
                    images=batch_images,
                    return_tensors="pt"
                ).to(device)
                with torch.no_grad():
                    batch_emb = model.get_image_features(**image_inputs)
                    img_emb.append(batch_emb.cpu().numpy())
            img_emb = np.vstack(img_emb)

        return text_emb, img_emb


    # Compute embeddings for text and images
    text_emb, img_emb = compute_embeddings_in_batches(
        texts=dataset['text'],
        images=dataset['image'],
        batch_size=10
    )

    # Save the embedding
    np.save("text_embeddings.npy", text_emb)
    np.save("image_embeddings.npy", img_emb)

    print(f"Text embeddings shape: {text_emb.shape}")
    print(f"Image embeddings shape: {img_emb.shape}")

