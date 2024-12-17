import pandas as pd
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch
from datasets import Dataset, Features, Value, Image
from PIL import Image as pil_image
import matplotlib.pyplot as plt


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
            for i in tqdm(range(0, len(texts), batch_size), desc="Processing text batches"):
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
            for i in tqdm(range(0, len(images), batch_size), desc="Processing image batches"):
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

    # Function to compute_embeddings for testing
    def compute_embeddings(texts=None, images=None):
        """
        Compute embeddings for texts and/or images using the CLIP model.

        Args:
            texts (list of str): List of text inputs to encode.
            images (list of PIL.Image.Image): List of image inputs to encode.

        Returns:
            tuple: (text_embeddings, image_embeddings)
        """
        text_emb = img_emb = None

        # Compute text embeddings if texts are provided
        if texts:
            text_inputs = processor(
                text=texts,
                padding=True,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                text_emb = model.get_text_features(**text_inputs)
                text_emb = text_emb.detach().cpu().numpy()
                text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)

        # Compute image embeddings if images are provided
        if images:
            image_inputs = processor(
                images=images,
                return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                img_emb = model.get_image_features(pixel_values=image_inputs['pixel_values'])
                img_emb = img_emb.detach().cpu().numpy()
                img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)

        return text_emb, img_emb

    # Function to find similar embeddings
    def find_similar(query_emb, embedding_matrix, top_k=5):
        """
        Find the top K most similar embeddings.

        Args:
            query_emb (np.ndarray): Query embedding.
            embedding_matrix (np.ndarray): Matrix of embeddings to search.
            top_k (int): Number of top results to return.

        Returns:
            list of tuples: (index, similarity score) for top K results.
        """

        # Ensure query_emb and embedding_matrix are 2D arrays
        if query_emb.ndim == 1:
            query_emb = query_emb[np.newaxis, :]
        if embedding_matrix.ndim == 1:
            embedding_matrix = embedding_matrix[np.newaxis, :]

        # Convert to PyTorch tensors
        query_emb = torch.tensor(query_emb, dtype=torch.float32)
        embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)

        # Compute cosine similarity
        similarities = F.cosine_similarity(query_emb, embedding_matrix, dim=1)

        # Convert to NumPy for sorting
        similarities = similarities.cpu().numpy()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [(idx, similarities[idx]) for idx in top_indices]

    # Function to test model
    def test_model(query_text_test, query_image_path_test, top_k=5):
        # Load saved embeedings and metadata
        text_emb_load = np.load("text_embeddings.npy")
        img_emb_load = np.load("image_embeddings.npy")

        # Load query image
        query_image_test = pil_image.open(query_image_path_test)

        # Compute query embeddings
        query_text_emb_test, _ = compute_embeddings(texts=[query_text_test])
        _, query_img_emb_test = compute_embeddings(images=[query_image_test])

        # Find similar embeddings
        text_similar = find_similar(query_text_emb_test, text_emb_load, top_k=top_k)
        img_similar = find_similar(query_img_emb_test, img_emb_load, top_k=top_k)

        # Prepare to visualize similar images with similarity scores for text query
        fig1, axes1 = plt.subplots(1, top_k + 1, figsize=(20, 5))
        fig1.suptitle('Top Similar Images for Text Query with Similarity Scores')

        # Display query image or placeholder
        axes1[0].set_title("Query (Text)")
        axes1[0].text(0.5, 0.5, query_text_test, fontsize=12, wrap=True, ha='center', va='center')

        text_similar_names_scores = []
        for ax, (idx, score) in zip(axes1[1:], text_similar):
            similar_image_path = df.iloc[idx]['image']
            similar_image_name = os.path.basename(similar_image_path)
            text_similar_names_scores.append((similar_image_name, score))
            similar_img = pil_image.open(similar_image_path)
            ax.imshow(similar_img)
            ax.set_title(f"Score: {score:.4f}")
            ax.axis('off')

        # Prepare to visualize similar images with similarity scores for image query
        fig2, axes2 = plt.subplots(1, top_k + 1, figsize=(20, 5))
        fig2.suptitle('Top Similar Images for Image Query with Similarity Scores')
        axes2[0].set_title("Query (Image)")
        axes2[0].imshow(query_image_test)

        img_similar_names_scores = []
        for ax, (idx, score) in zip(axes2[1:], img_similar):
            similar_image_path = df.iloc[idx]['image']
            similar_image_name = os.path.basename(similar_image_path)
            img_similar_names_scores.append((similar_image_name, score))
            similar_img = pil_image.open(similar_image_path)
            ax.imshow(similar_img)
            ax.set_title(f"Score: {score:.4f}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()

        return text_similar_names_scores, img_similar_names_scores


    # Test the model
    query_text_test = "Red t-shirt boy"
    query_image_path_test = "data/Footwear/8526.jpg"
    text_similar_result, img_similar_result = test_model(query_text_test, query_image_path_test)
    print(f"Recommendations using Text Query: {text_similar_result}")
    print(f"Recommendations using Image Query: {img_similar_result}")


