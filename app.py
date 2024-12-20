import pandas as pd
import numpy as np
from PIL import Image as pil_image
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr
from transformers import CLIPProcessor, CLIPModel
import torch


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
            text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)  # Normalize embeddings

    # Compute image embeddings if images are provided
    if images:
        image_inputs = processor(
            images=images,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            img_emb = model.get_image_features(pixel_values=image_inputs['pixel_values'])
            img_emb = img_emb.detach().cpu().numpy()
            img_emb = img_emb / np.linalg.norm(img_emb, axis=1, keepdims=True)  # Normalize embeddings

    return text_emb, img_emb


# Function to find similar embeddings
def find_similar(query_emb, embedding_matrix, top_k=10):
    """
    Find the top K most similar embeddings.

    Args:
        query_emb (np.ndarray): Query embedding.
        embedding_matrix (np.ndarray): Matrix of embeddings to search.
        top_k (int): Number of top results to return.

    Returns:
        list of tuples: (index, similarity score) for top K results.
    """
    similarities = cosine_similarity(query_emb, embedding_matrix)
    sorted_indices = np.argsort(similarities[0])[::-1]  # Sort by similarity

    unique_products = set()  # Track unique product IDs
    unique_scores = set()    # Track unique similarity scores
    results = []

    for idx in sorted_indices:
        product_id = df.iloc[idx]['ProductId']
        score = round(similarities[0][idx], 4)

        # Check for duplicates based on ProductId and Score
        if product_id not in unique_products and score not in unique_scores:
            unique_products.add(product_id)
            unique_scores.add(score)
            results.append((idx, score))

        # Stop when top_k unique results are collected
        if len(results) == top_k:
            break

    return results



# Gradio search function
def gradio_search(input_text=None, input_image=None):
    """
     Perform search based on text or image input using embeddings.

     Args:
         input_text (str): Text query for search.
         input_image (PIL.Image.Image): Image query for search.

     Returns:
         list of PIL.Image.Image, list of str: List of images and their descriptions as results.
     """

    if input_text:
        print(f"Searching using text: {input_text}")
        query_emb, _ = compute_embeddings(texts=[input_text])

    elif input_image:
        print("Searching using image...")
        input_image = input_image.convert("RGB")  # Ensure image is in RGB format
        _, query_emb = compute_embeddings(images=[input_image])

    else:
        return "Please provide either a text or an image input.", []

    results = find_similar(query_emb, text_emb if input_text else img_emb, top_k=7)

    # Collect images and create descriptions
    images = []
    descriptions = []

    for i, (idx, score) in enumerate(results):
        image_path = df.iloc[idx]['image']
        product_description = (f"{i + 1}. Product Name: {df.iloc[idx]['ProductTitle']}\n"
                               f"   Product Type: {df.iloc[idx]['ProductType']}\n"
                               f"   Gender: {df.iloc[idx]['Gender']}\n"
                               f"   Similarity Score: {score:.4f}")

        # Append image and structured description
        images.append(pil_image.open(image_path).convert("RGB"))
        descriptions.append(product_description)

    return images, "\n\n".join(descriptions)


# Gradio interface
interface = gr.Interface(
    fn=gradio_search,
    inputs=[
        gr.Textbox(label="Text", placeholder="Enter a description, e.g., 'Red T-shirt for boys'"),
        gr.Image(label="Image", type="pil")
    ],
    outputs=[
        gr.Gallery(label="Recommendation Items"),
        gr.Textbox(label="Descriptions", lines=5)
    ],
    title="Fashion Recommendation System",
    description="Search for recommendation items using text or image.",
)


if __name__ == "__main__":
    # Load saved embeddings and metadata
    text_emb = np.load("text_embeddings.npy")
    img_emb = np.load("image_embeddings.npy")
    df = pd.read_csv("processed_metadata.csv")

    # Load the model and processor
    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)

    # Set device (CPU or GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    interface.launch(share=True) #Set true to create a sharing link