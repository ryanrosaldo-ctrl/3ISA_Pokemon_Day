"""
Pokemon Image Handler
Fetches and caches Pokemon images from PokéAPI
"""

import os
import requests
import json
import base64
from pathlib import Path
import streamlit as st

# Image cache directory
IMAGE_CACHE_DIR = "pokemon_images"

def ensure_cache_dir():
    """Create image cache directory if it doesn't exist."""
    Path(IMAGE_CACHE_DIR).mkdir(exist_ok=True)

def get_image_url_from_api(pokemon_id: int) -> str:
    """
    Fetch the official artwork image URL from PokéAPI.
    Returns the official-artwork URL for high-quality images.
    """
    try:
        url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}/"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            # Try to get official artwork first, then fall back to other images
            image_url = data.get("sprites", {}).get("other", {}).get("official-artwork", {}).get("front_default")
            if not image_url:
                # Fallback to front_default
                image_url = data.get("sprites", {}).get("front_default")
            return image_url
    except Exception as e:
        print(f"Error fetching image URL for Pokemon {pokemon_id}: {e}")
    return None

def download_and_cache_image(pokemon_id: int, pokemon_name: str) -> str:
    """
    Download Pokemon image and cache it locally.
    Returns the local file path.
    """
    ensure_cache_dir()
    
    # Check if already cached
    cache_file = os.path.join(IMAGE_CACHE_DIR, f"{pokemon_id}_{pokemon_name.lower()}.png")
    if os.path.exists(cache_file):
        return cache_file
    
    try:
        image_url = get_image_url_from_api(pokemon_id)
        if not image_url:
            return None
        
        # Download the image
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            return cache_file
    except Exception as e:
        print(f"Error downloading image for {pokemon_name} ({pokemon_id}): {e}")
    
    return None

@st.cache_data
def get_pokemon_image_path(pokemon_id: int, pokemon_name: str) -> str:
    """
    Get the local path to a Pokemon image, downloading if necessary.
    Cached to avoid repeated downloads.
    """
    return download_and_cache_image(pokemon_id, pokemon_name)

def get_pokemon_image_url(pokemon_id: int, pokemon_name: str) -> str:
    """
    Get the image URL for a Pokemon (either cached or from PokéAPI).
    Returns the URL that can be used in HTML img tags or Streamlit image display.
    """
    image_path = get_pokemon_image_path(pokemon_id, pokemon_name)
    
    if image_path and os.path.exists(image_path):
        # For cached images, convert to base64 for embedding in HTML
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{image_data}"
    else:
        # Fallback to PokéAPI URL
        image_url = get_image_url_from_api(pokemon_id)
        return image_url if image_url else None

def get_pokemon_image_html(pokemon_id: int, pokemon_name: str, size: str = "medium") -> str:
    """
    Generate HTML img tag for a Pokemon image.
    Size options: 'small' (60px), 'medium' (100px), 'large' (150px)
    """
    size_map = {
        "small": "60px",
        "medium": "100px",
        "large": "150px"
    }
    
    img_size = size_map.get(size, "100px")
    image_url = get_pokemon_image_url(pokemon_id, pokemon_name)
    
    if image_url:
        return f'<img src="{image_url}" alt="{pokemon_name}" style="width: {img_size}; height: {img_size}; object-fit: contain; filter: drop-shadow(0 0 4px rgba(255, 203, 5, 0.3));">'
    else:
        # Placeholder if image not found
        return f'<div style="width: {img_size}; height: {img_size}; background: rgba(59, 76, 202, 0.3); border: 2px dashed #3b4cca; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: #a5b4fc; font-size: 0.8rem;">No Image</div>'

def preload_team_images(team_df):
    """
    Preload images for a team of Pokemon to avoid delays during display.
    """
    for _, row in team_df.iterrows():
        pokemon_id = row.get("id") or row.get("pokemon_id")
        if pokemon_id is not None:
            get_pokemon_image_path(int(pokemon_id), row["name"])
