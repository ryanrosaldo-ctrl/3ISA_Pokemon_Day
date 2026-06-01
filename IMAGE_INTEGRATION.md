# Pokémon Image Integration

## Overview
The Streamlit app now displays Pokémon images in team cards, fetched from PokéAPI and cached locally for performance.

## Features

### Image Retrieval
- **Source**: PokéAPI official artwork images
- **Quality**: High-resolution official artwork (preferred) with fallback to standard sprites
- **Caching**: Images are downloaded once and cached locally in the `pokemon_images/` directory

### Performance Optimization
- **Lazy Loading**: Images are preloaded before display to avoid delays
- **Local Cache**: Subsequent loads use cached images instead of re-downloading
- **Streamlit Caching**: Uses `@st.cache_data` decorator for efficient caching

### Display Locations
Images are displayed in:
1. **Team Engine** - Gym Leader team cards (3-column grid)
2. **Challenger Selection** - Recommended counter team cards (3-column grid)

## How It Works

### Image Handler Module (`pokemon_image_handler.py`)
- `get_image_url_from_api()` - Fetches image URL from PokéAPI
- `download_and_cache_image()` - Downloads and stores image locally
- `get_pokemon_image_path()` - Returns cached image path (with Streamlit caching)
- `get_pokemon_image_html()` - Generates HTML img tag for display
- `preload_team_images()` - Preloads all team images before rendering

### Integration Points
1. Import at top of `app.py`:
   ```python
   from pokemon_image_handler import get_pokemon_image_html, preload_team_images
   ```

2. Preload images before display:
   ```python
   preload_team_images(team_df)
   ```

3. Generate image HTML in card display:
   ```python
   image_html = get_pokemon_image_html(pokemon_id, pokemon_name, size="medium")
   ```

## Image Sizes
- `"small"` - 60px (for compact displays)
- `"medium"` - 100px (default for team cards)
- `"large"` - 150px (for detailed views)

## Fallback Behavior
- If image download fails, displays PokéAPI URL directly
- If both fail, shows a placeholder "No Image" box
- Gracefully handles missing Pokémon IDs

## Cache Directory
Images are stored in: `3ISA_Pokemon_Day/pokemon_images/`
- Format: `{pokemon_id}_{pokemon_name}.png`
- Example: `252_treecko.png`

## Requirements
- `requests` library (for HTTP requests to PokéAPI)
- `streamlit` (already required)
- `pandas` (already required)

## Performance Notes
- First load: ~2-3 seconds per team (downloading 6 images)
- Subsequent loads: <100ms (using cache)
- Images are ~50-100KB each
- Total cache size for all Pokémon: ~15-20MB

## Troubleshooting

### Images not showing
1. Check internet connection (needed for first download)
2. Verify `pokemon_images/` directory exists
3. Check browser console for image loading errors

### Slow performance
1. First load is slower due to downloads - this is normal
2. Subsequent loads should be instant
3. Clear cache if images seem outdated: delete `pokemon_images/` folder

### Missing images for specific Pokémon
- Some Pokémon may not have official artwork on PokéAPI
- Fallback to standard sprite is automatic
- Check PokéAPI directly if unsure: https://pokeapi.co/api/v2/pokemon/{id}
