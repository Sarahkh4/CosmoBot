import os
import requests

def astronomy_image_agent(user_input: str):
    """Handles astronomy image requests using NASA APIs"""
    user_input_lower = user_input.lower()
    search_keywords = ["galaxy", "jupiter", "mars", "nebula", "saturn"]
    found_keyword = next((kw for kw in search_keywords if kw in user_input_lower), None)

    if found_keyword:
        return _handle_nasa_image_search(found_keyword)
    else:
        return _handle_apod_request()

def _handle_nasa_image_search(keyword: str):
    try:
        response = requests.get(
            "https://images-api.nasa.gov/search",
            params={"q": keyword, "media_type": "image"}
        )
        response.raise_for_status()
        items = response.json().get("collection", {}).get("items", [])
        
        if not items:
            return {"output": f"No images found for {keyword}."}
        
        first_item = items[0]
        return {"output": f"✨ {keyword.capitalize()} Image:\n" + _format_image_data(first_item)}
    
    except Exception as e:
        return {"output": f"Image search failed: {str(e)}"}

def _handle_apod_request():
    try:
        response = requests.get(
            f"https://api.nasa.gov/planetary/apod",
            params={"api_key": os.getenv("NASA_API_KEY")}
        )
        response.raise_for_status()
        data = response.json()
        return {"output": "✨ Astronomy Picture of the Day:\n" + _format_apod_data(data)}
    
    except Exception as e:
        return {"output": f"APOD Error: {str(e)}"}

def _format_image_data(item):
    # Helper function to format image data
    return (
        f"Title: {item.get('data', [{}])[0].get('title', 'Untitled')}\n"
        f"Description: {item.get('data', [{}])[0].get('description', 'No description')}\n"
        f"URL: {item.get('links', [{}])[0].get('href', 'No URL')}"
    )