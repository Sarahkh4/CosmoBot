import os
import requests
from datetime import datetime

def space_events_agent():
    """Fetches upcoming space events from NASA's EONET API"""
    base_url = "https://eonet.gsfc.nasa.gov/api/v2.1/events"
    params = {"api_key": os.getenv("NASA_API_KEY")}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        events = response.json().get("events", [])
        
        if not events:
            return {"output": "🚀 **Space Events Agent:** No events found."}
        
        event_details = []
        for event in events:
            title = event.get('title', 'N/A')
            category = (event.get('categories', [{}])[0]).get('title', 'Unknown')
            geometries = event.get('geometries', [])
            
            if geometries:
                first_geometry = geometries[0]
                coordinates = first_geometry.get('coordinates', 'Unknown')
                event_date = first_geometry.get('date', 'Unknown')
            else:
                coordinates = 'Unknown'
                event_date = 'Unknown'
            
            event_details.append(
                f"**Event:** {title}\n"
                f"**Category:** {category}\n"
                f"**Location:** {coordinates}\n"
                f"**Date:** {event_date}"
            )
        
        return {"output": f"🚀 **Space Events Agent:**\n\n" + "\n\n".join(event_details)}
    
    except Exception as e:
        return {"output": f"🚀 **Space Events Agent Error:** {str(e)}"}