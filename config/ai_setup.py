from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
import os

def configure_ai():
    """Configure AI models and tools"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    search_tool = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))
    
    return llm, search_tool

SYSTEM_PROMPT = '''
This system provides real-time astronomical data and educational content.
Key functions:
1. Tavily Search: Latest space news/research
2. NASA APIs: Images/events/ISS data
3. Educational resources from trusted sources
Guidelines:
- Ensure scientific accuracy
- Prioritize clear visualizations
- Maintain engaging interactions
'''