import datetime
import json
import os
import time
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import END, StateGraph
from nasapy import Nasa
from pydantic import BaseModel
import requests
import streamlit as st
import tavily

# Load environment variables
load_dotenv()

# Initialize models and APIs
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def current_space_news():
    """
    Retrieves and formats the latest space-related news using Tavily search.
    
    Returns:
        dict: A structured response with key headlines and links.
    """
    try:
        # Use Tavily API to fetch real-time space news
        news_results = search.invoke({"query": "Latest space news, including recent discoveries, missions, and launches."})
        
        if not news_results:
            return {"output": "🚀 **Space News Update:** No recent news found at the moment. Check NASA, ESA, or SpaceX websites for updates."}

        # Format the top 3 space news articles
        formatted_news = "\n\n".join(
            [f"📰 **{article.get('title', 'Untitled')}**\n🔗 [Read More]({article.get('url', '#'}) )" for article in news_results[:3]]
        )

        return {"output": f"🚀 **Latest Space News:**\n\n{formatted_news}"}

    except Exception as e:
        return {"output": f"🚀 **Space News Update:** Unable to fetch news. Error: {str(e)}"}


def astronomy_image_agent(user_input: str):
    """
    Retrieves astronomy-related images from NASA's Image and Video Library or the Astronomy Picture of the Day (APOD) API.
    
    If the user query contains specific keywords (e.g., "galaxy", "jupiter", "mars", etc.), it fetches images
    from NASA's Image and Video Library using that keyword. If no specific keyword is found, it defaults to fetching 
    the Astronomy Picture of the Day (APOD).
    Args:
        user_input (str): The user input containing the query.
    Returns:
        dict: A dictionary containing details about the requested astronomy image, including the title, 
              description, and URL. If an error occurs, an error message is returned.
    """
    user_input_lower = user_input.lower()
    # List of keywords that trigger a search on NASA's Image and Video Library
    search_keywords = ["galaxy", "jupiter", "mars", "nebula", "saturn", "comet", "asteroid", "moon"]
    found_keyword = None
    for keyword in search_keywords:
        if keyword in user_input_lower:
            found_keyword = keyword
            break

    if found_keyword:
        # Query NASA's Image and Video Library API using the found keyword.
        search_url = "https://images-api.nasa.gov/search"
        params = {"q": found_keyword, "media_type": "image"}
        try:
            response = requests.get(search_url, params=params)
            response.raise_for_status()
            data = response.json()
            items = data.get("collection", {}).get("items", [])
            if items:
                first_item = items[0]
                links = first_item.get("links", [])
                image_url = links[0]["href"] if links else "No URL available"
                title = first_item.get("data", [{}])[0].get("title", "No Title")
                description = first_item.get("data", [{}])[0].get("description", "No Description")
                result = (
                    f"Title: {title}\n"
                    f"Description: {description}\n"
                    f"Image URL: {image_url}"
                )
                return {"output": f"✨ {found_keyword.capitalize()} Image Agent:\n\n{result}"}
            else:
                return {"output": f"No images found for {found_keyword}."}
        except Exception as e:
            return {"output": f"{found_keyword.capitalize()} image search failed: {str(e)}"}
    else:
        # Default to the APOD API if no specific keyword is detected.
        url = f"https://api.nasa.gov/planetary/apod?api_key={os.getenv('NASA_API_KEY')}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            result = (
                f"Title: {data.get('title', 'N/A')}\n"
                f"Date: {data.get('date', 'N/A')}\n"
                f"Explanation: {data.get('explanation', 'N/A')}\n"
                f"Image URL: {data.get('url', 'N/A')}\n"
                f"Media Type: {data.get('media_type', 'N/A')}\n"
                f"Copyright: {data.get('copyright', 'Public Domain')}"
            )
            return {"output": f"✨ Astronomy Image Agent:\n\n{result}"}
        except Exception as e:
            return {"output": f"Astronomy Image Agent Error: {str(e)}"}

def educational_resources_agent():
    """
    Retrieves and formats a list of high-quality educational resources on astronomy and space science. 
    Uses Tavily search to find trusted sources such as NASA, ESA, and other reputable platforms.
    Returns:
        dict: A dictionary containing an informative summary and a list of educational resources with titles, descriptions, and clickable links.
    Raises:
        Exception: If an error occurs while retrieving the resources or formatting the response.
    """
    try:
        # Use Tavily search to retrieve educational resources based on the query
        resources = search.invoke({
            "query": "Provide a curated list of high-quality educational resources on astronomy and space science, from trusted sources like NASA, ESA, and reputable educational platforms."
        })
        if not resources:
            return {"output": "No educational resources found. Please try again."}
        
        # Format each resource as a clickable link along with a description
        formatted_resources = "\n\n".join(
            [f"📘 **{r.get('title', 'Untitled Resource')}**\n🔗 [Visit Resource]({r.get('url', 'No URL provided')})" for r in resources]
        )
        
        # Add an informative introduction generated by LLM or manually written text
        informative_text = (
            "Based on your query, here is a curated list of high-quality educational resources on astronomy and space science. "
            "These resources have been carefully selected from trusted institutions such as NASA, ESA, and reputable academic platforms. "
            "They provide a wealth of knowledge on various space topics, including recent discoveries, research articles, and interactive educational materials."
        )
        
        full_response = f"🎓 **Educational Resources Agent:**\n\n{informative_text}\n\n{formatted_resources}"
        return {"output": full_response}
    except Exception as e:
        return {"output": f"Educational Resources Agent Error: {str(e)}"}

def iss_location_agent():
    """
    Fetches astronaut data and the current position of the International Space Station (ISS).
    
    It fetches the list of astronauts currently on the ISS, and retrieves the current position of the ISS.
    The data is updated at intervals and returned in a structured format.
    
    Returns:
        dict: A dictionary containing astronaut information and ISS position updates.
              Includes a list of people on the ISS and the positions over time.
    """
    # Fetch astronaut data
    api_url = "http://api.open-notify.org/astros.json"
    data = requests.get(api_url).json()

    result = {}

    if data['message'] == 'success':
        with open('iss-astros.json', 'w') as iss_file:
            json.dump(data, iss_file, indent=4)

        result["astronauts"] = f"There are currently {data['number']} people on the ISS."
        result["people"] = []
        for person in data['people']:
            result["people"].append(f"{person['name']} is currently on the ISS, Craft: {person['craft']}")
    else:
        result["error"] = 'Failed to obtain astronauts data.'

    # List to hold ISS position updates
    positions = []

    # fetching ISS position data
    for attempt in range(5):  # Retry 5 times
        try:
            data = requests.get("http://api.open-notify.org/iss-now.json", timeout=10).json()
            if data['message'] == 'success':
                # Extract ISS location
                location = data["iss_position"]
                latitude = float(location['latitude'])
                longitude = float(location['longitude'])

                position = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "time": str(datetime.datetime.fromtimestamp(data['timestamp']))
                }
                positions.append(position)
                break  # Successfully fetched data, break the loop
        except requests.exceptions.RequestException as e:
            result["error"] = str(e)
            time.sleep(3)  # Wait for 3 seconds before retrying

    result["positions"] = positions if positions else "Failed to retrieve ISS position data after retries."
    return result

search = TavilySearchResults(tavily_api_key=os.getenv("TAVILY_API_KEY"))

tools = [search, iss_location_agent, space_events_agent, astronomy_image_agent, educational_resources_agent]
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content='''This system is designed to provide real-time astronomical data, visualizations, and educational content. Below are the key functions and tools integrated into the system and their specific purposes:
1. **Tavily Search (`search`) Integration**:
   - **Purpose**: Provides users with up-to-date, relevant space-related information from Tavily's extensive search engine.
   - **Usage**: Enables the assistant to fetch the latest news, research articles, and educational resources related to space.
2. **NASA APOD Tool (`get_nasa_apod_tool`)**:
   - **Purpose**: Fetches the Astronomy Picture of the Day (APOD) from NASA's APOD API.
   - **Usage**: Provides the title, explanation, and image URL of the latest astronomy image shared by NASA, offering users daily insights into the wonders of space.
3. **NASA EONET Space Events (`fetch_nasa_eonet_events`)**:
   - **Purpose**: Retrieves real-time space-related events from NASA’s EONET API.
   - **Usage**: Provides details about ongoing or upcoming space events, including event type, title, location, and date, offering users information about significant celestial events.
4. **ISS Location Plotting (`plot_iss_location`)**:
   - **Purpose**: Displays the current real-time location of the International Space Station (ISS).
   - **Usage**: Visualizes the ISS position on a latitude-longitude graph, allowing users to track its orbit across the globe in real-time.
5. **ISS Astronaut Data (`ISS_data`)**:
   - **Purpose**: Provides data about astronauts currently aboard the ISS and their associated spacecraft.
   - **Usage**: Retrieves information on the number of astronauts aboard, their spacecraft, and visualizes the ISS’s position on a world map.
6. **Space Educational Fact (`educational_space_fact`)**:
   - **Purpose**: Fetches random educational facts about space from the Space Trivia API.
   - **Usage**: Provides fascinating trivia about celestial bodies, space phenomena, and other space-related topics, helping to educate users in an engaging manner.
7. **NEO Data Fetcher (`get_neo_data`)**:
   - **Purpose**: Retrieves real-time data on Near-Earth Objects (NEOs) from NASA’s NEO API.
   - **Usage**: Provides information on potentially hazardous asteroids and comets, helping users stay informed about objects that come close to Earth.
8. **Space News Agent (`retriever_tool`)**:
   - **Purpose**: Fetches the latest space news articles.
   - **Usage**: Provides users with up-to-date news articles related to space exploration, satellite launches, and astronomical discoveries.
### Workflow:
- The system responds dynamically to user queries by identifying the appropriate tool for each request.  
- If the user asks for data or visualizations related to astronomical objects or space events, the relevant NASA APIs or visualization tools are invoked.  
- For general space education or research, tools like Tavily and Space Trivia provide comprehensive, up-to-date educational content.  
- The system remembers the context of conversations to ensure relevant and seamless interactions with users, supported by the `MemorySaver`.
### Guidelines:
- **Scientific Accuracy**: All provided information is scientifically accurate and up-to-date.
- **Engagement**: Educational content is easy to understand and presented in an interactive and engaging way.
- **Visualization**: For requests requiring visual representations (such as ISS tracking or NEO data), interactive orbital plots or maps are rendered for clarity.
- **Tool Alignment**: The system uses the most appropriate tool based on the user’s request, ensuring minimal invocation of unnecessary functions.
This setup ensures that the assistant can efficiently address a wide range of user queries related to space, astronomy, and related educational content while keeping interactions intuitive and engaging.
''')
# Node
def assistant(state: MessagesState) -> MessagesState:
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"][-10:])]}

# Build graph
builder: StateGraph = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
memory: MemorySaver = MemorySaver()
# Compile the workflow into an application
react_graph_memory: CompiledStateGraph = builder.compile(checkpointer=memory)

st.title("🌠CosmoBot🌠: Unveiling the mysteries of space")
st.caption("Your AI-powered space exploration assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["content"].startswith("http"):
            st.image(message["content"])
        else:
            st.write(message["content"])

# User input via chat
if user_input := st.chat_input("Ask about space news, ISS location, Educational resource or astronomy images"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    try:
        # Prepare the messages for the model
        messages = [HumanMessage(content=user_input)]

        # Call the memory and invoke the response
        response = react_graph_memory.invoke({"messages": messages}, config={"configurable": {"thread_id": "1"}})
        assistant_response = response["messages"][-1].content

        # Use regex to detect image URLs in the assistant response
        image_pattern = re.compile(r'(https?://\S+\.(?:png|jpg|jpeg|gif))', re.IGNORECASE)
        match = image_pattern.search(assistant_response)
        if match:
            image_url = match.group(1)
            st.session_state.messages.append({"role": "assistant", "content": image_url})
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.image(image_url)
                st.write(assistant_response)
        else:
            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            with st.chat_message("assistant"):
                st.write(assistant_response)

    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        with st.chat_message("assistant"):
            st.write(error_msg)
