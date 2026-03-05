import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
import requests 

# Load variables from .env file
load_dotenv()
# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "mental-health-index"
LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
LLMOD_BASE_URL = "https://api.llmod.ai/v1"
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

SAFE_CHANNELS = [
    "UC9Yd6X-7Pz_F_J7vR6gX86Q", # Therapy in a Nutshell (Emma McAdam)
    "UCzBYOHyEEzlkRdDOSobbpvw", # Kati Morton
    "UCl8TEoIOnMq_5ntJOYMq-Zg", # Dr. Julie Smith
    "UClHVl2N3jPEbkNJVx-ItQIQ", # HealthyGamerGG (Dr. K)
    "UC_zQoiPtBDvsThGroagm3ww", # Psych Hub
    "UCvYVvA5Hn9nS6S_GgXv6gkA", # Psych2Go
    "UCisQYxK8L6v_9_oM7W1t6uA", # Headspace
    "UCh6HDKcLwJioBBRSprqfezA"  # The Anxiety Guy
]
    
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
client = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)


def get_query_embedding(text):
    response = client.embeddings.create(
        input=[text],
        model="RPRTHPB-text-embedding-3-small"
    )
    return response.data[0].embedding


def retrieve_context(user_query):
    query_vector = get_query_embedding(user_query)
    results = index.query(vector=query_vector, top_k=2, include_metadata=True)
    contexts = [match['metadata']['response'] for match in results['matches']]
    return "\n---\n".join(contexts)


def search_youtube_safely(query):

    if not YOUTUBE_API_KEY:
        return None
    
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,          
        "maxResults": 10,    
        "type": "video",
        "key": YOUTUBE_API_KEY
    }
    
    try:
        response = requests.get(url, params=params).json()
        if "items" in response:
            for item in response["items"]:
                channel_id = item["snippet"]["channelId"]
                # בדיקה האם הערוץ נמצא ברשימה הלבנה
                if channel_id in SAFE_CHANNELS:
                    video_id = item["id"]["videoId"]
                    title = item["snippet"]["title"]
                    return {
                        "url": f"https://www.youtube.com/watch?v={video_id}", 
                        "title": title,
                        "channel": item["snippet"]["channelTitle"]
                    }
    except Exception as e:
        print(f"YouTube API Error: {e}")
        return None
    return None


def mental_health_agent_autonomous(messages_history):
    steps = []
    max_retries = 2
    user_input_text = messages_history[-1]["content"] if messages_history else ""

    try:
        # --- STEP 1: BRAIN ---
        brain_system_prompt = (
            "You are an autonomous Mental Health Strategist. Decide how to help. "
            "Options: \n"
            "1. SEARCH_DB: If user needs professional facts or coping text. \n"
            "2. SEARCH_MEDIA: If user needs music, a visual exercise, or a psychological explanation. \n"
            "3. ANSWER: For general support. \n"
            "Format: Thought: <reasoning> Decision: <ACTION>"
        )
        
        brain_response = client.chat.completions.create(
            model="RPRTHPB-gpt-5-mini",
            messages=messages_history + [{"role": "system", "content": brain_system_prompt}],
            temperature=1
        ).choices[0].message.content

        steps.append({
            "module": "Smart/Generate LLM",
            "prompt": "Reasoning about the best course of action",
            "response": brain_response
        })

        # --- STEP 2: ACTION / KNOWLEDGE ACQUISITION ---
        context = ""
        media_link = ""
        
        if "SEARCH_DB" in brain_response.upper():
            context = retrieve_context(user_input_text)
            steps.append({
                "module": "Database",
                "prompt": user_input_text,
                "response": "Observation: Professional context retrieved from vector store."
            })
        
        elif "SEARCH_MEDIA" in brain_response.upper():
            search_query_prompt = f"Based on the user's distress: '{user_input_text}', what is the best type of video to help them? Give me ONLY the search query."
            search_query = client.chat.completions.create(
                model="RPRTHPB-gpt-5-mini",
                messages=[{"role": "user", "content": search_query_prompt}]
            ).choices[0].message.content
            
            steps.append({
                "module": "Smart/Generate LLM",
                "prompt": search_query_prompt,
                "response": f"Generated Query: {search_query}"
            })

            media_data = search_youtube_safely(search_query)
            if media_data:
                media_link = media_data["url"]
                context = f"Recommended Video: '{media_data['title']}'. Link: {media_link}"
                steps.append({
                    "module": "Smart/Generate LLM",
                    "prompt": search_query,
                    "response": f"Observation: Found safe video - {media_data['title']}, Link: {media_link}"
                })
            else:
                context = "No safe professional videos found for this specific query."
                steps.append({
                    "module": "Smart/Generate LLM",
                    "prompt": search_query,
                    "response": "Observation: No matching safe content found."
                })

        # --- STEP 3 & 4: GENERATOR & REFLECTOR ---
        attempts = 0
        while attempts < max_retries:
            system_instructions = (
                "You are a professional Mental Health Assistant. "
                f"Context for this session: {context}"
            )

            current_draft = client.chat.completions.create(
                model="RPRTHPB-gpt-5-mini",
                messages=[{"role": "system", "content": system_instructions}] + messages_history,
                temperature=1
            ).choices[0].message.content

            steps.append({
                "module": "Smart/Generate LLM",
                "prompt": "Drafting response with context",
                "response": current_draft
            })

            # Reflection
            reflect_prompt = f"Critique this response. Is the advice and the media link (if any) safe and helpful? Answer 'SAFE' or 'CRITIQUE':\n{current_draft}"
            reflection = client.chat.completions.create(
                model="RPRTHPB-gpt-5-mini",
                messages=[{"role": "user", "content": reflect_prompt}],
                temperature=1
            ).choices[0].message.content

            steps.append({
                "module": "Reflect LLM",
                "prompt": reflect_prompt,
                "response": reflection
            })

            if "SAFE" in reflection.upper():
                return current_draft, steps
            attempts += 1

        return "I'm here for you, but I want to make sure I give you the best possible support. If you're feeling overwhelmed, please speak with a professional.", steps

    except Exception as e:
        error_msg = str(e)
        steps.append({"module": "Error", "response": error_msg})
        return f"A system error occurred: {error_msg}", steps



# --- Example Execution ---
if __name__ == "__main__":
    # --- Example Execution ---
    if __name__ == "__main__":
        # Simulate a conversation history
        # This is what the Website/Frontend will send to the API
        history = [
            {"role": "user", "content": "Hi, I'm Oded."},
            {"role": "assistant", "content": "Hi Oded, how can I help you today?"},
            {"role": "user", "content": "i am thinking to suicide"}
        ]

        # Now we pass the WHOLE history to the agent
        final_answer, trace_steps = mental_health_agent_autonomous(history)

        print("\n" + "=" * 50)
        print("FINAL RESPONSE TO USER:")
        print(final_answer)
        print("=" * 50 + "\n")

        print("AGENT EXECUTION LOGS (STEPS):")
        for i, step in enumerate(trace_steps, 1):
            print(f"Step {i}: [{step['module']}]")
            print(f"--- Response/Thought: {step['response']}")
            print("-" * 20)
