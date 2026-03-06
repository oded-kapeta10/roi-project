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

# Checked that those are valid channels
PREFERRED_CHANNELS = {
    "THERAPY": ["UC9Yd6X-7Pz_F_J7vR6gX86Q", "UCzBYOHyEEzlkRdDOSobbpvw", "UCl8TEoIOnMq_5ntJOYMq-Zg", "UC_zQoiPtBDvsThGroagm3ww"],
    "MEDITATION": ["UCisQYxK8L6v_9_oM7W1t6uA", "UCOY83Z7f_N0o0Tz96_H6FkA", "UCvYVvA5Hn9nS6S_GgXv6gkA"],
    "MUSIC": ["UC_z679N2K_T5L1S5XG0A6Jg", "UC_SjW7NIdYm2y8p_qX8U_pA", "UCSJ4gkVC6NrvII8umztf0Ow"]
}
ALL_PREFERRED_IDS = [idx for cat in PREFERRED_CHANNELS.values() for idx in cat]

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

def verify_video_metadata(title, description, channel_id):
    """
    Check if a video is safe. If it is from the PREFERRED_CHANNELS, then it's safe; otherwise, the reflect LLM checks its metadata to decide.
    """
    if channel_id in ALL_PREFERRED_IDS:
        return True, "Verified Source"

    check_prompt = (
        f"Analyze this YouTube video for a mental health assistant.\n"
        f"Title: {title}\n"
        f"Description: {description}\n\n"
        "Criteria for SAFE (Be generous/lenient):\n"
        "- Is the content generally supportive, kind, or harmless? (Mark SAFE unless it's clearly toxic).\n"
        "- Is it free from obvious 'trolling', pranks, or dangerous medical misinformation?\n"
        "Answer ONLY 'SAFE' or 'UNSAFE'. If UNSAFE, explain why in 5 words."
    )
    
    try:
        response = client.chat.completions.create(
            model="RPRTHPB-gpt-5-mini",
            messages=[{"role": "user", "content": check_prompt}],
            temperature=0
        ).choices[0].message.content
        return "SAFE" in response.upper(), response.strip() # מחזיר גם את הנימוק
    except:
        return False, f"API Error: {str(e)}"
        
def search_youtube_autonomously(query):
    attempts_log = []
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key: 
        return None, [{"error": "YOUTUBE_API_KEY is missing from environment"}]
    
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {"part": "snippet", "q": query, "maxResults": 10, "type": "video", "key": api_key}
    
    try:
        response = requests.get(url, params=params).json()
        if "items" in response:
            for item in response["items"]:
                title = item["snippet"]["title"]
                desc = item["snippet"]["description"]
                channel_id = item["snippet"]["channelId"]
                
                is_safe, reason = verify_video_metadata(title, desc, channel_id)
                
                attempts_log.append({
                    "title": title,
                    "status": "SAFE" if is_safe else "UNSAFE",
                    "reason": reason
                })
                
                if is_safe:
                    return {
                        "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}", 
                        "title": title,
                        "channel": item["snippet"]["channelTitle"],
                        "v_method": reason
                    }, attempts_log 
                    
    except Exception as e:
        return None, [{"error": str(e)}]
        
    return None, attempts_log


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
            category_prompt = (
                f"User is feeling: '{user_input_text}'. Select: THERAPY, MEDITATION, or MUSIC."
            )
            selected_category = client.chat.completions.create(
                model="RPRTHPB-gpt-5-mini",
                messages=[{"role": "user", "content": category_prompt}]
            ).choices[0].message.content
            steps.append({
                "module": "Smart/Generate LLM",
                "prompt": "Deciding on media category",
                "response": f"Selected Category: {selected_category}"
            })
            search_query_prompt = (
                f"The user is feeling: '{user_input_text}'. "
                f"Act as a person searching YouTube for {selected_category} content. "
                "Create a VERY SHORT search query (3-5 words max). "
                "Examples: 'calming lofi stress', '5 minute breathing exercise', 'anxiety relief music'. "
                "Give me ONLY the search query text."
            )
            search_query = client.chat.completions.create(
                model="RPRTHPB-gpt-5-mini",
                messages=[{"role": "user", "content": search_query_prompt}]
            ).choices[0].message.content

            steps.append({
                "module": "Smart/Generate LLM",
                "prompt": "Generating simplified YouTube query",
                "response": f"Generated Query: {search_query}"
            })

            media_data, attempts_log = search_youtube_autonomously(search_query)
            steps.append({
                "module": "YouTube Tool Debug",
                "prompt": f"Analyzing candidates for: {search_query}",
                "response": attempts_log 
            })
            if media_data:
                media_link = media_data["url"]
                context = f"Recommended: '{media_data['title']}' (Source: {media_data['channel']}, Method: {media_data['v_method']}). Link: {media_link}"
                steps.append({
                    "module": "YouTube Tool",
                    "prompt": search_query,
                    "response": f"Found video: {media_data['title']} ({media_data['v_method']})"
                })
                
            else:
                context = "No sufficiently safe media found."
                steps.append({"module": "YouTube Tool", "prompt": search_query, "response": "No safe content found."})

        # --- STEP 3 & 4: GENERATOR & REFLECTOR ---
        attempts = 0
        while attempts < max_retries:
            system_instructions = (
                "You are a professional Mental Health Assistant. "
                "BE CONCISE. If a media link is provided in the context, share it and explain why it helps. "
                "If NO media was found (context says 'No safe media found'), do NOT provide long generic lists of songs. "
                "Instead, apologize briefly and ask the user for a more specific mood or genre."
                f"\nContext: {context}"
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
