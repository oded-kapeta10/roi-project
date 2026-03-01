import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI

# Load variables from .env file
load_dotenv()
# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "mental-health-index"
LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
LLMOD_BASE_URL = "https://api.llmod.ai/v1"

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




def mental_health_agent_autonomous(messages_history):
    steps = []
    max_retries = 2
    # Safe extraction of last message
    user_input_text = messages_history[-1]["content"] if messages_history else ""

    try:
        # --- STEP 1: BRAIN ---
        # Note: Temperature is set to 1 as requested
        brain_response = client.chat.completions.create(
            model="RPRTHPB-gpt-5-mini",
            messages=messages_history + [{"role": "system",
                                          "content": "Task: Decide if you need to search professional advice. Format: Thought: <reasoning> Decision: <SEARCH or ANSWER>"}],
            temperature=1
        ).choices[0].message.content

        steps.append({
            "module": "Smart/Generate LLM",
            "prompt": "Reasoning about search necessity based on history",
            "response": brain_response
        })

        # --- STEP 2: KNOWLEDGE BASE ---
        context = ""
        is_search_needed = "DECISION: SEARCH" in brain_response.upper()

        if is_search_needed:
            context = retrieve_context(user_input_text)
            steps.append({
                "module": "Database",
                "prompt": user_input_text,
                "response": "Observation: Context retrieved from vector store."
            })
        else:
            steps.append({
                "module": "Database",
                "prompt": user_input_text,
                "response": "Observation: Search skipped."
            })

        # --- STEP 3 & 4: GENERATOR & REFLECTOR ---
        attempts = 0
        while attempts < max_retries:
            system_instructions = (
                "You are a professional Mental Health Assistant. "
                f"Use this retrieved context if relevant: {context}"
            )

            # Note: Temperature is set to 1
            current_draft = client.chat.completions.create(
                model="RPRTHPB-gpt-5-mini",
                messages=[{"role": "system", "content": system_instructions}] + messages_history,
                temperature=1
            ).choices[0].message.content

            steps.append({
                "module": "Smart/Generate LLM",
                "prompt": f"Generating response with system instructions",
                "response": current_draft
            })

            # Reflection step - Note: Temperature is set to 1
            reflect_prompt = f"Is this response safe and helpful? Answer 'SAFE' or 'CRITIQUE':\n{current_draft}"
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

        return "I apologize, but I am having trouble. Please consult a professional.", steps

    except Exception as e:
        error_msg = str(e)
        # Check if it's a safety/policy error from the API
        if "content_policy_violation" in error_msg or "ResponsibleAI" in error_msg:
            friendly_message = "I cannot fulfill this request due to safety policy restrictions. If you are in distress, please contact professional help."
        else:
            friendly_message = f"A system error occurred: {error_msg}"

        return friendly_message, steps



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