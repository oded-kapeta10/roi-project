from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import os

from agent_logic import mental_health_agent_autonomous
app = Flask(__name__)
CORS(app)

from flask import render_template

@app.route('/')
def index():
    return render_template('index.html')

# --- Mandatory Endpoint 1: Team Info ---
@app.route('/api/team_info', methods=['GET'])
def get_team_info():
    """
    Returns group and student details for the project submission.
    """
    return jsonify({
        "group_batch_order_number": "Batch3_Order5",  # Update with your specific batch/order
        "team_name": "ROI",
        "students": [
            {"name": "Oded Kapeta", "email": "odedkapeta@campus.technion.ac.il"},
            {"name": "Rona Lavi", "email": "rona.lavi@campus.technion.ac.il"},
            {"name": "Itay Davidovich-Gross", "email": "idavidovich@campus.technion.ac.il"}
        ]
    })


# --- Mandatory Endpoint 2: Agent Info ---
@app.route('/api/agent_info', methods=['GET'])
def get_agent_info():
    return jsonify({
        "description": "An autonomous mental health support agent utilizing a ReAct architecture for reasoning and a Reflection loop for safety.",
        "purpose": "To provide empathetic, safe, and context-aware guidance for mental well-being and stress management.",
        "prompt_template": {
            "template": "Input: {user_prompt}\nContext: {history}"
        },
        "prompt_examples": [
            {
                "prompt": "I'm feeling overwhelmed with my exams at the Technion.",
                "full_response": "Hi Oded, I'm sorry to hear that. It's very common to feel this way. Let's try a grounding exercise.",
                "steps": [
                    {
                        "module": "Smart/Generate LLM",
                        "prompt": "Analyze user stress levels and history.",
                        "response": "Thought: User is a Technion student. High stress detected. No crisis. Decision: ANSWER."
                    },
                    {
                        "module": "Reflect LLM",
                        "prompt": "Is the response safe?",
                        "response": "SAFE"
                    }
                ]
            }
        ]
    })


# --- C) GET /api/model_architecture [cite: 52] ---
@app.route('/api/model_architecture', methods=['GET'])
def get_architecture():
    # This must return a PNG image[cite: 53, 61].
    # Ensure you have an architecture.png file in your directory.
    try:
        return send_file('architecture.png', mimetype='image/png')  # [cite: 62]
    except Exception as e:
        return jsonify({"status": "error", "error": "Architecture image not found"}), 404


# --- D) POST /api/execute [cite: 63] ---
@app.route('/api/execute', methods=['POST'])
def execute_agent():
    data = request.json
    user_prompt = data.get('prompt')

    # Extract history from the JSON body.
    # If not provided, fall back to a single message list.
    history = data.get('history', [{"role": "user", "content": user_prompt}])

    try:
        # Pass the entire conversation history to the agent logic
        final_answer, trace_steps = mental_health_agent_autonomous(history)

        return jsonify({
            "status": "ok",
            "error": None,
            "response": final_answer,
            "steps": trace_steps
        })
    except Exception as e:
        error_msg = str(e)
        # Check if the error is related to safety filters
        if "content_policy_violation" in error_msg.lower() or "ResponsibleAI" in error_msg:
            friendly_error = "I'm sorry, but I cannot fulfill this request due to safety policy restrictions. Please reach out to professional help if you are in distress."
        else:
            friendly_error = "A system error occurred. Please try again later."

        return jsonify({
            "status": "error",
            "error": friendly_error,
            "response": None,
            "steps": []
        })


if __name__ == "__main__":
    # Render מספקת את הפורט דרך משתנה סביבה. אם הוא לא קיים, נשתמש ב-5000
    port = int(os.environ.get("PORT", 5000))
    # חשוב מאוד: ה-host חייב להיות 0.0.0.0
    app.run(host="0.0.0.0", port=port)