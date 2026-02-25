from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
from agent_logic import mental_health_agent_autonomous
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "message": "Welcome to the Mental Health AI Agent API",
        "endpoints": {
            "team_info": "/api/team_info",
            "agent_info": "/api/agent_info",
            "model_architecture": "/api/model_architecture",
            "execute": "/api/execute (POST)"
        },
        "status": "online"
    })

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
@app.route('/api/execute', methods=['POST'])
def execute_agent():
    try:
        data = request.json

        # Check if the input contains a 'messages' list (for back-and-forth conversation)
        if 'messages' in data and isinstance(data['messages'], list):
            messages_history = data['messages']
        # If it only contains a single 'prompt' string (for simple one-off requests)
        elif 'prompt' in data:
            messages_history = [{"role": "user", "content": data['prompt']}]
        else:
            return jsonify({
                "status": "error",
                "error": "Invalid input: 'messages' list or 'prompt' string required.",
                "response": None,
                "steps": []
            }), 400

        # Now we pass the full history to your autonomous logic
        final_answer, trace_steps = mental_health_agent_autonomous(messages_history)

        return jsonify({
            "status": "ok",
            "error": None,
            "response": final_answer,
            "steps": trace_steps
        })

    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "response": None,
            "steps": []
        }), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)