# MindFulMate: Real-Time Emotional Support for the "In-Between" Moments

While formal therapy is essential, it is often limited by long wait times, high costs, and the need for scheduled appointments, leaving a critical gap in care during unpredictable **"in-between" moments** when distress hits suddenly, such as in the middle of the night. **MindFulMate** is an autonomous mental health AI agent designed to bridge this gap by providing a safe, real-time space to process emotions exactly when they arise. Powered by a **ReAct architecture** and grounded in **professional baseline data**, the agent ensures that empathetic and safe guidance is accessible 24/7, providing support in the moments it is needed most.

### 🔗 [Try MindFulMate Live Here](https://roi-project-lld9.onrender.com/)
Example prompts for using MindFulMate:
* **Academic Stress**: *"I'm feeling overwhelmed with my exams at the Technion, can you help?"*
* **Media Support**: *"I am stressed before my exam, maybe some music could help?"*
* **Grounding Needed**: *"I'm having a lot of anxiety right now and need a quick exercise to calm down."*
* **Emotional Venting**: *"I'm feeling a bit sad today and just need someone to talk to."*

## 🚀 Key Capabilities

* **Support for "In-Between" Moments**: Provides immediate, 24/7 accessibility for emotional processing between formal therapy sessions.
* **ReAct Architecture**: Utilizes a Reasoning + Acting framework (Thought -> Action -> Observation) to provide structured and logical support.
* **Professional Baseline**: Grounded in verified therapist transcripts to ensure responses are empathetic, safe, and professional.
* **Safety & Reflection Loop**: Every response is critiqued by a "Reflector" LLM to ensure it meets safety standards before being displayed.
* **Crisis Detection**: Proactively identifies high-risk emotional states and provides external emergency resources when necessary.
* **Therapeutic Media**: Autonomously searches and validates YouTube content (music, breathing exercises) to provide immediate coping tools.

## 📂 Project Structure

### Core Logic
* **`agent_logic.py`**: The "Brain" of the agent. Implements the ReAct architecture, YouTube tool integration, and safety validation logic.
* **`app.py`**: Flask backend serving the API endpoints (`/api/execute`, `/api/agent_info`) and managing deployment health checks.

### Data & Preprocessing
* **`kaggle_data_preperation.py`**: Scripts for cleaning and preparing the professional conversational datasets.
* **`importing_to_pinecone.py`**: Generates embeddings and manages the vector database (Pinecone) for RAG (Retrieval-Augmented Generation).
* **`data_concatination.py`**: Merges therapist transcripts and contextual data (Reddit/Kaggle) into a unified format.

### Frontend
* **`templates/index.html`**: A responsive chat interface featuring RTL support, real-time execution traces, and embedded video players.

## 🛠️ Technical Stack
* **LLM**: GPT-based models (via LLMOD).
* **Vector DB**: Pinecone (for semantic search and RAG).
* **Backend**: Python / Flask.
* **Frontend**: Vanilla JavaScript, Bootstrap 5 (RTL).
* **Deployment**: Render.

---
*Developed by Oded Kapeta, Rona Lavi and Itay Davidovich Gross*
