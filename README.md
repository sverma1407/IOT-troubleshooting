# IoTAssist AI — Smart IoT Troubleshooting Assistant

A multi-agent AI web application for diagnosing and fixing IoT device problems,
powered by **IBM watsonx.ai Granite Models** and built with **Flask**.

---

## Project Overview

IoTAssist AI provides four specialised AI agents that together cover the full
troubleshooting lifecycle — from initial diagnosis through knowledge retrieval,
guided step-by-step repair, and an open conversational chat interface.

| Agent | Route | Description |
|---|---|---|
| Agent 1 – Device Diagnosis | `/diagnosis` | Classifies the problem, identifies root causes, and produces a structured device health report |
| Agent 2 – Knowledge Center | `/knowledge` | Retrieval-Augmented Generation (RAG) over an IoT knowledge base — returns technical docs and best practices |
| Agent 3 – Troubleshooting | `/troubleshooting` | Generates a numbered, step-by-step repair workflow with diagnostic checklist and resolution time estimate |
| Agent 4 – AI Chat Support | `/chat` | Context-aware conversational chatbot that remembers the full session and handles free-form follow-up questions |

---

## Agent Details

### Agent 1 — Device Issue Understanding Agent
Accepts a plain-English description of the problem and returns a structured
JSON health assessment including:
- Problem summary and issue category (Wi-Fi, Bluetooth, Zigbee, Sensor, etc.)
- Possible root causes and affected components
- Severity level (High / Medium / Low) and device health (Critical / Degraded / Warning)
- AI explanation from IBM Granite

### Agent 2 — Knowledge Retrieval Agent (RAG)
Implements a three-step Retrieval-Augmented Generation pipeline:
1. **Retrieve** — keyword search across a 12-topic IoT knowledge base
2. **Augment** — inject retrieved documents into the Granite prompt
3. **Generate** — IBM Granite produces a grounded technical explanation

Returns topic summary, configuration recommendations, best practices, and setup guidelines.

### Agent 3 — Step-by-Step Troubleshooting Agent
Generates a complete repair workflow for a given issue type:
- 5 numbered troubleshooting steps with expected results
- Diagnostic checklist and corrective actions
- Verification steps and preventive maintenance tips
- Estimated resolution time and difficulty level

### Agent 4 — AI Chat Support (Conversational Agent)
A full multi-turn chatbot powered by IBM Granite:
- Maintains conversation history across the entire session (last 10 turns)
- Understands follow-up questions and context references
- Quick-question chips for common topics
- Real-time typing indicator and message timestamps

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask |
| AI Models | IBM watsonx.ai — `ibm/granite-4-h-small` |
| AI SDK | `ibm-watsonx-ai` |
| Frontend | Bootstrap 5.3, Bootstrap Icons, Vanilla JS |
| Config | `python-dotenv` |
| Production server | Gunicorn |

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/iot-troubleshooting.git
cd iot-troubleshooting
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the provided template and fill in your credentials:

```bash
# Windows
copy .env .env.local   # keep .env as the template
```

Open `.env` in a text editor and set:

```env
WATSONX_APIKEY=your_ibm_watsonx_api_key_here
PROJECT_ID=your_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
SECRET_KEY=a-long-random-string
```

**How to obtain credentials:**
- `WATSONX_APIKEY` — [IBM Cloud IAM → API Keys](https://cloud.ibm.com/iam/apikeys)
- `PROJECT_ID` — Open your project in [watsonx.ai](https://dataplatform.cloud.ibm.com/), go to **Manage → General** and copy the Project ID
- `WATSONX_URL` — Depends on your IBM Cloud region (`us-south`, `eu-de`, `jp-tok`, etc.)
- `SECRET_KEY` — Generate with: `python -c "import secrets; print(secrets.token_hex(32))"`

### 5. Run the application locally

```bash
python app.py
```

The app starts at **http://127.0.0.1:5000**.

Flask's built-in reloader is active in debug mode — saving `app.py` automatically restarts the server.

### 6. Production deployment (optional)

Use Gunicorn instead of Flask's development server:

```bash
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

---

## Project Structure

```
iot-troubleshooting/
├── app.py               # Main Flask application — all agents, routes, and templates
├── .env                 # Environment variables (NOT committed to Git)
├── .gitignore           # Git ignore rules
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Home page |
| `GET` | `/diagnosis` | Device Diagnosis page |
| `GET` | `/knowledge` | Knowledge Center page |
| `GET` | `/troubleshooting` | Troubleshooting page |
| `GET` | `/chat` | AI Chat Support page |
| `GET` | `/about` | About page |
| `POST` | `/api/diagnose` | Agent 1 — JSON body: `{"description": "..."}` |
| `POST` | `/api/knowledge` | Agent 2 — JSON body: `{"query": "..."}` |
| `POST` | `/api/troubleshoot` | Agent 3 — JSON body: `{"issue_type": "...", "device_details": "..."}` |
| `POST` | `/api/chat` | Agent 4 — JSON body: `{"message": "..."}` |
| `POST` | `/api/chat/clear` | Clear Agent 4 session history |

---

## Security Notes

- **Never commit `.env`** — it is listed in `.gitignore`
- Rotate your `WATSONX_APIKEY` immediately if it is ever exposed in a commit
- Set a strong, random `SECRET_KEY` in production — the default fallback is insecure
- Run behind HTTPS in production (use a reverse proxy such as Nginx or a PaaS platform)

---

## License

MIT
