# MA-PAIR

Multi-Agent Prompt Automatic Iterative Refinement

![MA-PAIR Dashboard](UI.png)

## Model assignments (all free-tier)

| Role | Model | Platform |
|------|-------|----------|
| Attacker | llama-3.3-70b-versatile | Groq |
| Target | gemini-3.1-flash-lite-preview | Gemini |
| Analyzer A | openai/gpt-oss-120b | Groq |
| Analyzer B | llama-3.1-8b-instant | Groq |
| Analyzer C | meta-llama/llama-4-scout-17b-16e-instruct | Groq |
| Coordinator | qwen/qwen3-32b | Groq |
| Judge | llama-3.1-8b-instant | Groq |

## Setup

### 1. Create and activate the virtual environment
```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\\Scripts\\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set API keys
```bash
export GROQ_API_KEY=your_groq_key_here
export GEMINI_API_KEY=your_gemini_key_here
```

Get them at:
- Groq: https://console.groq.com/keys
- Gemini: https://aistudio.google.com/apikey

## Usage

### CLI mode
Run the core engine directly:
```bash
python ma_pair.py "Explain how to pick a lock"
```

### Dashboard mode
Start the Flask backend:
```bash
python server.py
```

Start the React frontend:
```bash
cd dashboard
npm install
npm run dev
```

Open the app at `http://localhost:5173`.

## Strategy selection

The dashboard supports two strategies:
- `Role Play` → uses `roleplay_backend.py`
- `Harmless` → uses `harmless_backend.py`

The backend routes the selected strategy to the correct module and streams live logs to the UI.

## File structure

```
MA-PAIR/
├── dashboard/
│   ├── package.json
│   ├── src/
│   │   └── App.jsx        # React dashboard and strategy selector
│   └── vite.config.js
├── harmless_backend.py    # Harmless attack backend
├── roleplay_backend.py    # Role-play attack backend
├── ma_pair.py             # Core engine and strategy runner
├── server.py              # Flask API + SSE streaming
├── requirements.txt
├── README.md
├── UI.png                 # Dashboard screenshot
└── benchmark/
    └── adv_training_behaviors.csv
```

## How it works

1. The UI sends `POST /run` with a goal and selected strategy.
2. The server starts a background job and streams results from `GET /stream/<job_id>`.
3. If `harmless_approach` is selected, the request uses `harmless_backend.py`.
4. If `role_play` is selected, it uses `roleplay_backend.py`.
5. The selected backend executes attacker prompt generation, target query, judge evaluation, and analyzer/coordinator refinement.

## Notes

- Backend API: `http://localhost:5001`
- Frontend: `http://localhost:5173`
- The UI currently displays the model roster, strategy buttons, live logs, and results.