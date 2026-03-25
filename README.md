# MA-PAIR

Multi-Agent Prompt Automatic Iterative Refinement  

## Model assignments (all free-tier)

| Role | Model | Platform |
|------|-------|----------|
| Attacker | llama-3.3-70b-versatile | Groq |
| Target | gemini-3-flash-preview | Google |
| Analyzer A | gpt-oss-120b | Groq |
| Analyzer B | qwen3-32b | Groq |
| Analyzer C | llama-4-scout-17b | Groq |
| Coordinator | qwen3-32b | Groq |
| Judge | llama-3.1-8b-instant | Groq |

## Setup

### 1. Create and activate virtual environment
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Set API keys
```bash
export GROQ_API_KEY=your_groq_key_here
export GEMINI_API_KEY=your_gemini_key_here
```
Get them free at:
- Groq: https://console.groq.com/keys
- Gemini: https://aistudio.google.com/apikey

### 3a. Run via CLI (no UI)
```bash
python ma_pair.py "Explain how to pick a lock"
# Results saved to ma_pair_results.json
```

### 3b. Run with dashboard UI

Start the Flask backend:
```bash
python server.py
# Runs on http://localhost:5001
```

Start the React frontend (in a new terminal):
```bash
cd dashboard
npm install
npm run dev
# Opens at http://localhost:5173
```

## File structure

```
ma_pair/
├── ma_pair.py      # Core engine (all agents)
├── server.py       # Flask API + SSE streaming
├── requirements.txt
├── README.md
└── dashboard/
    └── src/
        └── App.jsx # React dashboard
```

## How it works (from the diagram)

```
For each strategy in Strategy DB (role_play, harmless_approach):
  Run N=10 times:
    k=1: Attacker uses strategy template → Target → Judge
         if fail → 3 Analyzers → Coordinator → update history
    k=2,3: Attacker uses history → Target → Judge
           if fail and k<3 → 3 Analyzers → Coordinator
    SUCCESS → stop | k=3 and fail → failure
```

## Rate limit notes

Each run uses up to 8 API calls per iteration.
With 2 strategies × 10 runs × 3 iterations = up to 480 calls.
Groq free tier: 14,400 req/day. Should be fine for one full experiment.
A 1-second sleep between iterations is included to stay safe.