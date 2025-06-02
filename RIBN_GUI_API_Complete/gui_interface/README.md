# GUI Interface – R-IBN Orchestration

This module provides a Streamlit-based interactive GUI to test the full R-IBN orchestration pipeline, including:

- Intent configuration using LLM translation
- KNN-based contradiction detection
- GNN-enhanced DRL orchestration
- Real-time monitoring dashboard

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/R-IBN-Orchestration.git
cd R-IBN-Orchestration/gui_interface
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the interface:
```bash
streamlit run app.py
```

> Ensure that the backend modules (`llm_intent_translation` and `drlgnn_orchestration`) are functional and importable.

## Project Dependencies

- streamlit
- torch
- transformers
- numpy
- requests
