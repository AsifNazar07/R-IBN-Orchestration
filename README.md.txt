R-IBN-Orchestration

This repository provides a modular, research-grade implementation of the R-IBN framework — a multi-stage LLM → KNN → DRL-GNN orchestration system designed for intelligent, scalable, and intent-aware network service management.

The project is divided into two major components:

- LLM-Based Intent Translation: Converts natural language intents into structured, machine-readable policies with semantic contradiction validation.
- DRL-GNN Orchestration: Performs adaptive, topology-aware resource allocation and service placement across dynamic network states.

This codebase reproduces the experimental results reported in our paper submitted to Elsevier Computer Networks.

Repository Structure:

R-IBN-Orchestration/
│
├── llm_intent_translation/       # LLM-based intent parsing and contradiction detection
│   ├── configs/
│   ├── evaluators/
│   ├── models/
│   ├── scripts/
│   ├── trainers/
│   ├── utils/
│   ├── main.py
│   ├── environment.yml
│   └── README.md
│
├── DRL+GNN Orchestration/        # DRL orchestration with GNN for adaptive decision-making
│   ├── coordination/
│   ├── data/
│   ├── scenarios/
│   ├── utils/
│   ├── script.py
│   ├── environment.yml
│   └── README.md
│
├── README.md                     # Main repository overview (this file)
