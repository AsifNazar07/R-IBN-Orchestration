# Hierarchical DRL-GNN Framework for Uncertainty-Aware Network Service Orchestration

This repository contains a research-grade implementation of a Deep Reinforcement Learning (DRL) framework tightly integrated with Graph Neural Networks (GNNs) for predictive and uncertainty-resilient service coordination across multi-domain virtualized networks.

The architecture leverages temporal-graph embeddings and policy-space factorization across stochastic service function chains (SFCs), enabling learning-based orchestration in non-stationary traffic topologies with partial observability.

> This implementation assumes expert-level familiarity with DRL, GNN embeddings, and constrained Markov Decision Processes. Reproducibility is guaranteed, but operational usage demands significant customization.

---

## Installation

Prepare an isolated Python 3.8 environment. Then install all dependencies:

```bash
python -m venv venv38
source venv38/bin/activate  # Linux/Mac
venv38\Scripts\activate     # Windows

pip install -r requirements.txt
