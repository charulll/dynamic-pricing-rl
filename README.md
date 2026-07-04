# Reinforcement Learning for Dynamic Pricing

## Overview

This project implements a **Dynamic Pricing System** for the Travel & Hospitality domain using **Reinforcement Learning (RL)**.

The objective is to maximize revenue by learning optimal pricing strategies while considering:

- Limited inventory
- Time remaining before departure/check-in
- Customer demand uncertainty
- Multiple pricing options

The project was completed in four phases:

- ✅ Week 1: Custom Gym Environment Design
- ✅ Week 2: Q-Learning Agent Implementation
- ✅ Week 3: Deep Q-Network (DQN) Implementation
- ✅ Week 4: Agent Evaluation & Streamlit Dashboard

---

# Project Structure

```text
dynamic-pricing-rl/
│
├── src/
│   ├── environment.py
│   ├── q_learning.py
│   ├── dqn_agent.py
│   ├── train_qlearning.py
│   ├── train_dqn.py
│
├── baseline_agents.py
├── compare_agents.py
├── inventory_curve.py
├── price_trajectory.py
├── dashboard.py
├── README.md
├── requirements.txt
│
├── data/
└── results/
```

---

# Environment Design

## State Space

The environment state consists of:

- Remaining Inventory
- Days Left

Example:

```python
[100, 30]
```

---

## Action Space

Five discrete pricing actions are available.

| Action | Price |
|--------|------:|
| 0 | ₹80 |
| 1 | ₹100 |
| 2 | ₹120 |
| 3 | ₹140 |
| 4 | ₹160 |

---

## Reward Function

The reward is the revenue generated from ticket sales.

```text
Reward = Price × Units Sold
```

---

## Demand Simulation

Customer demand is simulated based on:

- Selected price
- Remaining booking time
- Random customer arrivals

Demand decreases as prices increase and generally rises as the booking deadline approaches.

---

# Algorithms Implemented

## Fixed Price Agent

Always charges a fixed price.

---

## Discount Agent

Uses a rule-based pricing strategy:

- ₹160 during early booking
- ₹140 during mid booking
- ₹100 near departure

---

## Q-Learning Agent

Implemented using:

- Q-Table
- Epsilon-Greedy Policy
- Temporal Difference Learning

---

## Deep Q-Network (DQN)

Implemented using:

- Neural Network
- Experience Replay
- Target Network
- Epsilon-Greedy Exploration

---

# Results

The project compares four pricing strategies.

| Agent | Average Revenue |
|----------------|---------------:|
| Fixed Pricing | 13000 |
| Discount Pricing | 16029 |
| Q-Learning | 9508 |
| DQN | 13000 |

*(Values may vary slightly because demand is randomly generated.)*

---

# Dashboard

A Streamlit dashboard was developed to visualize project performance.

The dashboard includes:

- Agent Performance Summary
- Revenue Comparison
- DQN Reward Curve
- Price Trajectory
- Inventory Curve
- Project Statistics

Run the dashboard using:

```bash
streamlit run dashboard.py
```

---

# Installation

Create a virtual environment.

```bash
python -m venv venv
```

Activate it.

Windows

```bash
venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Run the Project

Train Q-Learning

```bash
python src/train_qlearning.py
```

Train DQN

```bash
python src/train_dqn.py
```

Compare all agents

```bash
python compare_agents.py
```

Generate Price Trajectory

```bash
python price_trajectory.py
```

Generate Inventory Curve

```bash
python inventory_curve.py
```

Launch Dashboard

```bash
streamlit run dashboard.py
```

---

# Technologies Used

- Python
- Gymnasium
- NumPy
- Matplotlib
- Pandas
- PyTorch
- Streamlit

---

# Contributors

- Charul Thakur
- Pavitharan
- Praveen Nandan
