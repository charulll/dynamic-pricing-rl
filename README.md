# Reinforcement Learning for Dynamic Pricing

## Overview

This project implements a Dynamic Pricing system for the Travel & Hospitality domain using Reinforcement Learning.

The objective is to maximize revenue by learning optimal pricing strategies while considering:

* Limited inventory
* Time remaining before departure/check-in
* Customer demand uncertainty
* Multiple pricing options

The project is developed in phases:

* Week 1: Custom Gym Environment Design
* Week 2: Q-Learning Agent Implementation
* Week 3: Deep Q-Network (DQN) Implementation

---

## Project Structure

```text
dynamic-pricing-rl/
│
├── src/
│   ├── environment.py
│   ├── q_learning.py
│   ├── train_qlearning.py
│
├── baseline_agents.py
├── compare_agents.py
├── requirements.txt
├── README.md
│
├── data/
└── results/
```

---

## Environment Design

### State Space

The environment state consists of:

* Remaining Inventory
* Days Left

Example:

```python
[100, 30]
```

---

### Action Space

Five discrete pricing actions are available:

| Action | Price |
| ------ | ----- |
| 0      | ₹80   |
| 1      | ₹100  |
| 2      | ₹120  |
| 3      | ₹140  |
| 4      | ₹160  |

---

### Reward Function

Revenue generated from sales:

```text
Reward = Price × Units Sold
```

---

### Demand Simulation

Customer demand is stochastic and depends on:

* Selected price
* Time remaining
* Random customer arrivals

Demand decreases with higher prices and increases as the booking deadline approaches.

---

## Q-Learning Agent

The agent learns an optimal pricing policy using:

* Q-Table
* Epsilon-Greedy Exploration
* Temporal Difference Learning

Q-Learning Update Rule:

```text
Q(s,a) = Q(s,a) + α [ r + γ max Q(s',a') - Q(s,a) ]
```

---

## Baseline Agents

### Fixed Price Agent

Always selects:

```text
₹120
```

### Discount Agent

Pricing strategy:

* ₹160 when demand horizon is large
* ₹140 during mid-horizon
* ₹100 near the deadline

---

## Results

### Baseline Comparison

| Agent             | Average Revenue |
| ----------------- | --------------: |
| Fixed Price Agent |           12000 |
| Discount Agent    |           15855 |
| Q-Learning Agent  |            8232 |

These results provide a benchmark for future DQN implementation and evaluation.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Q-Learning Training

```bash
python src/train_qlearning.py
```

---

## Compare Pricing Strategies

```bash
python compare_agents.py
```

---

## Future Work

* Deep Q-Network (DQN)
* Experience Replay Buffer
* Target Network
* Hyperparameter Optimization
* Revenue Performance Comparison

---

## Contributors

* Charul 
* Pavitharan
* Praveen Nandan
