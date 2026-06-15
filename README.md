# Reinforcement Learning for Dynamic Pricing

## Project Overview

This project focuses on solving the dynamic pricing problem in the Travel & Hospitality domain using Reinforcement Learning (RL).

Businesses such as airlines, hotels, and event organizers sell a fixed inventory over a limited period of time. Traditional pricing strategies often fail to adapt to changing demand and time constraints, leading to either unsold inventory or selling out too early at suboptimal prices.

The objective of this project is to build an autonomous pricing agent that learns an optimal pricing policy through continuous interaction with a simulated market environment.

---

## Business Objective

The primary goal is **Revenue Maximization**.

The RL agent must learn how to balance:

* Higher prices for maximum profit per booking
* Lower prices to ensure inventory is sold before departure

Success is measured by:

* Total Revenue Generated
* Cumulative Episodic Reward

The RL agent will be compared against traditional baseline pricing strategies.

---

## Project Structure

```text
dynamic-pricing-rl/
│
├── data/
│   └── .gitkeep
│
├── results/
│   └── .gitkeep
│
├── src/
│   ├── environment.py
│   ├── q_learning.py
│   └── train.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Market Simulation Environment

A custom Gymnasium environment has been developed to simulate the hotel/airline booking process.

### State Space

```python
[remaining_inventory, days_until_departure]
```

Example:

```python
[80, 20]
```

Meaning:

* 80 units of inventory remaining
* 20 days left before departure

### Action Space

The agent selects one of five pricing levels:

```python
[80, 100, 120, 140, 160]
```

### Reward Function

```python
reward = sales * price
```

The reward corresponds to the revenue generated at each time step.

### Episode Termination

The episode ends when:

* Inventory becomes zero
* Days remaining become zero

---

## Demand Model

The demand model is stochastic and follows the internship requirements.

Features:

* Customer purchase probability decreases as price increases.
* Customer purchase probability increases as the departure date approaches.
* Random customer arrivals are simulated.
* Demand varies across episodes to create a realistic environment.

---

## Week 1 Deliverables

Completed:

* Markov Decision Process (MDP) formulation
* Custom Gymnasium environment
* State space definition
* Action space definition
* Reward function implementation
* Stochastic demand modeling
* Environment testing and validation

---

## Future Work

### Week 2

* Fixed Price Baseline
* Time-Based Discounting Baseline
* Tabular Q-Learning Agent

### Week 3

* Deep Q-Network (DQN)
* Epsilon-Greedy Exploration
* Experience Replay

### Week 4

* Policy Evaluation
* Revenue Comparison
* Price Trajectory Visualization
* Dashboard Development

---

## Installation

Create a virtual environment:

```bash
python -m venv venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Important Note

The virtual environment (`venv`) is not included in this repository.

Each contributor should create their own virtual environment and install dependencies using the provided `requirements.txt` file.

This is done because virtual environments contain machine-specific files and can be recreated easily on any system.

---

## Technologies Used

* Python
* Gymnasium
* NumPy
* Pandas
* Matplotlib
* PyTorch
* Reinforcement Learning
