import streamlit as st
import pandas as pd
from PIL import Image
import os

# ---------------- Page Configuration ----------------

st.set_page_config(
    page_title="Dynamic Pricing Dashboard",
    page_icon="📈",
    layout="wide"
)
# ---------------- Sidebar ----------------

with st.sidebar:

    st.title("📊 Dashboard Menu")

    st.markdown("---")

    st.subheader("Project")

    st.write("Dynamic Pricing using Reinforcement Learning")

    st.markdown("---")

    st.subheader("Algorithms")

    st.write("✅ Fixed Pricing")
    st.write("✅ Discount Pricing")
    st.write("✅ Q-Learning")
    st.write("✅ Deep Q-Network (DQN)")

    st.markdown("---")

    st.subheader("Technology Stack")

    st.write("- Python")
    st.write("- Gymnasium")
    st.write("- PyTorch")
    st.write("- Streamlit")
    st.write("- Matplotlib")

    st.markdown("---")

    st.success("Week 4 Completed")
st.title("📈 Dynamic Pricing using Reinforcement Learning")
st.markdown("### Week 4 - Performance Dashboard")

# ---------------- Load CSV ----------------

csv_path = "results/agent_comparison.csv"

if not os.path.exists(csv_path):
    st.error("agent_comparison.csv not found. Please run compare_agents.py first.")
    st.stop()

df = pd.read_csv(csv_path)

# ---------------- Find Best Agent ----------------

best_row = df.loc[df["Average Revenue"].idxmax()]
best_agent = best_row["Agent"]
best_revenue = best_row["Average Revenue"]

# ---------------- KPI Cards ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="🏆 Best Agent",
        value=best_agent
    )

with col2:
    st.metric(
        label="💰 Best Revenue",
        value=f"₹{best_revenue:.0f}"
    )

with col3:
    st.metric(
        label="🤖 Agents Compared",
        value=len(df)
    )

st.divider()

# ---------------- Revenue Table ----------------

# ---------------- Revenue Table ----------------

st.subheader("📋 Agent Revenue Comparison")

st.dataframe(df, use_container_width=True)

st.write("### 📈 Summary Statistics")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Highest Revenue",
        f"₹{df['Average Revenue'].max():,.0f}"
    )

with col2:
    st.metric(
        "Lowest Revenue",
        f"₹{df['Average Revenue'].min():,.0f}"
    )

with col3:
    st.metric(
        "Average of All Agents",
        f"₹{df['Average Revenue'].mean():,.0f}"
    )

st.divider()
# ---------------- Revenue Comparison Graph ----------------

st.subheader("📊 Revenue Comparison")

revenue_img = "results/revenue_comparison.png"

if os.path.exists(revenue_img):
    st.image(revenue_img, use_container_width=True)
else:
    st.warning("Revenue comparison graph not found.")

st.divider()

# ---------------- Two Column Layout ----------------

left, right = st.columns(2)

# Left Column
with left:

    st.subheader("📈 DQN Training Reward Curve")

    reward_img = "results/reward_curve.png"

    if os.path.exists(reward_img):
        st.image(reward_img, use_container_width=True)
    else:
        st.warning("Reward curve not found.")

    st.subheader("💰 Price Trajectory")

    price_img = "results/price_trajectory.png"

    if os.path.exists(price_img):
        st.image(price_img, use_container_width=True)
    else:
        st.warning("Price trajectory not found.")

# Right Column
with right:

    st.subheader("📦 Inventory Curve")

    inventory_img = "results/inventory_curve.png"

    if os.path.exists(inventory_img):
        st.image(inventory_img, use_container_width=True)
    else:
        st.warning("Inventory curve not found.")

    st.subheader("📌 Dashboard Summary")

    st.success(f"🏆 Best Performing Agent: **{best_agent}**")

    st.info(f"💰 Average Revenue: **{best_revenue:.2f}**")

with st.expander("📖 Project Overview"):

    st.write("""
This project implements a **Dynamic Pricing System** using Reinforcement Learning.

### Objectives
- Maximize revenue
- Learn optimal pricing strategies
- Compare traditional pricing with RL-based methods

### Algorithms Used
- Fixed Pricing
- Discount Pricing
- Q-Learning
- Deep Q-Network (DQN)

### Dashboard Contents
- Revenue comparison
- Training reward curve
- Price trajectory
- Inventory curve
- Performance summary
""")
st.divider()

st.caption("Dynamic Pricing using Reinforcement Learning | Week 4 Dashboard")
