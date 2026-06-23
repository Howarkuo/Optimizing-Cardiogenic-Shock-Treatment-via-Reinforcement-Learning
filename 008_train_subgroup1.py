import torch
import numpy as np
import tianshou as ts
from tianshou.utils.net.common import Net
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import DQNPolicy
from tianshou.trainer import OffpolicyTrainer

# ==========================================
# ⚙️ CONFIGURATION (Thesis Specs)
# ==========================================
LR = 1e-4             # Learning Rate (Requested)
GAMMA = 0.99          # Discount Factor (Requested)
EPS_MIN = 0.05        # Exploration Min (Requested)
BATCH_SIZE = 128      # Larger batch for stable medical gradients
STATE_SHAPE = 20      # Your 20 clinical features
ACTION_SHAPE = 5      # Your 5 vasopressor bins
MAX_STEPS = 20        # Max trajectory length for Shock

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# 1. EXPERIENCE REPLAY BUFFER (Offline Data)
# ==========================================
# Assuming 'rl_data' is the dictionary from our previous tensor script
def prepare_buffer(rl_data):
    # Calculate buffer size based on your Compensated subgroup transitions
    buffer_size = len(rl_data['s']) 
    buffer = ReplayBuffer(size=buffer_size)
    
    # Load historical transitions into the buffer
    # S, A, R, S_prime, Done
    buffer.add(Batch(
        obs=rl_data['s'].numpy(),
        act=rl_data['a'].numpy(),
        rew=rl_data['r'].numpy(),
        obs_next=rl_data['sp'].numpy(),
        terminated=rl_data['done'].numpy(),
        truncated=np.zeros(buffer_size, dtype=bool) # Truncation logic if needed
    ))
    print(f"📥 Offline Replay Buffer loaded with {len(buffer)} transitions.")
    return buffer

# ==========================================
# 2. DUELING NETWORK (The "Brain")
# ==========================================
# Shared trunk + split Value (V) and Advantage (A) streams
net = Net(
    state_shape=STATE_SHAPE,
    action_shape=ACTION_SHAPE,
    hidden_sizes=[256, 256], # Wider layers for complex medical noise
    device=device,
    dueling_param=(
        {"hidden_sizes": [128]}, # Advantage stream
        {"hidden_sizes": [128]}  # Value stream
    )
).to(device)

optim = torch.optim.Adam(net.parameters(), lr=LR)

# ==========================================
# 3. DOUBLE DQN POLICY
# ==========================================
policy = DQNPolicy(
    model=net,
    optim=optim,
    discount_factor=GAMMA,
    estimation_step=3,        # 3-step return for better credit assignment
    target_update_freq=500,   # Slow target update for clinical stability
    is_double=True            # <--- Enables Double DQN logic
).to(device)

# # ==========================================
# # 4. OFFLINE TRAINING (Episodes: 500-1000)
# # ==========================================
# # Since this is offline, we skip the 'Collector' and use the buffer directly
# # to simulate "epochs" of learning from historical data.

# buffer = prepare_buffer(rl_data)

# # Training progression tracker
# for epoch in range(1, 1001): # Episodes/Epochs (Requested)
#     # Perform training steps from the buffer
#     stats = policy.update(sample_size=BATCH_SIZE, buffer=buffer)
    
#     # Update Exploration (Epsilon Decay)
#     eps = max(EPS_MIN, 1.0 - (epoch / 500)) 
#     policy.set_eps(eps)

#     if epoch % 100 == 0:
#         print(f"🚀 Episode {epoch}/1000 | Loss: {stats['loss']:.4f} | Q-Avg: {stats['v_avg']:.2f}")

# # Final Policy Export
# torch.save(policy.state_dict(), "d3qn_compensated_shock_model.pth")

import matplotlib.pyplot as plt

# 1. Initialize lists to store metrics
q_values = []
losses = []
rewards = []

# --- Inside your training loop (from the previous script) ---
for epoch in range(1, 1001):
    # Train step
    stats = policy.update(sample_size=BATCH_SIZE, buffer=buffer)
    
    # Capture the "Critic's" opinion
    q_values.append(stats['v_avg']) # v_avg is the state value estimate
    losses.append(stats['loss'])
    
    # 2. Every 100 episodes, run a "Test" on the Validation Tensor
    if epoch % 100 == 0:
        with torch.no_grad():
            # Get AI's predicted Q-values for the Validation set
            val_output = policy.model(X_val) # X_val from Step 7
            q_avg_val = val_output.mean().item()
            print(f"📈 Val Q-Avg: {q_avg_val:.2f}")

# ==========================================
# 4. GENERATING THE THESIS FIGURES
# ==========================================

# Figure 1: Convergence Plot
plt.figure(figsize=(10, 5))
plt.plot(q_values, label='Average Q-Value (V)')
plt.title('D3QN Value Convergence: Compensated Shock Subgroup')
plt.xlabel('Training Steps')
plt.ylabel('Estimated State Value')
plt.legend()
plt.savefig("convergence_plot.png")

# Figure 2: The AI vs Clinician "Difference" Logic
# (This is calculated AFTER training is done)
def plot_clinical_deviation(policy, val_buffer):
    # 1. Get AI actions vs Real actions
    # 2. Calculate Dose Difference (Clinician NED - AI NED)
    # 3. Bin the differences and calculate mortality for each bin
    pass