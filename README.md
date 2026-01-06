# Policy Gradient (REINFORCE) with Baseline

A compact PyTorch implementation of the REINFORCE algorithm enhanced with a learned baseline (value network). This repository trains a policy network on OpenAI Gym's CartPole-v1 environment using Monte Carlo returns and a value function baseline to reduce variance.

---

## 🔧 How the algorithm works (in English + a bit of math)

- Objective: maximize expected return by adjusting policy parameters θ.
- We use episodic Monte Carlo returns: for each time step t in an episode, the return is
  G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ...
  where 0 ≤ γ < 1 is the discount factor.
- Policy gradient (REINFORCE) update (Monte Carlo estimate):
  ∇_θ J(θ) ≈ E[∑_t ∇_θ log π_θ(a_t | s_t) * A_t]
  where A_t is the advantage at time t.
- Baseline (value function): we train a value network V_φ(s) to predict G_t and use the advantage
  A_t = G_t - V_φ(s_t),
  which reduces the variance of the policy gradient update without changing its expectation.
- Implementation details in this repo:
  - Policy network outputs action probabilities via a softmax and we sample actions from a Categorical distribution.
  - Value network outputs a scalar value V(s) trained with mean-squared error against G_t.
  - Losses:
    - Policy loss (to be minimized): L_policy = - E[∑_t log π_θ(a_t|s_t) * A_t]
    - Value loss (to be minimized): L_value = E[(V_φ(s_t) - G_t)^2]

> Summary: We collect full episodes, compute returns G_t, use a value network to form advantages A_t = G_t - V(s_t), update the policy using ∇_θ log π * A, and update the value network by minimizing MSE with returns.

---

## 🚀 Installation

Recommended: use a virtual environment (venv) and Python 3.8+.

1. Clone the repo and enter the directory:

   ```bash
   git clone <repo-url>
   cd Policy-Gradient-Reinforce-with-Baseline
   ```

2. Create and activate a virtual environment (Windows example):

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Note: This project uses PyTorch and gymnasium. If you want GPU support, install a CUDA-capable PyTorch build following instructions at https://pytorch.org.

---

## ▶️ How to run

- Run training with the main script:

  ```bash
  python main.py
  ```

- By default it runs 100 episodes (change the number in `main.py`). The script prints episode rewards periodically.

---

## 📁 Key files

- `main.py` — project entrypoint that calls the training loop
- `train.py` — training loop (collects episodes, computes returns, updates policy & value)
- `networks.py` — `PolicyNet` and `ValueNet` network definitions
- `returns.py` — computes discounted returns G_t using `G_{t} = r_t + γ G_{t+1}`
- `losses.py` — policy and value loss implementations and optimizer steps
- `config.py` — hyperparameters (learning rates, gamma)

---

## 💡 Notes & suggestions

- Reproducibility: you can add seeding (for torch, env, numpy) to make runs deterministic.
- Checkpoints: consider saving model weights periodically to resume training or evaluate later.
- Improvements: entropy regularization, batch updates, advantage normalization, or using GAE (generalized advantage estimation) can improve learning stability.

---

## 📜 License

This repository is released under the terms in the `LICENSE` file.

---

