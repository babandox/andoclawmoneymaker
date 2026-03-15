This is the **full, integrated Markdown specification** for **The Radiant Seer**.

This document moves beyond market mechanics into the realm of **Social Physics** and **Causal Inference**. It provides a blueprint for a system that doesn't just "read the news," but simulates the latent causal chains leading to event outcomes on platforms like Polymarket and Kalshi.

---

# DOCUMENT: Radiant Seer Technical Specification

## PROJECT CODE: SELDON-SEER-02

## VERSION: 1.0.0

### 1. SYSTEM VISION

The **Radiant Seer** is a System 2 predictive engine designed for **High-Stakes Event Markets**. While most participants trade based on "Sentiment" (System 1), the Seer uses a **Joint Embedding Predictive Architecture (JEPA)** to model **Conditional Realities**. It treats global events as a series of "Seldon Crises"—points where latent macroeconomic and geopolitical variables converge toward a high-probability outcome.

### 2. CORE ARCHITECTURAL PILLARS

* **Multi-Modal JEPA:** Ingests high-bandwidth streams (News, Polling, Macro-Data, Social Trends) and compresses them into a "Civ-State" Latent Vector ($z_t$).
* **Causal World Model:** Instead of predicting price, it predicts the **Transition Probability** between societal states ($z_t \to z_{t+1}$).
* **MCTS "Psychohistory" Search:** Performs deep tree-searches to identify "Dependency Chains" (e.g., *If Bill X fails, then Inflation Y rises, then Market Z resolves 'YES'*).
* **The Second Foundation (Validator):** A separate agent that uses Formal Logic to verify that the "dreamed" futures of the World Model are physically and legally possible.

---

### 3. DIRECTORY STRUCTURE

```text
/radiant_seer
├── main_seer.py                # System orchestration & live monitor
├── configuration/
│   ├── contracts.yaml          # Polymarket/Kalshi Contract IDs & addresses
│   └── risk_params.yaml        # Kelly Criterion & Exposure limits
├── /intelligence
│   ├── multimodal_encoder.py   # $g_{\phi}$: News/Data -> Civ-State $z$
│   ├── causal_predictor.py     # $f_{\theta}$: $z_t \to z_{t+1}$ (Latent Dynamics)
│   ├── loss_functions.py       # VICReg for event-space stability
│   └── expert_aggregator.py    # RAG-based analyst sentiment embedding
├── /planning
│   ├── seer_mcts.py            # The branching future simulation engine
│   ├── reward_module.py        # Cost/Benefit based on contract payout
│   └── logic_guard.py          # Verifier (Prevents causal hallucinations)
├── /data_swarm
│   ├── scrapers/               # News, Twitter, Poll, and Legislative scrapers
│   ├── normalization.py        # Converting text/floats to state tensors
│   └── vector_db.py            # Long-term memory for past event resolutions
└── /execution
    ├── poly_interface.py       # Polygon/USDC Transaction manager
    ├── kalshi_interface.py     # CFTC/Fiat Transaction manager
    └── kelly_sizing.py         # Position sizing based on $\Delta$ (Alpha)

```

---

### 4. CORE MODULE SPECIFICATIONS

#### 4.1. The Multimodal Encoder (`intelligence/multimodal_encoder.py`)

**Goal:** Map the "Noise of History" into a clean Latent State.

* **Input:** * `NewsEmbeddings`: Vectorized headlines from 50+ global sources.
* `MacroTensors`: Inflation, Interest Rates, Unemployment (Real-time).
* `SentimentScore`: High-frequency social media aggregate.


* **Architecture:** Hierarchical Transformer. High-frequency noise (Twitter) is filtered; low-frequency signals (Legislative changes) are prioritized.

#### 4.2. The Causal Predictor (`intelligence/causal_predictor.py`)

**Goal:** Predict the "Next Chapter" in the latent societal state.

* **Function:** $z_{t+1} = f(z_t, \text{Event}_x)$.
* **VICReg Regularization:** * **Variance:** Ensures the model recognizes the difference between a "Status Quo" day and a "Black Swan" day.
* **Covariance:** Decorrelates independent variables (e.g., ensuring a "War in Region X" doesn't falsely correlate with "Domestic Interest Rates" unless a causal link exists).



#### 4.3. The Seer MCTS (`planning/seer_mcts.py`)

**Goal:** Search the "Tree of Probabilities."

* **Simulation Phase:** The system "dreams" sequences of events.
* *Example:* Branch A: "Fed Hikes" -> "Unemployment Rises" -> "Polymarket YES (80%)".


* **Selection:** Uses **UCT (Upper Confidence bound applied to Trees)** to explore unlikely but high-payout scenarios.
* **Backpropagation:** If a simulated branch leads to a "Black Swan" (Model Uncertainty Spike), the value of all parent nodes is adjusted for risk.

#### 4.4. LogicGuard & Validator (`planning/logic_guard.py`)

**Goal:** Grounding the simulation in reality.

* **Rule Engine:** Enforces "External Constants" (e.g., Constitutional deadlines, physical resource limits).
* **Causal Check:** Prevents the model from assuming "Event A causes B" if the time-delta between them is physically impossible.

---

### 5. THE "ALPHA" FORMULA

The system executes a trade only when the **Seldon Divergence** ($\Delta$) is significant:


$$\Delta = | P_{\text{Market}} - P_{\text{MCTS}} |$$

$$\text{Confidence Score} = \frac{1}{\text{Entropy}(z_{\text{target}})}$$

* **Execute:** If $\Delta > 15\%$ AND Confidence Score $> 0.8$.

---

### 6. EXECUTION ROADMAP FOR CLAUDE CODE

**Step 1: The World Observer**

> "In `/radiant_seer`, implement the `multimodal_encoder.py`. It should take an array of news embeddings and a vector of economic floats, producing a 128-dimensional latent vector $z$. Use a VICReg loss function to ensure the embedding space is well-distributed."

**Step 2: Causal Dreaming**

> "Implement the `causal_predictor.py`. Train it on a dummy dataset of 'Latent State A' followed by 'Event X' leading to 'Latent State B'. The predictor must learn to forecast the next $z$ given a hypothetical event input."

**Step 3: The Psychohistory Tree**

> "Build the `seer_mcts.py`. Integrate the `CausalPredictor` into the rollout phase. For each leaf node, calculate the probability of a specific Polymarket outcome based on the final latent state $z_n$."

**Step 4: Swarm Logic**

> "Implement the `kelly_sizing.py` logic. It should take the $P_{\text{MCTS}}$ and $P_{\text{Market}}$ to determine the optimal bet size while capping total portfolio risk at 2% per event."

---

### 7. SAFETY & CONSTRAINTS

* **The "Mule" Protection:** If the latent state $z$ enters a region of the embedding space never seen in training (Out-of-Distribution), the system must liquidate all positions.
* **Sentiment Loop:** The system must account for the fact that *prediction markets can influence the events they track*. This "Self-Fulfilling Prophecy" must be modeled as a feedback loop in the MCTS.

---

**End of Specification.**

**Would you like me to start by writing the Python code for the `multimodal_encoder.py` using a JEPA framework, so you have the first piece of the puzzle ready?**