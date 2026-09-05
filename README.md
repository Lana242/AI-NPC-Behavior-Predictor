# 🎮 AI NPC Behavior Predictor

### What if NPCs didn't just follow scripts... but actually *made decisions*?

Think about your favorite games.

for example, in gta6 as rockstar announced how their NPCs will remember things, and they will react to the players action.

or In **Red Dead Redemption 2**, the world feels alive because characters react to the player, their surroundings, and what's happening around them.

In **Halo**, enemy encounters can feel different depending on how you approach them. Bungie has discussed investing heavily in AI systems designed to make encounters more reactive and less predictable.

And then there's **Alien: Isolation**.

The Xenomorph is probably one of the coolest examples of game AI because it doesn't simply wait for a scripted jumpscare. Creative Assembly built a system where the Alien searches for the player using its own behavior and senses, while a separate "director" system manages the overall tension. The result is an enemy that can feel unpredictable without simply cheating.

That's the idea behind this project.

---

<img width="736" height="1104" alt="image" src="https://github.com/user-attachments/assets/913d8ed8-cee5-40e4-865c-92164ae977cd" />
<img width="480" height="720" alt="image" src="https://github.com/user-attachments/assets/24ecdb88-0c95-41d6-87fc-0144eb397384" />


## 🧠 So... Can We Teach an NPC to Make Decisions?

This project explores that question using **Machine Learning + Reinforcement Learning**.

Instead of telling an NPC:

> "If the player is close → attack."

we give the AI information about the situation:

```text
❤️ NPC Health
⚡ Stamina
🎯 Player Health
📏 Distance
👥 Allies Nearby
☠️ Enemies Nearby
🛡️ Cover Available
🚪 Escape Route
👀 Player Visible
🔫 Player Attacking
💥 Ammo
⏱️ Time Since Seen
```

Then the NPC has to decide:

```text
ATTACK
CHASE
PATROL
RETREAT
SEARCH
TAKE COVER
```

Suddenly, the NPC isn't just following one script.

It's choosing.

---

<img width="554" height="554" alt="image" src="https://github.com/user-attachments/assets/1a005ba9-682f-4bc2-a5e0-41ff67d10e5d" />


## 🤖 Two Different Ways to Build the NPC

### 1. Supervised Learning — "What would an NPC normally do?"

The first part of the project learns from **100,000 gameplay observations**.

I tested:

* Logistic Regression
* Random Forest
* XGBoost

The best accuracy came from **XGBoost — 69.74%**.

The best Macro F1 came from **Random Forest — 59.09%**.

Macro F1 was especially important because some behaviors, such as `PATROL`, appear much less frequently than others.

---

### 2. Reinforcement Learning — "What should the NPC do?"

This is where things get more interesting.

Instead of giving the agent the correct answer, I created a simulated environment where its actions have consequences.

For example:

```text
Player attacking?
        ↓
NPC chooses TAKE_COVER
        ↓
NPC becomes safer
        ↓
Reward increases
```

Or:

```text
NPC has low health
        ↓
NPC chooses CHASE
        ↓
NPC becomes more vulnerable
        ↓
Reward decreases
```

The PPO agent learns through this trial-and-error process.

After **100,000 training timesteps**, the agent achieved:

**96.26 mean episode reward**

and produced a diverse behavior policy rather than simply choosing one action all the time.

---

## 🎮 What Makes Game AI So Interesting?

The coolest part of NPC AI isn't making an enemy *strong*.

It's making the enemy feel **alive**.

A good NPC should make you think:

> "Did it actually notice me?"

> "Is it searching for me?"

> "Why did it take cover?"

> "Is it retreating because it's hurt?"

> "Will it come back?"

That's why systems like the Xenomorph in **Alien: Isolation** are so memorable. The AI doesn't need to be genuinely conscious or "learn" like a human. Carefully designed decision systems can create the *illusion* of intelligence and unpredictability. **It never walks the same path twice:** Because the "Director Brain" changes its hints based on your real-time panic levels and position, and **It tracks your gaze:** The Xenomorph's physical "Micro Brain" is programmed to look for human-like movement.

The same idea appears in different forms across games such as:

* **Red Dead Redemption 2** — reactive world and NPC behaviors
* **Halo** — reactive combat encounters
* **Alien: Isolation** — systemic hunting and tension management
* **Left 4 Dead** — dynamic pacing through an AI Director
* **F.E.A.R.** — coordinated enemy combat behaviors
* **Far Cry** — enemies reacting to stealth, combat and environmental situations

Different games use very different technologies, but they all chase the same goal:

### Make the player feel like the world is reacting to them.

---

<img width="600" height="900" alt="image" src="https://github.com/user-attachments/assets/478e9235-ab10-4cb3-a106-8902d52bb51f" />


## 🚀 What This Project Demonstrates

This project combines two sides of AI:

**Supervised Learning**

> Learn from what NPCs have done before.

**Reinforcement Learning**

> Learn what actions lead to better outcomes.

Together, they provide an interesting foundation for future game AI systems where NPCs could become more adaptive, contextual, and unpredictable.

---

## 📊 Final Results

| Approach            |                Result |
| ------------------- | --------------------: |
| Logistic Regression |       61.05% Accuracy |
| Random Forest       |   **59.09% Macro F1** |
| XGBoost             |   **69.74% Accuracy** |
| PPO                 | **96.26 Mean Reward** |

PPO action distribution:

| Behavior   | Frequency |
| ---------- | --------: |
| SEARCH     |    47.89% |
| RETREAT    |    24.08% |
| ATTACK     |    15.22% |
| TAKE_COVER |     6.87% |
| PATROL     |     3.90% |
| CHASE      |     2.04% |

---

## 🛠️ Technical Notes

Built with:

**Python · Pandas · NumPy · Scikit-learn · XGBoost · Gymnasium · Stable-Baselines3 · PPO**

The supervised models were trained using episode-level splits to reduce data leakage between gameplay episodes.

The RL environment uses a custom simulator so that NPC actions can actually influence future states instead of simply replaying recorded data.

The project is primarily an **AI/game-AI exploration and portfolio project**, rather than an attempt to reproduce the proprietary AI systems used by commercial games.

---

## 🎯 The Bigger Idea

Game AI doesn't have to be about creating an NPC that is "smarter than the player."

It is about creating an NPC that makes the player **believe it is thinking**.

And sometimes...

that's what makes a game unforgettable.

## Project Structure

```
AI-NPC-Behavior-Predictor/
│
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── models/
│   ├── ppo_npc_simulator.zip
│   ├── vec_normalize_simulator.pkl
│   └── ppo_simulator_evaluation_results.csv
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_and_split.ipynb
│   ├── 03_preprocessing_pipeline.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_rl_environment.ipynb
│   ├── 06_rl_training.ipynb
│   └── 07_results_and_comparison.ipynb
│
├── src/
│   └── npc_environment.py
│
├── README.md
├── requirements.txt
└── .gitignore
```
