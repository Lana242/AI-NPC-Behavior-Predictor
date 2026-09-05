import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


ACTION_MAP = {
    0: "ATTACK",
    1: "CHASE",
    2: "PATROL",
    3: "RETREAT",
    4: "SEARCH",
    5: "TAKE_COVER",
}


OBSERVATION_COLUMNS = [
    "npc_health",
    "npc_stamina",
    "player_health",
    "distance_to_player",
    "allies_nearby",
    "enemies_nearby",
    "has_cover",
    "escape_route",
    "player_visible",
    "player_attacking",
    "npc_ammo",
    "time_since_seen",
]


def calculate_reward(state, action):
    action_name = ACTION_MAP[int(action)]

    health = float(state["npc_health"])
    stamina = float(state["npc_stamina"])
    distance = float(state["distance_to_player"])
    allies = int(state["allies_nearby"])
    enemies = int(state["enemies_nearby"])
    has_cover = int(state["has_cover"])
    escape_route = int(state["escape_route"])
    player_visible = int(state["player_visible"])
    player_attacking = int(state["player_attacking"])
    ammo = int(state["npc_ammo"])

    reward = 0.0

    # Low health
    if health < 30:
        if action_name == "RETREAT":
            reward += 6
        elif action_name == "TAKE_COVER":
            reward += 5
        elif action_name == "ATTACK":
            reward -= 5
        elif action_name == "CHASE":
            reward -= 4

    # Player attacking
    if player_attacking:
        if action_name == "TAKE_COVER":
            reward += 6
        elif action_name == "RETREAT":
            reward += 5
        elif action_name == "ATTACK":
            reward += 1
        elif action_name == "PATROL":
            reward -= 5
        elif action_name == "SEARCH":
            reward -= 4

    # Player visible and close
    if player_visible and distance < 25:
        if action_name == "ATTACK":
            reward += 6
        elif action_name == "CHASE":
            reward += 2
        elif action_name == "SEARCH":
            reward -= 5
        elif action_name == "PATROL":
            reward -= 4

    # Player visible and medium distance
    elif player_visible and distance < 60:
        if action_name == "ATTACK":
            reward += 4
        elif action_name == "CHASE":
            reward += 4
        elif action_name == "SEARCH":
            reward -= 4
        elif action_name == "PATROL":
            reward -= 3

    # Player visible and far
    elif player_visible and distance >= 60:
        if action_name == "CHASE":
            reward += 5
        elif action_name == "ATTACK":
            reward += 1
        elif action_name == "SEARCH":
            reward -= 3

    # Player not visible
    if not player_visible:
        if action_name == "SEARCH":
            reward += 4
        elif action_name == "PATROL":
            reward += 2
        elif action_name == "ATTACK":
            reward -= 4
        elif action_name == "CHASE":
            reward -= 1

    # No ammunition
    if ammo <= 0:
        if action_name == "ATTACK":
            reward -= 7
        elif action_name == "RETREAT":
            reward += 2
        elif action_name == "SEARCH":
            reward += 1

    # Cover
    if has_cover and player_attacking:
        if action_name == "TAKE_COVER":
            reward += 4
        elif action_name == "RETREAT":
            reward += 2

    # Escape route
    if escape_route and health < 40 and action_name == "RETREAT":
        reward += 3

    # Allies vs enemies
    if allies > enemies:
        if action_name == "ATTACK":
            reward += 3
        elif action_name == "RETREAT":
            reward -= 2

    elif enemies > allies:
        if action_name == "RETREAT":
            reward += 3
        elif action_name == "ATTACK":
            reward -= 2

    # Safe unseen situation
    if not player_visible and not player_attacking:
        if action_name == "PATROL":
            reward += 3

    # Low stamina
    if stamina < 25:
        if action_name == "CHASE":
            reward -= 4
        elif action_name == "PATROL":
            reward += 1

    # Bad action choices
    if action_name == "TAKE_COVER" and not has_cover:
        reward -= 3

    if action_name == "SEARCH" and player_visible:
        reward -= 4

    if action_name == "PATROL" and player_attacking:
        reward -= 4

    # Small health bonus
    if health > 70:
        reward += 0.5

    # Small step cost
    reward -= 0.1

    return float(reward)


class NPCBehaviorEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, data, max_steps=60):
        super().__init__()

        self.data = data.reset_index(drop=True).copy()
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(len(ACTION_MAP))

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(OBSERVATION_COLUMNS),),
            dtype=np.float32,
        )

        self.state = None
        self.current_step = 0

    def _normalize_observation(self, state):
        values = []

        for column in OBSERVATION_COLUMNS:
            value = float(state[column])

            if column in [
                "npc_health",
                "npc_stamina",
                "player_health",
                "distance_to_player",
            ]:
                value = value / 100.0

            elif column in [
                "allies_nearby",
                "enemies_nearby",
                "npc_ammo",
                "time_since_seen",
            ]:
                value = min(value / 10.0, 1.0)

            else:
                value = float(np.clip(value, 0, 1))

            values.append(value)

        return np.asarray(values, dtype=np.float32)

    def _get_observation(self):
        return self._normalize_observation(self.state)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        random_index = self.np_random.integers(0, len(self.data))
        row = self.data.iloc[random_index].copy()

        self.state = {
            column: row[column]
            for column in OBSERVATION_COLUMNS
        }

        self.state["npc_health"] = float(
            np.clip(self.state["npc_health"], 1, 100)
        )
        self.state["npc_stamina"] = float(
            np.clip(self.state["npc_stamina"], 1, 100)
        )
        self.state["player_health"] = float(
            np.clip(self.state["player_health"], 1, 100)
        )
        self.state["distance_to_player"] = float(
            np.clip(self.state["distance_to_player"], 1, 100)
        )

        self.current_step = 0

        observation = self._get_observation()

        return observation, {}

    def step(self, action):

        action = int(action)
        action_name = ACTION_MAP[action]

        reward = calculate_reward(self.state, action)

        # -----------------------------
        # NPC action effects
        # -----------------------------

        if action_name == "ATTACK":
            if self.state["npc_ammo"] > 0:
                self.state["npc_ammo"] -= 1
                self.state["player_health"] -= np.random.uniform(5, 15)
                self.state["npc_stamina"] -= np.random.uniform(3, 7)

        elif action_name == "CHASE":
            self.state["distance_to_player"] -= np.random.uniform(5, 15)
            self.state["npc_stamina"] -= np.random.uniform(4, 8)

        elif action_name == "PATROL":
            self.state["distance_to_player"] += np.random.uniform(-5, 5)
            self.state["npc_stamina"] += np.random.uniform(3, 7)
            self.state["time_since_seen"] += 1

        elif action_name == "RETREAT":
            self.state["distance_to_player"] += np.random.uniform(8, 18)
            self.state["npc_stamina"] -= np.random.uniform(2, 5)

        elif action_name == "SEARCH":
            self.state["time_since_seen"] += 1

            if np.random.random() < 0.25:
                self.state["player_visible"] = 1
                self.state["time_since_seen"] = 0

        elif action_name == "TAKE_COVER":
            self.state["npc_stamina"] += np.random.uniform(5, 10)

        # -----------------------------
        # World simulation
        # -----------------------------

        self.state["npc_stamina"] += np.random.uniform(-2, 2)

        if self.state["player_visible"] and self.state["distance_to_player"] < 60:
            if np.random.random() < 0.35:
                self.state["player_attacking"] = 1
            else:
                self.state["player_attacking"] = 0
        else:
            self.state["player_attacking"] = 0

        # Player attacks NPC
        if self.state["player_attacking"]:

            damage = np.random.uniform(3, 10)

            if self.state["has_cover"]:
                damage *= 0.4

            if action_name == "TAKE_COVER":
                damage *= 0.5

            self.state["npc_health"] -= damage

        # Visibility changes
        if self.state["player_visible"]:

            if np.random.random() < 0.08:
                self.state["player_visible"] = 0
                self.state["time_since_seen"] = 1

        else:

            if np.random.random() < 0.05:
                self.state["player_visible"] = 1
                self.state["time_since_seen"] = 0

        # Random environment changes
        self.state["allies_nearby"] = int(
            np.clip(
                self.state["allies_nearby"]
                + np.random.choice([-1, 0, 1]),
                0,
                10,
            )
        )

        self.state["enemies_nearby"] = int(
            np.clip(
                self.state["enemies_nearby"]
                + np.random.choice([-1, 0, 1]),
                0,
                10,
            )
        )

        # Clamp state
        self.state["npc_health"] = float(
            np.clip(self.state["npc_health"], 0, 100)
        )

        self.state["npc_stamina"] = float(
            np.clip(self.state["npc_stamina"], 0, 100)
        )

        self.state["player_health"] = float(
            np.clip(self.state["player_health"], 0, 100)
        )

        self.state["distance_to_player"] = float(
            np.clip(self.state["distance_to_player"], 1, 100)
        )

        self.state["time_since_seen"] = int(
            np.clip(self.state["time_since_seen"], 0, 10)
        )

        self.current_step += 1

        terminated = (
            self.state["npc_health"] <= 0
            or self.state["player_health"] <= 0
        )

        truncated = self.current_step >= self.max_steps

        observation = self._get_observation()

        info = {
            "action": action_name,
            "npc_health": self.state["npc_health"],
            "npc_stamina": self.state["npc_stamina"],
            "player_health": self.state["player_health"],
            "distance_to_player": self.state["distance_to_player"],
        }

        return observation, float(reward), terminated, truncated, info