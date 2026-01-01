

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn



def stable_role(agent_id: str) -> int:
    h = hashlib.md5(agent_id.encode("utf-8")).hexdigest()
    return int(h, 16) % 3


def role_one_hot(role: int, k: int = 3) -> np.ndarray:
    v = np.zeros(k, dtype=np.float32)
    v[role % k] = 1.0
    return v



class SharedPolicy(nn.Module):
    def __init__(self, obs_dim=16, hidden_dim=256, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class StudentAgent:
    """
    Predator-only agent.
    During evaluation this class will be instantiated once, then get_action will be called many times.
    """

    def __init__(self):
        self.submission_dir = Path(__file__).parent
        self.device = torch.device("cpu")


        self.model16 = SharedPolicy(obs_dim=16, hidden_dim=256, action_dim=5).to(self.device)
        self.model16.eval()


        self.model14 = SharedPolicy(obs_dim=14, hidden_dim=256, action_dim=5).to(self.device)
        self.model14.eval()

        # Load policy weights
        self.policy_path = self.submission_dir / "mappo_policy.pth"
        self.loaded = False

        if self.policy_path.exists():
            sd = torch.load(str(self.policy_path), map_location="cpu")

           
            try:
                self.model16.load_state_dict(sd, strict=True)
                self.loaded = True
                self._which = "obs16"
            except Exception:
                
                try:
                    self.model14.load_state_dict(sd, strict=True)
                    self.loaded = True
                    self._which = "obs14"
                except Exception as e:
                    # Keep unloaded; fallback heuristic will be used
                    self.loaded = False
                    self._which = f"unloaded ({type(e).__name__})"
        else:
            self.loaded = False
            self._which = "missing_file"

    def get_action(self, observation, agent_id: str):
        """
        Args:
            observation: numpy array (predator obs, typically 16 dims in simple_tag_v3)
            agent_id: e.g. "adversary_0"
        Returns:
            int action in {0,1,2,3,4}
        """
        # Only predators controlled by students
        if "adversary" not in agent_id:
            return 0

        obs = np.asarray(observation, dtype=np.float32).reshape(-1)

        # If model not loaded, do a small safe fallback
        if not self.loaded:
            return self._fallback(obs)

        # Build input = obs + role
        r = role_one_hot(stable_role(agent_id), 3)
        x = np.concatenate([obs, r], axis=0).astype(np.float32)

        if obs.shape[0] == 16 and self._which == "obs16":
            xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model16(xt)
            return int(torch.argmax(logits, dim=1).item())

        if obs.shape[0] == 14 and self._which == "obs14":
            xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model14(xt)
            return int(torch.argmax(logits, dim=1).item())

        # Mismatch case: still try best-effort (trim/pad)
        return self._fallback(obs)

    def _fallback(self, obs: np.ndarray) -> int:
    
        if obs.shape[0] == 16:
            # Simple heuristic: chase the prey
            other_pos = obs[2 + 2 + 4 : 2 + 2 + 4 + 6].reshape(3, 2)
            prey_rel = other_pos[2]
            x, y = float(prey_rel[0]), float(prey_rel[1])
            ax, ay = abs(x), abs(y)
            if ax < 1e-4 and ay < 1e-4:
                return 0
            if ax >= ay:
                return 2 if x > 0 else 1
            return 4 if y > 0 else 3

       
        return 0
