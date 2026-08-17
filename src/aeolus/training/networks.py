"""Masked task-pointer actor with a centralized value function."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from aeolus.core.tasks import (
    ACTOR_GLOBAL_FEATURE_DIM,
    CRITIC_GLOBAL_FEATURE_DIM,
    RESOURCE_FEATURE_DIM,
    TASK_FEATURE_DIM,
)


class TaskPointerActorCritic(nn.Module):
    """Parameter-shared actor and centralized critic for variable task sets.

    Actors receive one resource feature vector and the candidate task set. The
    critic pools all resources/tasks plus the global training state. The GRU
    state is per resource, preserving a temporal channel for delayed
    observations without allowing the actor to read the truth state.
    """

    def __init__(self, hidden_dim: int = 192):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_encoder = nn.Sequential(
            nn.Linear(TASK_FEATURE_DIM, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.resource_encoder = nn.Sequential(
            nn.Linear(RESOURCE_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.actor_global_encoder = nn.Sequential(
            nn.Linear(ACTOR_GLOBAL_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.critic_global_encoder = nn.Sequential(
            nn.Linear(CRITIC_GLOBAL_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.actor_gru = nn.GRUCell(hidden_dim * 3, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.task_bias = nn.Linear(TASK_FEATURE_DIM, 1, bias=False)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        resource: Tensor,
        tasks: Tensor,
        action_mask: Tensor,
        actor_global_state: Tensor,
        critic_global_state: Tensor,
        hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return masked logits `[B,N,K]`, team value `[B]`, and next hidden."""

        batch, agents, _ = resource.shape
        task_embedding = self.task_encoder(tasks)  # B,K,H
        resource_embedding = self.resource_encoder(resource)  # B,N,H
        actor_global_embedding = self.actor_global_encoder(actor_global_state)  # B,H
        critic_global_embedding = self.critic_global_encoder(critic_global_state)  # B,H
        valid = action_mask.any(dim=1).float().unsqueeze(-1)  # B,K,1
        task_pool = (task_embedding * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        global_per_agent = actor_global_embedding.unsqueeze(1).expand(-1, agents, -1)
        task_per_agent = task_pool.unsqueeze(1).expand(-1, agents, -1)
        actor_input = torch.cat((resource_embedding, task_per_agent, global_per_agent), dim=-1)
        if hidden is None:
            hidden = resource.new_zeros((batch, agents, self.hidden_dim))
        next_hidden = self.actor_gru(
            actor_input.reshape(batch * agents, -1), hidden.reshape(batch * agents, -1)
        )
        next_hidden = next_hidden.reshape(batch, agents, self.hidden_dim)
        logits = (
            torch.einsum("bnh,bkh->bnk", self.query(next_hidden), self.key(task_embedding))
            / self.hidden_dim**0.5
        )
        logits = logits + self.task_bias(tasks).squeeze(-1).unsqueeze(1)
        logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        resource_pool = resource_embedding.mean(dim=1)
        value = self.critic(
            torch.cat((resource_pool, task_pool, critic_global_embedding), dim=-1)
        ).squeeze(-1)
        return logits, value, next_hidden

    def act(
        self,
        resource: Tensor,
        tasks: Tensor,
        action_mask: Tensor,
        actor_global_state: Tensor,
        critic_global_state: Tensor,
        hidden: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        logits, value, next_hidden = self(
            resource,
            tasks,
            action_mask,
            actor_global_state,
            critic_global_state,
            hidden,
        )
        distribution = torch.distributions.Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), distribution.entropy(), value, next_hidden

    def evaluate_actions(
        self,
        resource: Tensor,
        tasks: Tensor,
        action_mask: Tensor,
        actor_global_state: Tensor,
        critic_global_state: Tensor,
        hidden: Tensor,
        actions: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits, value, _ = self(
            resource,
            tasks,
            action_mask,
            actor_global_state,
            critic_global_state,
            hidden,
        )
        distribution = torch.distributions.Categorical(logits=logits)
        return distribution.log_prob(actions), distribution.entropy(), value
