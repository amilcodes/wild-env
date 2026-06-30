"""Masked task-pointer actor with a centralized value function."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from aeolus.config import TrainingConfig
from aeolus.core.tasks import (
    ACTOR_GLOBAL_FEATURE_DIM,
    CRITIC_GLOBAL_FEATURE_DIM,
    RESOURCE_FEATURE_DIM,
    TASK_CAPACITY_SCALE,
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
        value = self.critic(torch.cat((resource_pool, task_pool, critic_global_embedding), dim=-1)).squeeze(
            -1
        )
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
        actions, log_prob, entropy = self._capacity_aware_actions(
            logits,
            tasks,
            action_mask,
            deterministic=deterministic,
        )
        return actions, log_prob, entropy, value, next_hidden

    @staticmethod
    def _capacity_aware_actions(
        logits: Tensor,
        tasks: Tensor,
        action_mask: Tensor,
        *,
        actions: Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Autoregressively enforce candidate capacity in the joint action.

        The task pointer still factorizes policy scores by resource.  Sampling
        conditions each subsequent resource on capacity consumed by preceding
        assignments, which removes the environment-side rejection noise that a
        set of independent categorical samples otherwise creates.
        """

        batch, agents, task_count = logits.shape
        remaining = torch.clamp(
            torch.round(tasks[..., 7] * TASK_CAPACITY_SCALE),
            min=1,
        ).long()
        remaining[:, 0] = agents
        selected: list[Tensor] = []
        log_prob: list[Tensor] = []
        entropy: list[Tensor] = []
        batch_index = torch.arange(batch, device=logits.device)
        for agent_index in range(agents):
            conditional_mask = action_mask[:, agent_index] & (remaining > 0)
            conditional_mask[:, 0] = True
            conditional_logits = logits[:, agent_index].masked_fill(
                ~conditional_mask,
                torch.finfo(logits.dtype).min,
            )
            distribution = torch.distributions.Categorical(logits=conditional_logits)
            if actions is None:
                action = conditional_logits.argmax(dim=-1) if deterministic else distribution.sample()
            else:
                action = actions[:, agent_index]
            selected.append(action)
            log_prob.append(distribution.log_prob(action))
            entropy.append(distribution.entropy())
            used = action != 0
            remaining[batch_index[used], action[used]] -= 1
        return (
            torch.stack(selected, dim=1),
            torch.stack(log_prob, dim=1),
            torch.stack(entropy, dim=1),
        )

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
        _, log_prob, entropy = self._capacity_aware_actions(
            logits,
            tasks,
            action_mask,
            actions=actions,
        )
        return log_prob, entropy, value


class EntityAttentionActorCritic(TaskPointerActorCritic):
    """Permutation-equivariant resource/task encoder for shared-airspace teams.

    Resource self-attention represents fleet telemetry and site/attack task
    self-attention represents the current operational graph.  The pointer head
    remains fixed-shape and maskable, while the critic receives privileged
    global state only during training.
    """

    def __init__(
        self,
        hidden_dim: int = 192,
        *,
        attention_heads: int = 4,
        attention_layers: int = 2,
    ):
        nn.Module.__init__(self)
        if hidden_dim % attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.hidden_dim = hidden_dim
        self.task_encoder = nn.Sequential(
            nn.Linear(TASK_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.resource_encoder = nn.Sequential(
            nn.Linear(RESOURCE_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        def encoder_layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=attention_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

        self.task_attention = nn.TransformerEncoder(
            encoder_layer(),
            num_layers=attention_layers,
            enable_nested_tensor=False,
        )
        self.resource_attention = nn.TransformerEncoder(
            encoder_layer(),
            num_layers=attention_layers,
            enable_nested_tensor=False,
        )
        self.actor_global_encoder = nn.Sequential(
            nn.Linear(ACTOR_GLOBAL_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.critic_global_encoder = nn.Sequential(
            nn.Linear(CRITIC_GLOBAL_FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.actor_gru = nn.GRUCell(hidden_dim * 3, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.task_bias = nn.Linear(TASK_FEATURE_DIM, 1, bias=False)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 3, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
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
        batch, agents, _ = resource.shape
        task_valid = action_mask.any(dim=1)
        task_embedding = self.task_attention(
            self.task_encoder(tasks),
            src_key_padding_mask=~task_valid,
        )
        resource_embedding = self.resource_attention(self.resource_encoder(resource))
        actor_global_embedding = self.actor_global_encoder(actor_global_state)
        critic_global_embedding = self.critic_global_encoder(critic_global_state)
        valid_weight = task_valid.to(task_embedding.dtype).unsqueeze(-1)
        task_pool = (task_embedding * valid_weight).sum(dim=1) / valid_weight.sum(dim=1).clamp_min(1.0)
        actor_input = torch.cat(
            (
                resource_embedding,
                task_pool[:, None, :].expand(-1, agents, -1),
                actor_global_embedding[:, None, :].expand(-1, agents, -1),
            ),
            dim=-1,
        )
        if hidden is None:
            hidden = resource.new_zeros((batch, agents, self.hidden_dim))
        next_hidden = self.actor_gru(
            actor_input.reshape(batch * agents, -1),
            hidden.reshape(batch * agents, -1),
        ).reshape(batch, agents, self.hidden_dim)
        logits = (
            torch.einsum(
                "bnh,bkh->bnk",
                self.query(next_hidden),
                self.key(task_embedding),
            )
            / self.hidden_dim**0.5
        )
        logits += self.task_bias(tasks).squeeze(-1).unsqueeze(1)
        logits = logits.masked_fill(
            ~action_mask,
            torch.finfo(logits.dtype).min,
        )
        value = self.critic(
            torch.cat(
                (
                    resource_embedding.mean(dim=1),
                    task_pool,
                    critic_global_embedding,
                ),
                dim=-1,
            )
        ).squeeze(-1)
        return logits, value, next_hidden


def build_policy_network(training: TrainingConfig) -> TaskPointerActorCritic:
    """Construct the checkpoint-compatible policy selected by a manifest."""

    if training.policy_architecture == "entity_attention":
        return EntityAttentionActorCritic(
            training.hidden_dim,
            attention_heads=training.attention_heads,
            attention_layers=training.attention_layers,
        )
    return TaskPointerActorCritic(training.hidden_dim)
