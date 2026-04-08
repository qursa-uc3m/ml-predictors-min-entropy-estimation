from abc import ABC, abstractmethod

import torch


class DistillationStrategy(ABC):

    @abstractmethod
    def compute_loss(self, student, teacher, x, y, config, device): ...


class RADStrategy(DistillationStrategy):
    """REINFORCE-based Autoregressive Distillation.

    Student generates tokens autoregressively; teacher scores them.
    Loss drives student toward sequences with high joint probability
    under the teacher (MAP approximation for min-entropy).
    """

    def __init__(self, num_steps=5, reward_decay=1.0, temperature=1.0):
        self.num_steps = num_steps
        self.reward_decay = reward_decay
        self.temperature = temperature

    def compute_loss(self, student, teacher, x, y, config, device):
        batch_size = x.size(0)
        target_bits = config["target_bits"]
        eps = 1e-10

        log_probs = []
        joint_prob = torch.ones(batch_size, device=device)
        current_x = x.clone()

        for step in range(self.num_steps):
            output = student(current_x)
            logits = output.logits[:, -1, :] / self.temperature

            probs = torch.softmax(logits, dim=-1)
            actions = torch.distributions.Categorical(probs).sample()

            log_probs_all = torch.log_softmax(logits, dim=-1)
            step_logp = log_probs_all.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            log_probs.append(step_logp)

            with torch.no_grad():
                ref_out = teacher(current_x)
                ref_probs = torch.softmax(ref_out.logits[:, -1, :], dim=-1)
                chosen_ref_probs = ref_probs.gather(-1, actions.unsqueeze(-1)).squeeze(
                    -1
                )
                joint_prob = joint_prob * (chosen_ref_probs**self.reward_decay)

            current_x = torch.cat([current_x[:, 1:], actions.unsqueeze(1)], dim=1)

        min_entropy_per_bit = -torch.log2(joint_prob + eps) / (
            self.num_steps * target_bits
        )
        rewards = -min_entropy_per_bit

        total_logp = sum(log_probs)
        advantages = rewards - rewards.mean()
        return -(total_logp * advantages).mean()


class VADStrategy(DistillationStrategy):
    """Variational AD via Gumbel-Softmax (not yet implemented)."""

    def __init__(self, num_steps=5, tau_start=1.0, tau_end=0.1, **kwargs):
        self.num_steps = num_steps
        self.tau_start = tau_start
        self.tau_end = tau_end

    def compute_loss(self, student, teacher, x, y, config, device):
        raise NotImplementedError("VAD: Gumbel-Softmax relaxation not yet implemented.")


class IRBCStrategy(DistillationStrategy):
    """Iterative Refinement + Behaviour Cloning (not yet implemented)."""

    def __init__(self, num_steps=5, num_candidates=16, **kwargs):
        self.num_steps = num_steps
        self.num_candidates = num_candidates

    def compute_loss(self, student, teacher, x, y, config, device):
        raise NotImplementedError(
            "IRBC: teacher sampling + cloning not yet implemented."
        )
