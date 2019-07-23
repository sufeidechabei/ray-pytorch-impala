"""Adapted from A3CTFPolicy to add V-trace.

Keep in sync with changes to A3CTFPolicy and VtraceSurrogatePolicy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch import nn
import gym
import ray
import torch
import logging
import numpy as np
import vtrace
from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.torch_action_dist import TorchCategorical
from ray.rllib.policy.torch_policy_template import build_torch_policy
from utils import clip_gradient, torch_action_dist

# Frozen logits of the policy that computed the action
logger = logging.getLogger(__name__)

BEHAVIOUR_LOGITS = "behaviour_logits"


class VTraceLoss(object):
    def __init__(self,
                 actions,
                 actions_logp,
                 actions_entropy,
                 dones,
                 behaviour_logits,
                 target_logits,
                 discount,
                 rewards,
                 values,
                 bootstrap_value,
                 dist_class,
                 valid_mask,
                 vf_loss_coeff=0.5,
                 entropy_coeff=0.01,
                 clip_rho_threshold=1.0,
                 clip_pg_rho_threshold=1.0,
                 ):
        """Policy gradient loss with vtrace importance weighting.

                VTraceLoss takes tensors of shape [T, B, ...], where `B` is the
                batch_size. The reason we need to know `B` is for V-trace to properly
                handle episode cut boundaries.

                Args:
                    actions: An int|float32 tensor of shape [T, B, ACTION_SPACE].
                    actions_logp: A float32 tensor of shape [T, B].
                    actions_entropy: A float32 tensor of shape [T, B].
                    dones: A bool tensor of shape [T, B].
                    behaviour_logits: A list with length of ACTION_SPACE of float32
                        tensors of shapes
                        [T, B, ACTION_SPACE[0]],
                        ...,
                        [T, B, ACTION_SPACE[-1]]
                    target_logits: A list with length of ACTION_SPACE of float32
                        tensors of shapes
                        [T, B, ACTION_SPACE[0]],
                        ...,
                        [T, B, ACTION_SPACE[-1]]
                    discount: A float32 scalar.
                    rewards: A float32 tensor of shape [T, B].
                    values: A float32 tensor of shape [T, B].
                    bootstrap_value: A float32 tensor of shape [B].
                    dist_class: action distribution class for logits.
                    valid_mask: A bool tensor of valid RNN input elements (#2992).
                """
        # Compute vtrace on the CPU for the better perf
        device = behaviour_logits[0].get_device()
        if device >= 0:
            device = torch.device("cuda:" + str(device))
        else:
            device = torch.device("cpu")
        for i in range(len(behaviour_logits)):
            behaviour_logits[i] = behaviour_logits[i].data.cpu()
            target_logits[i] = target_logits[i].data.cpu()

            # Make sure tensor ranks are as expected
            # The rest will be checked by from_aciton_log_probs.
            assert len(behaviour_logits[i].size()) == 3
            assert len(target_logits[i].size()) == 3
        reverse_dones = (dones.float() == torch.zeros_like(dones.float()))
        self.vtrace_returns = vtrace.multi_from_logits(behaviour_policy_logits=behaviour_logits,
                                                       target_policy_logits=target_logits,
                                                       actions=torch.unbind(actions.data.cpu(), dim=2),
                                                       discounts=reverse_dones.float() * discount,
                                                       rewards=rewards.data.cpu(),
                                                       values=values.data.cpu(),
                                                       bootstrap_value=bootstrap_value.data.cpu(),
                                                       dist_class=dist_class,
                                                       clip_rho_threshold=clip_rho_threshold,
                                                       clip_pg_rho_threshold=clip_pg_rho_threshold)

        self.value_targets = self.vtrace_returns.vs
        valid_mask = (valid_mask == torch.ones_like(valid_mask).to(device))
        self.pi_loss = -torch.sum((actions_logp * self.vtrace_returns.pg_advantages.to(device))[valid_mask])
        self.valid_mask = valid_mask
        self.values = values
        self.logp_min = actions_logp.min()
        self.logp_mean = actions_logp.mean()
        self.logp_max = actions_logp.max()
        self.ad_min = self.vtrace_returns.pg_advantages.min()
        self.ad_mean = self.vtrace_returns.pg_advantages.mean()
        self.ad_max = self.vtrace_returns.pg_advantages.max()
        delta = (values - self.vtrace_returns.vs)[valid_mask]
        self.vf_loss = 0.5 * (delta ** 2).sum()
        self.entropy = torch.sum(actions_entropy[valid_mask])
        self.total_loss = (self.pi_loss + self.vf_loss * vf_loss_coeff - self.entropy * entropy_coeff)


def _make_time_major(policy, tensor, drop_last=False):
    """Swaps batch and trajectory axis.

        Arguments:
            policy: Policy reference
            tensor: A tensor or list of tensors to reshape.
            drop_last: A bool indicating whether to drop the last
            trajectory item.

        Returns:
            res: A tensor with swapped axes or a list of tensors with
            swapped axes.
    """
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        return [_make_time_major(policy, t, drop_last) for t in tensor]

    if policy.state_in:
        B = len(policy.seq_lens)
        T = len(tensor) // B
    else:
        T = policy.config['sample_batch_size']
        B = len(tensor) // T
    if len(tensor.size()) == 2:
        shape = tuple([B, T, int(tensor.size()[1])])
    elif len(tensor.size()) == 1:
        shape = (B, T)
    else:
        shape = tuple([B, T, list(tensor.size()[1:])])

    rs = tensor.view(shape)
    # swap B and T axes
    res =rs.permute([1, 0] + list(range(2, 1 + len(tensor.size()))))
    if drop_last:
        return res[:-1]
    return res


def build_vtrace_loss(policy, batch_tensors):
    if isinstance(policy.action_space, gym.spaces.Discrete):
        is_multidiscrete = False
    elif isinstance(policy.action_space,
                    gym.spaces.multi_discrete.MultiDiscrete):
        is_multidiscrete = True
    else:
        is_multidiscrete = False

    def make_time_major(*args, **kw):
        return _make_time_major(policy, *args, **kw)
    actions = batch_tensors[SampleBatch.ACTIONS]
    print("=========================================")
    print(policy.model)
    logits, _, values, _ = policy.model({SampleBatch.CUR_OBS: batch_tensors[SampleBatch.CUR_OBS]}, [])
    policy.logits = logits
    dones = batch_tensors[SampleBatch.DONES]
    rewards = batch_tensors[SampleBatch.REWARDS]
    device_id = actions.get_device()
    if device_id < 0:
        device = torch.device("cpu")
    else:
        device = torch.device("gpu:" + str(device_id))
    #dones = torch.ones_like(dones).to(device)
    #rewards = torch.ones_like(rewards).to(device)
    #actions = torch.ones_like(actions).to(device)
    behavior_logits = batch_tensors[BEHAVIOUR_LOGITS]
    unpacked_behavior_logits = [behavior_logits]
    unpacked_outputs = [logits]
    policy.dist_class = torch_action_dist
    policy.action_dist = policy.dist_class(logits)
    action_dist = policy.action_dist
    policy.state_in = None

    if policy.state_in:
        max_seq_len = torch.max(policy.seq_lens) - 1
        mask = torch.zeros((len(policy.seq_lens), max_seq_len))
        for i in range(policy.seq_lens):
            mask[:, :policy.seq_lens[i]] = 1
    else:
        mask = torch.ones_like(rewards)
    # Prepare actions for loss
    loss_actions = actions if is_multidiscrete else actions.unsqueeze(dim=1)
    policy.loss = VTraceLoss(
        actions=make_time_major(loss_actions, drop_last=True),
        actions_logp=make_time_major(action_dist.logp(actions), drop_last=True),
        actions_entropy=make_time_major(action_dist.entropy(), drop_last=True),
        dones=make_time_major(dones, drop_last=True),
        behaviour_logits=make_time_major(unpacked_behavior_logits, drop_last=True),
        target_logits=make_time_major(unpacked_outputs, drop_last=True),
        discount=policy.config["gamma"],
        rewards=make_time_major(rewards, drop_last=True),
        values=make_time_major(values, drop_last=True),
        bootstrap_value=make_time_major(values)[-1],
        dist_class=TorchCategorical if is_multidiscrete else policy.dist_class,
        valid_mask=make_time_major(mask, drop_last=True),
        vf_loss_coeff=policy.config['vf_loss_coeff'],
        entropy_coeff=policy.config['entropy_coeff'],
        clip_rho_threshold=policy.config["vtrace_clip_rho_threshold"],
        clip_pg_rho_threshold=policy.config["vtrace_clip_pg_rho_threshold"],
        )
    return policy.loss.total_loss


def stats(policy, batch_tensors):
    return {
        "policy_loss": policy.loss.pi_loss,
        "entropy": policy.loss.entropy,
        'policy_total_loss': policy.loss.total_loss,
        'vf_loss': policy.loss.vf_loss,
        'values_shape': policy.loss.values.size(),
        'logp_min': policy.loss.logp_min,
        'logp_mean': policy.loss.logp_mean,
        'logp_max': policy.loss.logp_max,
        'ad_min': policy.loss.ad_min,
        'ad_mean': policy.loss.ad_mean,
        'ad_max': policy.loss.ad_max,
        'valid_mask_sum': policy.loss.valid_mask.sum()
    }


def postprocess_trajectory(policy,
                           sample_batch,
                           other_agent_batches=None,
                           episode=None):
    # not used, so save some bandwidth
    del sample_batch.data[SampleBatch.NEXT_OBS]
    return sample_batch


def apply_grad_clipping(policy):
    info = {}
    grad_list = []
    if policy.config["grad_clip"]:
        grad_norm = clip_gradient(policy.model, policy.config["grad_clip"])
        info["grad_gnorm"] = grad_norm
        for parameter in policy.model.parameters():
            if parameter.grad is not None:
                grad_list.append(parameter.grad.data.cpu().numpy().sum())
    info['grad_sum'] = grad_list
    return info


def add_behaviour_logits(policy, input_dict, state_batches, model_out):
    return {BEHAVIOUR_LOGITS: model_out[0].data.cpu().numpy()}


def validate_config(policy, obs_space, action_space, config):
    assert config["batch_mode"] == "truncate_episodes", \
        "Must use `truncate_episodes` batch mode with V-trace."


def choose_optimizer(policy, config):
    if policy.config["opt_type"] == "adam":
        return torch.optim.Adam(policy.model.parameters(), lr=config['lr'])
    else:
        return torch.optim.RMSprop(policy.model.parameters(), lr=config['lr'],
                                   eps=config["epsilon"],
                                   weight_decay=config["decay"],
                                   momentum=config["momentum"])


class ValueNetworkMixin(object):
    def _value(self, obs):
        with self.lock:
            obs = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            vf = self.model({"obs": obs}, [])
            return vf[2][0].detach().cpu().numpy()


VTraceTorchPolicy = build_torch_policy(
    name="VTraceTorchPolicy",
    get_default_config=lambda: ray.rllib.agents.impala.impala.DEFAULT_CONFIG,
    loss_fn=build_vtrace_loss,
    stats_fn=stats,
    postprocess_fn=postprocess_trajectory,
    optimizer_fn=choose_optimizer,
    extra_action_out_fn=add_behaviour_logits,
    extra_grad_process_fn=apply_grad_clipping,
    mixins=[ValueNetworkMixin]
    )






