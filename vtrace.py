# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.

In addition to the original paper's code, changes have been made
to support MultiDiscrete action spaces. behaviour_policy_logits,
target_policy_logits and actions parameters in the entry point
multi_from_logits method accepts lists of tensors instead of just
tensors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import torch
import numpy as np
from torch.utils.tensorboard import writer
from ray.rllib.models.torch_action_dist import TorchCategorical

VTraceFromLogitsReturns = collections.namedtuple("VTraceFromLogitsReturns", [
    "vs", "pg_advantages", "log_rhos", "behaviour_action_log_probs",
    "target_action_log_probs"
])

VTraceReturns = collections.namedtuple("VTraceReturns", "vs pg_advantages")
wr = writer.SummaryWriter('./tensorboard')


def log_probs_from_logits_and_actions(policy_logits,
                                      actions,
                                      dist_class=TorchCategorical):
    return multi_log_probs_from_logits_and_actions([policy_logits], [actions], dist_class)[0]


def multi_log_probs_from_logits_and_actions(policy_logits, actions,
                                            dist_class):
    """Computes action log-probs from policy logits and actions.

  In the notation used throughout documentation and comments, T refers to the
  time dimension ranging from 0 to T-1. B refers to the batch size and
  ACTION_SPACE refers to the list of numbers each representing a number of
  actions.

  Args:
    policy_logits: A list with length of ACTION_SPACE of float32
      tensors of shapes
      [T, B, ACTION_SPACE[0]],
      ...,
      [T, B, ACTION_SPACE[-1]]
      with un-normalized log-probabilities parameterizing a softmax policy.
    actions: A list with length of ACTION_SPACE of
      tensors of shapes
      [T, B, ...],
      ...,
      [T, B, ...]
      with actions.

  Returns:
    A list with length of ACTION_SPACE of float32
      tensors of shapes
      [T, B],
      ...,
      [T, B]
      corresponding to the sampling log probability
      of the chosen action w.r.t. the policy.
  """
    log_probs = []
    for i in range(len(policy_logits)):
        p_shape = policy_logits[i].size()
        a_shape = actions[i].size()
        policy_shape = list(torch.cat((torch.tensor([-1]), torch.tensor(p_shape[2:])), dim=0).data.cpu().numpy())
        if len(a_shape) >= 3:
            action_shape = list(torch.cat((torch.tensor([-1]), torch.tensor(a_shape[2:])), dim=0).data.cpu().numpy())
        else:
            action_shape = [-1]

        policy_logits_flat = policy_logits[i].contiguous().view(policy_shape)
        actions_flat = actions[i].contiguous().view(action_shape)
        log_probs.append(dist_class(policy_logits_flat).logp(actions_flat).view(a_shape[:2]))
    return log_probs


def from_logits(behaviour_policy_logits,
                target_policy_logits,
                actions,
                discounts,
                rewards,
                values,
                bootstrap_value,
                dist_class=TorchCategorical,
                clip_rho_threshold=1.0,
                clip_pg_rho_threshold=1.0,
                ):
    """multi_from_logits wrapper used only for tests"""
    res = multi_from_logits([behaviour_policy_logits],
                            [target_policy_logits],
                            [actions],
                            discounts,
                            rewards,
                            values,
                            bootstrap_value,
                            dist_class,
                            clip_rho_threshold=clip_rho_threshold,
                            clip_pg_rho_threshold=clip_pg_rho_threshold)
    return VTraceFromLogitsReturns(vs=res.vs,
                                   pg_advantages=res.pg_advantages,
                                   log_rhos=res.log_rhos,
                                   behaviour_action_log_probs=torch.squeeze(res.behavior_action_log_probs, dim=0),
                                   target_action_log_probs=torch.squeeze(res.target_action_log_probs, dim=0))


def multi_from_logits(behaviour_policy_logits,
                      target_policy_logits,
                      actions,
                      discounts,
                      rewards,
                      values,
                      bootstrap_value,
                      dist_class,
                      clip_rho_threshold=1.0,
                      clip_pg_rho_threshold=1.0):
    r"""V-trace for softmax policies.

  Calculates V-trace actor critic targets for softmax polices as described in

  "IMPALA: Scalable Distributed Deep-RL with
  Importance Weighted Actor-Learner Architectures"
  by Espeholt, Soyer, Munos et al.

  Target policy refers to the policy we are interested in improving and
  behaviour policy refers to the policy that generated the given
  rewards and actions.

  In the notation used throughout documentation and comments, T refers to the
  time dimension ranging from 0 to T-1. B refers to the batch size and
  ACTION_SPACE refers to the list of numbers each representing a number of
  actions.

  Args:
    behaviour_policy_logits: A list with length of ACTION_SPACE of float32
      tensors of shapes
      [T, B, ACTION_SPACE[0]],
      ...,
      [T, B, ACTION_SPACE[-1]]
      with un-normalized log-probabilities parameterizing the softmax behaviour
      policy.
    target_policy_logits: A list with length of ACTION_SPACE of float32
      tensors of shapes
      [T, B, ACTION_SPACE[0]],
      ...,
      [T, B, ACTION_SPACE[-1]]
      with un-normalized log-probabilities parameterizing the softmax target
      policy.
    actions: A list with length of ACTION_SPACE of
      tensors of shapes
      [T, B, ...],
      ...,
      [T, B, ...]
      with actions sampled from the behaviour policy.
    discounts: A float32 tensor of shape [T, B] with the discount encountered
      when following the behaviour policy.
    rewards: A float32 tensor of shape [T, B] with the rewards generated by
      following the behaviour policy.
    values: A float32 tensor of shape [T, B] with the value function estimates
      wrt. the target policy.
    bootstrap_value: A float32 of shape [B] with the value function estimate at
      time T.
    dist_class: action distribution class for the logits.
    clip_rho_threshold: A scalar float32 tensor with the clipping threshold for
      importance weights (rho) when calculating the baseline targets (vs).
      rho^bar in the paper.
    clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold
      on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)).
    name: The name scope that all V-trace operations will be created in.

  Returns:
    A `VTraceFromLogitsReturns` namedtuple with the following fields:
      vs: A float32 tensor of shape [T, B]. Can be used as target to train a
          baseline (V(x_t) - vs_t)^2.
      pg_advantages: A float 32 tensor of shape [T, B]. Can be used as an
        estimate of the advantage in the calculation of policy gradients.
      log_rhos: A float32 tensor of shape [T, B] containing the log importance
        sampling weights (log rhos).
      behaviour_action_log_probs: A float32 tensor of shape [T, B] containing
        behaviour policy action log probabilities (log \mu(a_t)).
      target_action_log_probs: A float32 tensor of shape [T, B] containing
        target policy action probabilities (log \pi(a_t)).
  """

    for i in range(len(behaviour_policy_logits)):
        behaviour_policy_logits[i] = behaviour_policy_logits[i].clone()
        target_policy_logits[i] = target_policy_logits[i].clone()

        # Make sure tensor ranks are as expected
        # The rest will be checked by from_aciton_log_probs.
        assert len(behaviour_policy_logits[i].size()) == 3
        assert len(target_policy_logits[i].size()) == 3
    target_action_log_probs = multi_log_probs_from_logits_and_actions(
        target_policy_logits, actions, dist_class)
    behaviour_action_log_probs = multi_log_probs_from_logits_and_actions(
        behaviour_policy_logits, actions, dist_class)
    log_rhos = get_log_rhos(target_action_log_probs, behaviour_action_log_probs)
    vtrace_returns = from_importance_weights(
        log_rhos=log_rhos,
        discounts=discounts,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold)

    return VTraceFromLogitsReturns(
        log_rhos=log_rhos,
        behaviour_action_log_probs=behaviour_action_log_probs,
        target_action_log_probs=target_action_log_probs,
        **vtrace_returns._asdict())


def from_importance_weights(log_rhos,
                            discounts,
                            rewards,
                            values,
                            bootstrap_value,
                            clip_rho_threshold=1.0,
                            clip_pg_rho_threshold=1.0,
                            ):
    r"""V-trace from log importance weights.

  Calculates V-trace actor critic targets as described in

  "IMPALA: Scalable Distributed Deep-RL with
  Importance Weighted Actor-Learner Architectures"
  by Espeholt, Soyer, Munos et al.

  In the notation used throughout documentation and comments, T refers to the
  time dimension ranging from 0 to T-1. B refers to the batch size. This code
  also supports the case where all tensors have the same number of additional
  dimensions, e.g., `rewards` is [T, B, C], `values` is [T, B, C],
  `bootstrap_value` is [B, C].

  Args:
    log_rhos: A float32 tensor of shape [T, B] representing the
      log importance sampling weights, i.e.
      log(target_policy(a) / behaviour_policy(a)). V-trace performs operations
      on rhos in log-space for numerical stability.
    discounts: A float32 tensor of shape [T, B] with discounts encountered when
      following the behaviour policy.
    rewards: A float32 tensor of shape [T, B] containing rewards generated by
      following the behaviour policy.
    values: A float32 tensor of shape [T, B] with the value function estimates
      wrt. the target policy.
    bootstrap_value: A float32 of shape [B] with the value function estimate at
      time T.
    clip_rho_threshold: A scalar float32 tensor with the clipping threshold for
      importance weights (rho) when calculating the baseline targets (vs).
      rho^bar in the paper. If None, no clipping is applied.
    clip_pg_rho_threshold: A scalar float32 tensor with the clipping threshold
      on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)). If
      None, no clipping is applied.

  Returns:
    A VTraceReturns namedtuple (vs, pg_advantages) where:
      vs: A float32 tensor of shape [T, B]. Can be used as target to
        train a baseline (V(x_t) - vs_t)^2.
      pg_advantages: A float32 tensor of shape [T, B]. Can be used as the
        advantage in the calculation of policy gradients.
  """

    log_rhos = log_rhos.clone()
    discounts = discounts.clone()
    rewards = rewards.clone()
    values = values.clone()
    bootstrap_value = bootstrap_value.clone()
    rho_rank = len(log_rhos.size())
    device_id = log_rhos.get_device()
    device = torch.device("cuda:" + str(device_id)) if device_id >= 0 else torch.device("cpu")
    assert len(bootstrap_value.size()) == rho_rank - 1
    assert len(discounts.size()) == rho_rank
    assert len(rewards.size()) == rho_rank
    assert len(values.size()) == rho_rank
    rhos = torch.exp(log_rhos)
    clip_rho_threshold = torch.tensor(clip_rho_threshold).to(device)
    clip_pg_rho_threshold = torch.tensor(clip_pg_rho_threshold).to(device)
    if clip_rho_threshold is not None:
        clipped_rhos = torch.min(clip_rho_threshold, rhos)
        wr.add_histogram('clipped_rhos_1000', torch.min(torch.tensor(1000.0).to(device), rhos))
        wr.add_scalar('num_of_clipped_rhos', torch.sum(clipped_rhos == clip_rho_threshold))
        wr.add_scalar('size_of_clipped_rhos', clipped_rhos.numel())
    else:
        clipped_rhos = rhos
    cs = torch.min(torch.tensor(1.0).to(device), rhos)
    values_t_plus_1 = torch.cat([values[1:], bootstrap_value.unsqueeze(dim=0)], dim=0)
    deltas = clipped_rhos * (rewards + discounts * values_t_plus_1 - values)
    sequences = (
        torch.flip(discounts, dims=[0]),
        torch.flip(cs, dims=[0]),
        torch.flip(deltas, dims=[0])
    )

    # V-trace vs are calculated through a scan from the back to the
    # beginning of the given trajectory.

    initial_values = torch.zeros(bootstrap_value.size()).unsqueeze(dim=0)
    vs_minus_v_xs = sequences[2] + sequences[0] * sequences[1] * initial_values
    vs = vs_minus_v_xs + values
    vs_t_plus_1 = torch.cat((vs[1:], bootstrap_value.unsqueeze(dim=0)), dim=0)
    if clip_pg_rho_threshold is not None:
        clipped_pg_rhos = torch.min(clip_pg_rho_threshold, rhos)
    else:
        clipped_pg_rhos = rhos
    pg_advantages = (
            clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))
    return VTraceReturns(
        vs=vs,
        pg_advantages=pg_advantages)


def get_log_rhos(target_action_log_probs, behaviour_action_log_probs):
    """With the selected log_probs for multi-discrete actions of behaviour
        and target policies we compute the log_rhos for calculating the vtrace."""
    t = torch.stack(target_action_log_probs)
    b = torch.stack(behaviour_action_log_probs)
    log_rhos = torch.sum(t-b, dim=0)
    return log_rhos




















