import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 auxiliary_loss_coef=0,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 remove_actor_grads_on_shared=False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.auxiliary_loss_coef = auxiliary_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        #Special one time experiment to see whether removing
        #action and entropy gradients on shared layers influences
        #performance at all
        self.remove_actor_grads_on_shared = remove_actor_grads_on_shared

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        auxiliary_loss_epoch = 0

        clipfracs = []
        explained_vars = []
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ, auxiliary_pred_batch, auxiliary_truth_batch = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, auxiliary_preds = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                logratio = action_log_probs - old_action_log_probs_batch 
                ratio = torch.exp(logratio)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                #Andy: compute approx kl
                with torch.no_grad():
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                    ]

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                
                if self.actor_critic.has_auxiliary:
                    auxiliary_loss = 0.5 * (auxiliary_truth_batch - auxiliary_preds).pow(2).mean()
                else:
                    auxiliary_loss = torch.zeros(1)
                # print(auxiliary_truth_batch, auxiliary_preds, auxiliary_loss)

                if self.remove_actor_grads_on_shared:
                    self.optimizer.zero_grad()
                    # First get gradient of everything except for value loss
                    (action_loss + 
                    auxiliary_loss * self.auxiliary_loss_coef -
                    dist_entropy * self.entropy_coef).backward(retain_graph=True)
                    params = list(self.actor_critic.parameters())
                    # Zero out first 4 param sets, as these are the shared layers
                    for i in range(4):
                        params[i].grad = torch.zeros(params[i].shape)
                    (value_loss * self.value_loss_coef).backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                             self.max_grad_norm)
                    self.optimizer.step()
                else:
                    self.optimizer.zero_grad()
                    (value_loss * self.value_loss_coef + action_loss + 
                    auxiliary_loss * self.auxiliary_loss_coef -
                    dist_entropy * self.entropy_coef).backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                            self.max_grad_norm)
                    self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                auxiliary_loss_epoch += auxiliary_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates



        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, \
            approx_kl, clipfracs, auxiliary_loss_epoch



class PPOAux():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 auxiliary_loss_coef=0,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 remove_actor_grads_on_shared=False):

        self.actor_critic = actor_critic
        self.auxiliary_types = actor_critic.base.auxiliary_layer_types
        self.cross_entropy_loss = nn.CrossEntropyLoss() #used for multiclass aux loss

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.auxiliary_loss_coef = auxiliary_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        
        self.remove_actor_grads_on_shared = remove_actor_grads_on_shared

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        auxiliary_loss_epoch = 0

        clipfracs = []
        explained_vars = []
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                    adv_targ, auxiliary_pred_batch, auxiliary_truth_batch = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _, auxiliary_preds = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                logratio = action_log_probs - old_action_log_probs_batch 
                ratio = torch.exp(logratio)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                #Andy: compute approx kl
                with torch.no_grad():
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_param).float().mean().item()
                    ]

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                
                if self.actor_critic.has_auxiliary:
                    # auxiliary_loss = torch.tensor(0.)
                    auxiliary_losses = torch.zeros(len(self.auxiliary_types))
                    for i, aux_type in enumerate(self.auxiliary_types):
                        if aux_type == 0:
                            # auxiliary_loss += 0.5 * (auxiliary_truth_batch[i] - auxiliary_preds[i]).pow(2).mean()
                            auxiliary_losses[i] += 0.5 * (auxiliary_truth_batch[i] - auxiliary_preds[i]).pow(2).mean()
                        elif aux_type == 1:
                            # auxiliary_loss += self.cross_entropy_loss(auxiliary_preds[i], auxiliary_truth_batch[i])
                            auxiliary_losses[i] += self.cross_entropy_loss(auxiliary_preds[i], auxiliary_truth_batch[i].long().squeeze())
                    auxiliary_loss = auxiliary_losses.sum()
                else:
                    auxiliary_loss = torch.zeros(1)

                if self.remove_actor_grads_on_shared:
                    self.optimizer.zero_grad()
                    # First get gradient of everything except for value loss
                    (action_loss + 
                    auxiliary_loss * self.auxiliary_loss_coef -
                    dist_entropy * self.entropy_coef).backward(retain_graph=True)
                    params = list(self.actor_critic.parameters())
                    # Zero out first 4 param sets, as these are the shared layers
                    for i in range(4):
                        params[i].grad = torch.zeros(params[i].shape)
                    (value_loss * self.value_loss_coef).backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                             self.max_grad_norm)
                    self.optimizer.step()
                else:
                    self.optimizer.zero_grad()
                    (value_loss * self.value_loss_coef + action_loss + 
                    auxiliary_loss * self.auxiliary_loss_coef -
                    dist_entropy * self.entropy_coef).backward()
                    nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                            self.max_grad_norm)
                    self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                auxiliary_loss_epoch += auxiliary_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates



        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, \
            approx_kl, clipfracs, auxiliary_loss_epoch
