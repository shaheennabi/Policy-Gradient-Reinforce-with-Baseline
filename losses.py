import torch
import torch.nn.functional as F


def policy_loss(log_probs, advantage, policy_optim):
    loss = - (torch.stack(log_probs) * advantage).mean()
    policy_optim.zero_grad()  ## resetting
    loss.backward()  ## tracking -autograd
    policy_optim.step()



def value_loss(values, actual_returns, value_optim):
    loss = F.mse_loss(values, actual_returns)
    value_optim.zero_grad()  ## resetting
    loss.backward()  ## tracking -autograd
    value_optim.step()



    