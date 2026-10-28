"""Pure-torch utilities for sensor-only recurrent PPO with a privileged critic."""
import torch


def generalized_advantage(rewards,values,dones,last_value,gamma=.99,lam=.95):
    advantage=torch.zeros_like(rewards)
    carry=torch.zeros_like(last_value)
    for t in reversed(range(len(rewards))):
        next_value=last_value if t==len(rewards)-1 else values[t+1]
        continuation=(~dones[t]).to(rewards.dtype)
        delta=rewards[t]+gamma*next_value*continuation-values[t]
        carry=delta+gamma*lam*continuation*carry
        advantage[t]=carry
    return advantage,advantage+values


def recurrent_actions(model,features,initial_hidden,reset_before):
    """Replay time-major rollout, clearing only histories whose episode ended."""
    hidden=initial_hidden
    outputs=[]
    for t in range(len(features)):
        hidden=hidden*(~reset_before[t])[None,:,None]
        action,hidden=model(features[t,:,None],hidden)
        outputs.append(action[:,0])
    return torch.stack(outputs),hidden


def continuation_advantage(rewards, values, dones, last_value, trainable,
                           next_trainable, gamma=.99, lam=.95):
    """Exclude frozen-prefix transitions and cut returns across phase boundaries.

    trainable describes the actor that produced each transition, not the phase
    after stepping. A successful prefix handoff is therefore still excluded.
    Environment terminal flags must include second-action completion/failure.
    """
    if trainable.shape!=rewards.shape or dones.shape!=rewards.shape:
        raise ValueError('Transition mask shapes must match rewards')
    zero=torch.zeros_like(rewards)
    masked_rewards=torch.where(trainable,rewards,zero)
    masked_values=torch.where(trainable,values,zero)
    bootstrap=torch.where(next_trainable,last_value,torch.zeros_like(last_value))
    # Also reject a training->prefix edge if the caller omitted a terminal flag.
    next_phase=torch.cat([trainable[1:],next_trainable[None]],dim=0)
    cut=dones|~trainable|~next_phase
    advantage,returns=generalized_advantage(masked_rewards,masked_values,cut,bootstrap,gamma,lam)
    return torch.where(trainable,advantage,zero),torch.where(trainable,returns,zero)
