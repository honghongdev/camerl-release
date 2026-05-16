from mav_baselines.torch.recurrent_ppo.recurrent.policies import (
    RecurrentActorCriticCnnPolicy,
    RecurrentActorCriticPolicy,
    RecurrentMultiInputActorCriticPolicy,
)
from stable_baselines3.common.policies import register_policy

MlpLstmPolicy        = RecurrentActorCriticPolicy
CnnLstmPolicy        = RecurrentActorCriticCnnPolicy
MultiInputLstmPolicy = RecurrentMultiInputActorCriticPolicy

register_policy("MlpLstmPolicy",        RecurrentActorCriticPolicy)
register_policy("CnnLstmPolicy",        RecurrentActorCriticCnnPolicy)
register_policy("MultiInputLstmPolicy", RecurrentMultiInputActorCriticPolicy)
