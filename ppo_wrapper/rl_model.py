import torch 

from ppo_wrapper.actor_critic import ActorCritic

import numpy

class RLModel:
    def __init__(self, name, type="mlp", acargs=(12,12,2), ackwargs = {'actor_hidden_dims': [128,128], 'critic_hidden_dims': [128,128]}):
        path = f"/root/catkin_ws/src/hound_core/src/models/{name}"
        loaded_dict = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if type == "mlp":
            self.Model = ActorCritic(*acargs, **ackwargs)
        else:
            raise ValueError("Invalid model type")
        self.Model.load_state_dict(loaded_dict["model_state_dict"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Model.to(self.device)

    def inference(self, state):
        state = torch.Tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            actions = self.Model.act_inference(state).squeeze(0).numpy().astype(numpy.float32)
            clipped_actions = numpy.clip(actions, -1, 1)
            return clipped_actions
        
    def get_value(self, state):
        state = torch.Tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.Model.evaluate(state).squeeze(0).numpy().astype(numpy.float32)