import torch
from torch import nn

class BaseModel(nn.Module):
    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def get_weights_by_list(self):
        weights = []
        for p in self.parameters():
            weights.append(p.clone())
        return weights

    def set_weights(self, weights):
        self.load_state_dict(weights, strict=False)

    def set_weights_by_list(self, weights):
        for w, p in zip(weights, self.named_parameters()):
            p[1].data = w

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.named_parameters()):
            if g is not None:
                p[1].grad = g

    def H_pow(self, p):
        temp_weight = self.get_weights()
        for key in temp_weight:
            # temp_weight[key] = torch.pow(temp_weight[key], p) # may cause nan
            temp_weight[key] = torch.sign(temp_weight[key]) * torch.pow(torch.abs(temp_weight[key]), p)
        self.set_weights(temp_weight)

    def delta_sub(self, prev_model, curr_model):
        delta = []
        # print(f'prev_model = {prev_model[0]}')
        # print(f'curr_model = {curr_model[0]}')
        for idx in range(len(curr_model)):
            temp = torch.add(-curr_model[idx], prev_model[idx])
            delta.append(temp)
        return delta
