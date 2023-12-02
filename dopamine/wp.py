import torch
import copy
from optimizer import Dopamine
from torch import nn

torch.manual_seed(0)
torch.cuda.manual_seed(0)


class WeightPerturbation(nn.Module):
    def __init__(
        self,
        model,
        optimizer,
        noise_mean=0.0,
        noise_stddev=1e-4,
        lr=1e-3,
        sr=1.0,
        lambda_reg=1e-8,
    ):
        super(WeightPerturbation, self).__init__()
        # Device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.noise_mean = noise_mean  # Noise mean
        self.noise_stddev = noise_stddev  # Noise standard deviation
        self.lr = lr  # Initial learning rate
        self.sr= sr # Spectral radius
        self.lambda_reg = lambda_reg  # L2 regularization coefficient
        # Initialize model
        self.model = model

        # Initial Reward
        self.reward = 0.0

        # Initial step FLAG
        self.init_FLAG = True

        # Optimizer
        self.optimizer = optimizer

    def l2(self, model):
        l2_norms = list(map(lambda x: torch.sum(x**2), list(model.parameters())))
        l2_norms_sum = sum(l2_norms)

        return l2_norms_sum / (2 * self.batch_size)

    def add_noise(self, param):
        noise = (
            torch.randn_like(param).to(self.device) * self.noise_stddev
            + self.noise_mean
        )
        param.data.add_(noise)

        return param

    def step(self, x):
        self.batch_size = x.shape[0]
        return self.model(x,None)

    def step_hat(self, x):
        self.model_hat = copy.deepcopy(self.model).to(self.device)
        list(
            map(lambda param: self.add_noise(param), list(self.model_hat.parameters()))
        )

        return self.model_hat(x,None)

    def forward(self, x):
        y = self.step(x)
        y_hat = self.step_hat(x)
        return y, y_hat

    def update(self, params, reward):
        psi = params[1] - params[0]  # Perturbation

        params[0].lr = self.optimizer.compute_lr(params[0], reward)

        if len(params[0].size()) == 1: # Bias
            deltaW = -params[0].lr * reward * psi
        else:
            deltaW = -params[0].lr * reward * psi / torch.var(psi, unbiased=False)

        params[0].data.add_(deltaW)

        # if len(params[0].shape) != 1: # Bias
        #     if params[0].shape[-2] == params[0].shape[-1]: # Only square matrix
        #         # Calculate the spectral radius of the parameter.
        #         spectral_radius = torch.linalg.eigvals(params[0].data).abs().max()
        #         # Scale the parameter by a factor such that the spectral radius becomes 1.1.
        #         params[0].data *=self.sr
        #         params[0].data /=spectral_radius

        # Normalize the matrix columnwise
        # if len(params[0].shape) != 1: # Bias
        #     if params[0].shape[-2] == params[0].shape[-1]: # Only square matrix
        #         params[0].data /= params[0].data.max(0, keepdim=True)[0]
        
    def backward(self, reward):
        # Compute L2 reg for model_hat and model
        reg_hat = self.l2(self.model_hat)
        reg = self.l2(self.model)
        reward = reward + self.lambda_reg * (reg_hat - reg)

        if self.init_FLAG:
            for i, param in enumerate(list(self.model.parameters())):
                param.lr = torch.ones_like(param) * self.optimizer.lr
                param.lr.to(self.device)
                param.s = torch.ones_like(param) * self.optimizer.s
                param.s.to(self.device)

        _ = list(
            map(
                lambda params: self.update(params, reward),
                zip(list(self.model.parameters()), list(self.model_hat.parameters())),
            )
        )

        self.init_FLAG = False

        pass
