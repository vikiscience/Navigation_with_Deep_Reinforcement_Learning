import const

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

torch.random.manual_seed(const.random_seed)


class DQN(nn.Module):

    model_path = const.file_path_model

    def __init__(self, num_inputs: int = const.state_size,
                 num_outputs: int = const.action_size,
                 lr: float = const.model_learning_rate,
                 fc1_num: int = const.model_fc1_num,
                 fc2_num: int = const.model_fc2_num):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.learning_rate = lr
        self.fc1_num = fc1_num
        self.fc2_num = fc2_num
        self._build_model()

        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        self.fc1 = nn.Linear(self.num_inputs, self.fc1_num)
        self.fc2 = nn.Linear(self.fc1_num, self.fc2_num)
        self.fc3 = nn.Linear(self.fc2_num, self.num_outputs)

        self.model = nn.Sequential(self.fc1, nn.ELU(), self.fc2, nn.ELU(), self.fc3)
        print(self.model)

    def fit(self, X, y):
        x_tensor = torch.Tensor(X)
        y_tensor = torch.Tensor(y)

        # Clear the gradients
        self.optimizer.zero_grad()

        # Forward pass, backward pass, update weights
        output = self.model.forward(x_tensor)
        loss = self.criterion(output, y_tensor)
        loss.backward()
        self.optimizer.step()

    def predict(self, X):
        x_tensor = torch.Tensor(X)

        self.model.eval()
        with torch.no_grad():
            result = self.model(x_tensor).detach().numpy()
        self.model.train()

        return result

    def get_weights(self):
        return self.model.parameters()

    def set_weights(self, params):
        for target_param, source_param in zip(self.model.parameters(), params):
            target_param.data.copy_(source_param.data)

    def save_weights(self, fp=None):
        torch.save(self.model.state_dict(), fp)

    def load_weights(self, fp=None):
        self.model.load_state_dict(torch.load(fp))
        self.model.eval()  # change the model to evaluation mode (to use only for inference)
