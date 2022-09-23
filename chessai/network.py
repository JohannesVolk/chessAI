import itertools
import torch
import torch.nn as nn
import pytorch_lightning as pl


class ChessBoardEvalNN(pl.LightningModule):
    def __init__(self, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.example_input_array = torch.zeros(1, 12, 8, 8)
        # input is a 1 hot encoded 1d tensor (64 bitmap for each piece type + a bit for current player)

        class ResidualBlock(nn.Sequential):
            def __init__(self, neuron_external, neurons_internal=None):
                if neurons_internal is None:
                    neurons_internal = neuron_external

                super().__init__(
                    nn.ReLU(),
                    nn.BatchNorm1d(neuron_external),
                    nn.Linear(neuron_external, neurons_internal),
                    nn.ReLU(),
                    nn.BatchNorm1d(neurons_internal),
                    nn.Linear(neurons_internal, neuron_external),
                )

        self.intake = nn.Sequential(nn.Flatten(),)

        self.residual_1 = ResidualBlock(768)
        self.residual_2 = ResidualBlock(768)
        self.residual_3 = ResidualBlock(768)
        self.residual_4 = ResidualBlock(768)

        self.exhaust_score = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.blocks = [
            self.intake,
            self.residual_1,
            self.residual_2,
            self.residual_3,
            self.residual_4,
            
            self.exhaust_score,
        ]

    def forward(self, x):
        """
        Forward pass of the neural network.
        """
        x = self.intake(x)

        x = self.residual_1(x) + x
        x = self.residual_2(x) + x
        x = self.residual_3(x) + x
        x = self.residual_4(x) + x
        x = self.exhaust_score(x)
        x = torch.squeeze(x)

        return x

    def configure_optimizers(self):

        params = list(map(nn.Sequential.parameters, self.blocks))

        optim = torch.optim.Adam(
            itertools.chain(*params), lr=self.hparams["learning_rate"],
        )

        return optim

    def training_step(self, batch, _):
        positions, _, target_eval_score = batch

        # Perform a forward pass on the network with inputs
        score_prediction = self.forward(positions)

        # calculate the loss with the network predictions and ground truth targets
        loss_function_score = torch.nn.L1Loss()
        loss_score = loss_function_score(score_prediction, target_eval_score)
        self.log("loss_score", loss_score)

        return {"loss": loss_score}

    def validation_step(self, batch, _):
        positions, _, target_eval_score = batch

        # Perform a forward pass on the network with inputs
        score_prediction = self.forward(positions)

        # calculate the loss with the network predictions and ground truth targets
        loss_function_score = torch.nn.L1Loss()
        loss_score = loss_function_score(score_prediction, target_eval_score)
        self.log("val_loss_score", loss_score)

        return {"val_loss": loss_score}
    
    
    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(hparams):
        model = ChessBoardEvalNN(hparams=hparams)
        model.load_state_dict(torch.load("./models/model.model"))

        return model
