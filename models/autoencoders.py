import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, 
                 activation=nn.ReLU, dropout_p: float = 0.0, use_bn: bool = False):
        """
        Initializes a Multi-Layer Perceptron (MLP) with configurable activation, 
        dropout, and batch normalization.

        Args:
            in_dim (int): Dimension of the input features.
            hidden (list[int]): List of dimensions for the hidden layers.
            out_dim (int): Dimension of the output layer.
            activation (nn.Module): Activation function to use (default: nn.ReLU).
            dropout_p (float): Dropout probability (0.0 to disable).
            use_bn (bool): Whether to include BatchNorm1d layers.
        """
        super().__init__()

        layers = []
        last_dim = in_dim
        num_hidden = len(hidden)

        # ----------------------------------------------------
        # Hidden Layers Loop
        # ----------------------------------------------------
        for i, h in enumerate(hidden):
            
            # 1. Linear Layer
            layers.append(nn.Linear(last_dim, h))
            
            # 2. Batch Normalization (BN should go BEFORE activation)
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            
            # 3. Activation
            layers.append(activation())
            
            # 4. Dropout (Dropout should go AFTER activation)
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
                
            last_dim = h

        # ----------------------------------------------------
        # Output Layer (no BN, no Dropout, no Activation by default for regression)
        # ----------------------------------------------------
        layers.append(nn.Linear(last_dim, out_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, n_feat: int, hidden: list[int], n_latent: int,
                 activation=nn.ReLU, dropout_p: float = 0.0, use_bn: bool = False):
        """
        Initializes the Encoder. Maps input features (n_feat) to the latent space (n_latent).
        """
        super().__init__()
        # The output dimension of the Encoder is the latent dimension (n_latent)
        self.net = MLP(n_feat, hidden, n_latent, 
                       activation=activation, dropout_p=dropout_p, use_bn=use_bn)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, n_latent: int, hidden: list[int], n_recon: int,
                 activation=nn.ReLU, dropout_p: float = 0.0, use_bn: bool = False):
        """
        Initializes the Decoder. Maps latent space (n_latent) back to input feature space (n_recon).
        """
        super().__init__()
        # The input dimension is n_latent, the output dimension is n_recon (which should be n_feat)
        self.net = MLP(n_latent, hidden, n_recon, 
                       activation=activation, dropout_p=dropout_p, use_bn=use_bn)

    def forward(self, z):
        return self.net(z)


class TaskHead(nn.Module):
    def __init__(self, n_latent: int, hidden: list[int], n_head_out: int,
                 activation=nn.ReLU, dropout_p: float = 0.0, use_bn: bool = False):
        """
        Initializes the TaskHead. Maps latent space (n_latent) to the supervised output (n_head_out).
        """
        super().__init__()
        # The input dimension is n_latent, the output dimension is n_head_out (e.g., 1)
        self.net = MLP(n_latent, hidden, n_head_out, 
                       activation=activation, dropout_p=dropout_p, use_bn=use_bn)

    def forward(self, z):
        return self.net(z)


class AutoEncoder(nn.Module):
    def __init__(
        self,
        n_feat: int,
        n_latent: int,
        encoder_hidden: list[int],
        decoder_hidden: list[int],
        head_hidden: list[int],
        
        # --- NEW REGULARIZATION PARAMETERS FOR ALL SUB-MODULES ---
        activation=nn.ReLU,
        dropout_p: float = 0.0,
        use_bn: bool = False,
    ):
        """
        Initializes the full Multi-Task AutoEncoder structure.
        """
        super().__init__()

        # Pass regularization parameters to the Encoder
        self.encoder = Encoder(n_feat, encoder_hidden, n_latent,
                               activation=activation, dropout_p=dropout_p, use_bn=use_bn)
        
        # Pass regularization parameters to the Decoder
        self.decoder = Decoder(n_latent, decoder_hidden, n_feat, # n_recon must be n_feat
                               activation=activation, dropout_p=dropout_p, use_bn=use_bn)
        
        # Pass regularization parameters to the TaskHead
        self.head = TaskHead(n_latent, head_hidden, 1, # Output dimension fixed to 1 for typical regression
                             activation=activation, dropout_p=dropout_p, use_bn=use_bn)

    def forward(self, x):
        """
        Forward pass: X -> Z -> (X_hat, Y_hat)
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.head(z)
        return z, x_hat, y_hat