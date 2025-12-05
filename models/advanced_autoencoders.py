import torch
import torch.nn as nn
# Assurez-vous que la classe MLP est définie ci-dessus

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], out_dim: int, 
                 activation=nn.ReLU, dropout_p: float = 0.0, use_bn: bool = False):
        # [MLP class definition unchanged from the user's input]
        super().__init__()

        layers = []
        last_dim = in_dim
        
        for i, h in enumerate(hidden):
            layers.append(nn.Linear(last_dim, h))
            
            if use_bn:
                layers.append(nn.BatchNorm1d(h))
            
            layers.append(activation())
            
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
                
            last_dim = h

        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --- Classes Encoder, Decoder, TaskHead restent identiques et flexibles ---
# Elles doivent accepter les paramètres de régularisation et d'activation

class Encoder(nn.Module):
    def __init__(self, n_feat: int, hidden: list[int], n_latent: int,
                 activation=nn.ReLU, dropout_p: float = 0.0, use_bn: bool = False):
        super().__init__()
        self.net = MLP(n_feat, hidden, n_latent, activation=activation, dropout_p=dropout_p, use_bn=use_bn)
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, n_latent: int, hidden: list[int], n_recon: int,
                 activation=nn.ReLU, dropout_p: float = 0.0, use_bn: bool = False):
        super().__init__()
        self.net = MLP(n_latent, hidden, n_recon, activation=activation, dropout_p=dropout_p, use_bn=use_bn)
    def forward(self, z): return self.net(z)

class TaskHead(nn.Module):
    def __init__(self, n_latent: int, hidden: list[int], n_head_out: int,
                 activation=nn.ReLU, dropout_p: float = 0.0, use_bn: bool = False):
        super().__init__()
        self.net = MLP(n_latent, hidden, n_head_out, activation=activation, dropout_p=dropout_p, use_bn=use_bn)
    def forward(self, z): return self.net(z)


# --- CLASSE AUTOENCODER MISE À JOUR ---

class AutoEncoder(nn.Module):
    def __init__(
        self,
        n_feat: int,
        n_latent: int,
        encoder_hidden: list[int],
        decoder_hidden: list[int],
        head_hidden: list[int],
        
        # --- PARAMÈTRES SPÉCIFIQUES À CHAQUE MODULE ---
        
        # Activations (peut être une classe unique, ou un dictionnaire/tuple de 3 classes)
        activations = (nn.ReLU, nn.ReLU, nn.ReLU), # (Encoder, Decoder, Head)
        
        # Dropouts
        dropout_ps: tuple[float, float, float] = (0.0, 0.0, 0.0), # (Encoder, Decoder, Head)
        
        # Batch Normalization
        use_bns: tuple[bool, bool, bool] = (False, False, False), # (Encoder, Decoder, Head)
    ):
        """
        Initializes the full Multi-Task AutoEncoder structure, allowing specific 
        hyperparameters for the Encoder, Decoder, and TaskHead.
        """
        super().__init__()

        act_enc, act_dec, act_head = activations
        dp_enc, dp_dec, dp_head = dropout_ps
        bn_enc, bn_dec, bn_head = use_bns
        
        # Encoder (Input -> Latent)
        self.encoder = Encoder(n_feat, encoder_hidden, n_latent,
                               activation=act_enc, dropout_p=dp_enc, use_bn=bn_enc)
        
        # Decoder (Latent -> Recon)
        self.decoder = Decoder(n_latent, decoder_hidden, n_feat,
                               activation=act_dec, dropout_p=dp_dec, use_bn=bn_dec)
        
        # TaskHead (Latent -> Output)
        self.head = TaskHead(n_latent, head_hidden, 1,
                             activation=act_head, dropout_p=dp_head, use_bn=bn_head)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        y_hat = self.head(z)
        return z, x_hat, y_hat
