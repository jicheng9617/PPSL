import torch
import math
import torch.nn.functional as F
import torch.nn as nn


class PSModelHyper(torch.nn.Module): 
    def __init__(self, 
                 n_params, 
                 n_dim, 
                 n_obj, 
                 params_hidden_size: int = 100, 
                 psm_hidden_size: int = 64, 
                 psm_n_layer : int = 2,
                 ) -> None:
        super().__init__()
        self.n_params = n_params
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.params_hidden_size = params_hidden_size
        self.psm_hidden_size = psm_hidden_size
        self.psm_n_layer = psm_n_layer

        self.params_mlp = nn.Sequential(
            nn.Linear(self.n_params, self.params_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.params_hidden_size, self.params_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.params_hidden_size, self.params_hidden_size),
        )

        self.hidden0_weights = nn.Linear(
            self.params_hidden_size, self.psm_hidden_size * self.n_obj
        )
        self.hidden0_bias = nn.Linear(self.params_hidden_size, self.psm_hidden_size)

        for i in range(1, self.psm_n_layer):
            setattr(
                self,
                f"hidden{i}_weights",
                nn.Linear(self.params_hidden_size, self.psm_hidden_size * self.psm_hidden_size),
            )
            setattr(self, f"hidden{i}_bias", nn.Linear(self.params_hidden_size, self.psm_hidden_size))

        setattr(
            self,
            f"hidden{self.psm_n_layer}_weights",
            nn.Linear(self.params_hidden_size, self.psm_hidden_size * self.n_dim),
        )
        setattr(self, f"hidden{self.psm_n_layer}_bias", nn.Linear(self.params_hidden_size, self.n_dim))


    def forward(self, params): 

        features = self.params_mlp(params)

        out_dict = {}

        for j in range(self.psm_n_layer+1):
            out_dict[f"hidden{j}.weights"] = getattr(self, f"hidden{j}_weights")(
                features
            )
            out_dict[f"hidden{j}.bias"] = getattr(self, f"hidden{j}_bias")(
                features
            ).flatten()

        return out_dict


class PSModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj, hidden_size: int = 64, n_layer: int = 2):
        super().__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.n_layer = n_layer 
        self.hidden_size = hidden_size

    def forward(self, pref, weights=None):

        x = F.linear(
            pref,
            weight=weights["hidden0.weights"].reshape(
                self.hidden_size ,self.n_obj
            ),
            bias=weights["hidden0.bias"],
        )
        x = F.relu(x)

        for i in range(1, self.n_layer):
            x = F.linear(
                x,
                weight=weights[f"hidden{i}.weights"].reshape(
                    self.hidden_size,
                    self.hidden_size,
                ),
                bias=weights[f"hidden{i}.bias"],
            )
            x = F.relu(x)

        x = F.linear(
            x,
            weight=weights[f"hidden{self.n_layer}.weights"].reshape(
                self.n_dim, self.hidden_size
            ),
            bias=weights[f"hidden{self.n_layer}.bias"],
        )
        x = F.sigmoid(x)

        return x.to(torch.float64)
    
    
class PSModelLoRAHyper(torch.nn.Module): 
    def __init__(self, 
                 n_params, 
                 n_dim, 
                 n_obj, 
                 free_rank: int = 3, 
                 params_hidden_size: int = 1024, 
                 psm_hidden_size: int = 256, 
                 psm_n_layer : int = 2,
                 ) -> None:
        super().__init__()
        self.n_params = n_params
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.free_rank = free_rank
        self.params_hidden_size = params_hidden_size
        self.psm_hidden_size = psm_hidden_size
        self.psm_n_layer = psm_n_layer

        self.params_mlp = nn.Sequential(
            nn.Linear(self.n_params, self.params_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.params_hidden_size, self.params_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.params_hidden_size, self.params_hidden_size),
        )

        self.A0 = nn.Linear(
            self.params_hidden_size, self.psm_hidden_size * self.free_rank
        )
        self.B0 = nn.Linear(self.params_hidden_size, self.free_rank * self.n_obj)

        for i in range(1, self.psm_n_layer):
            setattr(self, f"A{i}", nn.Linear(self.params_hidden_size, self.psm_hidden_size * self.free_rank))
            setattr(self, f"B{i}", nn.Linear(self.params_hidden_size, self.free_rank * self.psm_hidden_size))

        setattr(self, f"A{self.psm_n_layer}", nn.Linear(self.params_hidden_size, self.n_dim * self.free_rank))
        setattr(self, f"B{self.psm_n_layer}", nn.Linear(self.params_hidden_size, self.free_rank * self.psm_hidden_size))
        
        # Initialize weights
        self.init_weights()

    def init_weights(self, init_type='xavier_uniform', seed=42):
        """
        Initialize weights with a specific method and seed for reproducibility.
        
        Args:
            init_type: Type of initialization ('xavier_uniform', 'xavier_normal', 
                      'kaiming_uniform', 'kaiming_normal', 'normal', 'uniform')
            seed: Random seed for reproducibility
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Initialize params_mlp
        for module in self.params_mlp.modules():
            if isinstance(module, nn.Linear):
                if init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif init_type == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif init_type == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                elif init_type == 'normal':
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                elif init_type == 'uniform':
                    nn.init.uniform_(module.weight, a=-0.1, b=0.1)
                else:
                    raise ValueError(f"Unknown init_type: {init_type}")
                
                # Initialize bias to zero
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        
        # Initialize A and B matrices
        for i in range(self.psm_n_layer + 1):
            for prefix in ['A', 'B']:
                layer = getattr(self, f"{prefix}{i}")
                if init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(layer.weight)
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(layer.weight)
                elif init_type == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(layer.weight)
                elif init_type == 'kaiming_normal':
                    nn.init.kaiming_normal_(layer.weight)
                elif init_type == 'normal':
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                elif init_type == 'uniform':
                    nn.init.uniform_(layer.weight, a=-0.1, b=0.1)
                
                # Initialize bias to zero
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
    
    def reset_parameters(self, seed=42):
        """
        Reset all parameters to initial values with the same seed.
        This ensures identical initialization across runs.
        """
        self.init_weights(seed=seed)

    def forward(self, params): 
        features = self.params_mlp(params)

        out_dict = {}
        for j in range(self.psm_n_layer+1):
            out_dict[f"A{j}"] = getattr(self, f"A{j}")(features)
            out_dict[f"B{j}"] = getattr(self, f"B{j}")(features)
        
        return out_dict


class PSModelLoRA(torch.nn.Module):
    def __init__(self, 
                 n_dim, 
                 n_obj, 
                 free_rank: int = 3,
                 hidden_size: int = 64, 
                 n_layer: int = 2):
        super().__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.n_layer = n_layer 
        self.free_rank = free_rank
        self.hidden_size = hidden_size
        
        self.base_model = PSbaseModel(n_dim=self.n_dim, 
                                      n_obj=self.n_obj, 
                                      hidden_size=self.hidden_size, 
                                      n_layer=self.n_layer)

    def forward(self, pref, weights=None):
        
        params = list(self.base_model.parameters())

        x = F.linear(
            pref,
            weight=params[0] + \
                weights["A0"].reshape(self.hidden_size, self.free_rank) @ weights["B0"].reshape(self.free_rank, self.n_obj),
            bias=params[1],
        )
        x = F.relu(x)

        for i in range(1, self.n_layer):
            x = F.linear(
                x,
                weight=params[i*2] + \
                    weights[f"A{i}"].reshape(self.hidden_size, self.free_rank) @ weights[f"B{i}"].reshape(self.free_rank, self.hidden_size),
                bias=params[i*2+1],
            )
            x = F.relu(x)

        x = F.linear(
            x,
            weight=params[-2]+ \
                weights[f"A{self.n_layer}"].reshape(self.n_dim, self.free_rank) @ weights[f"B{self.n_layer}"].reshape(self.free_rank, self.hidden_size),
            bias=params[-1],
        )
        x = F.sigmoid(x)

        return x.to(torch.float64)
    

class PSbaseModel(torch.nn.Module):
    def __init__(self, n_dim, n_obj, hidden_size: int = 256, n_layer: int = 2):
        super().__init__()
        self.n_dim = n_dim
        self.n_obj = n_obj
        self.hidden_size = hidden_size
        self.n_layer = n_layer
       
        self.first_layer = nn.Linear(self.n_obj, self.hidden_size)
        for i in range(0, self.n_layer-1): 
            setattr(self, f'hidden_layer{i}', nn.Linear(self.hidden_size, self.hidden_size))
        self.last_layer = nn.Linear(self.hidden_size, self.n_dim)
        
        # Initialize weights
        self.init_weights()
      
    def init_weights(self, init_type='xavier_uniform', seed=42):
        """
        Initialize weights with a specific method and seed for reproducibility.
        
        Args:
            init_type: Type of initialization ('xavier_uniform', 'xavier_normal', 
                      'kaiming_uniform', 'kaiming_normal', 'normal', 'uniform', 'he')
            seed: Random seed for reproducibility
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # List of all layers to initialize
        layers_to_init = [self.first_layer]
        for i in range(self.n_layer-1):
            layers_to_init.append(getattr(self, f'hidden_layer{i}'))
        layers_to_init.append(self.last_layer)
        
        # Initialize each layer
        for layer in layers_to_init:
            if isinstance(layer, nn.Linear):
                # Weight initialization
                if init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(layer.weight)
                elif init_type == 'xavier_normal':
                    nn.init.xavier_normal_(layer.weight)
                elif init_type == 'kaiming_uniform' or init_type == 'he':
                    # Kaiming/He initialization is good for ReLU
                    nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                elif init_type == 'kaiming_normal':
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                elif init_type == 'normal':
                    nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                elif init_type == 'uniform':
                    # Uniform initialization with bounds based on layer size
                    fan_in = layer.weight.size(1)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(layer.weight, -bound, bound)
                else:
                    raise ValueError(f"Unknown init_type: {init_type}")
                
                # Bias initialization
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
        
        # Special initialization for last layer (sigmoid output)
        # Smaller weights to start near 0.5 after sigmoid
        with torch.no_grad():
            self.last_layer.weight.data *= 0.1
    
    def init_weights_custom(self, seed=42):
        """
        Custom initialization specifically designed for this architecture.
        Uses Kaiming for hidden layers (good for ReLU) and special init for output.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # First layer: Xavier initialization (no activation before it)
        nn.init.xavier_uniform_(self.first_layer.weight)
        nn.init.constant_(self.first_layer.bias, 0.0)
        
        # Hidden layers: Kaiming initialization (ReLU activation)
        for i in range(self.n_layer-1):
            layer = getattr(self, f'hidden_layer{i}')
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.0)
        
        # Last layer: Small weights for sigmoid output to start near 0.5
        nn.init.uniform_(self.last_layer.weight, -0.1, 0.1)
        nn.init.constant_(self.last_layer.bias, 0.0)
    
    def reset_parameters(self, init_type='kaiming_uniform', seed=42):
        """
        Reset all parameters to initial values with the same seed.
        This ensures identical initialization across runs.
        """
        self.init_weights(init_type=init_type, seed=seed)
    
    def forward(self, pref):
        x = torch.relu(self.first_layer(pref))
        for i in range(0, self.n_layer-1):
            x = torch.relu(
                getattr(self, f'hidden_layer{i}')(x)
            )
        x = torch.sigmoid(self.last_layer(x)) 
        
        return x.to(torch.float64)