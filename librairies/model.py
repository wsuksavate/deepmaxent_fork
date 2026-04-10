import torch
import torch.nn as nn

class deepmaxent_model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_nbr):
        super(deepmaxent_model, self).__init__()
        
        self.fc1_lambda = nn.Linear(input_size, hidden_size)
        self.hidden_layers_lambda = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_nbr)])
        self.fc3_lambda = nn.Linear(hidden_size, output_size)
        
    def forward(self, xinput):
        x = self.fc1_lambda(xinput).relu()
        for layer in self.hidden_layers_lambda:
            x = layer(x).relu()+x
        x = self.fc3_lambda(x)
        
        return x

### My modification: Add species embedding
### num_species = output_size
class deepmaxent_embedding_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_species, hidden_nbr, embedding_dim=3):
        super(deepmaxent_embedding_model, self).__init__()
        self.num_species = num_species
        # Define the Embedding Matrix for all species
        self.species_embedding = nn.Embedding(num_embeddings=num_species, embedding_dim=embedding_dim)
        # The input layer now accepts the original input PLUS the 3 embedding values for each species
        self.fc1_lambda = nn.Linear(input_size + embedding_dim, hidden_size)
        # Hidden layers remain exactly the same (Shared across all species!)
        self.hidden_layers_lambda = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(hidden_nbr)
        ])
        # The output layer now outputs 1 value (the score for a single site-species pair)
        self.fc3_lambda = nn.Linear(hidden_size, 1)
        
    def forward(self, xinput):
        # xinput shape is typically: [batch_size, input_size]
        batch_size = xinput.size(0) # number of sites
        # --- DATA PREPARATION ---
        # We want to predict for ALL species at each site in the batch simultaneously.
        # Step 1: Duplicate the site covariates for every species
        # Transforms shape from [batch_size, input_size] -> [batch_size, num_species, input_size]
        x_expanded = xinput.unsqueeze(1).expand(batch_size, self.num_species, -1)
        # Step 2: Get the 3-element embeddings for ALL species
        species_ids = torch.arange(self.num_species, device=xinput.device)
        emb = self.species_embedding(species_ids) # Shape: [num_species, 3]
        # Duplicate the embeddings for every site in the batch
        # Transforms shape from [num_species, 3] -> [batch_size, num_species, 3]
        emb_expanded = emb.unsqueeze(0).expand(batch_size, self.num_species, -1)
        # Step 3: Merge (concatenate) the site covariates and the species embeddings
        # Shape becomes: [batch_size, num_species, input_size + 3]
        merged_input = torch.cat([x_expanded, emb_expanded], dim=2)
        # --- NEURAL NETWORK PASS ---
        # PyTorch automatically applies Linear layers across all species independently
        x = self.fc1_lambda(merged_input).relu()
        for layer in self.hidden_layers_lambda:
            x = layer(x).relu() + x   
        out = self.fc3_lambda(x) # Shape: [batch_size, num_species, 1]
        # Remove the final dimension to match your original output format: [batch_size, num_species]
        return out.squeeze(-1)

    
def save_mlp_model(args, model):
    """
    Save the MLP model to a PyTorch model file.

    Args:
        args (argparse.Namespace): Arguments passed to the function.
        model (models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel): MLP model.
    """
    mlp_filepath = (
        args.outputdir + "model/MLP.pth"
    )  # Define the file path to save the MLP model file
    torch.save(model, mlp_filepath)  # Save the MLP model to a PyTorch model file


def load_mlp_model(args, model):
    """
    Load the pre-trained MLP model.

    Args:
        args (argparse.Namespace): An object containing the necessary arguments.
        model (models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel): The MLP model object.

    Returns:
        models.mlp.MLP | torch.nn.parallel.data_parallel.DataParallel: The loaded pre-trained MLP model.
    """
    mlp_filepath = (
        args.outputdir + "model/MLP.pth"
    )  # Filepath of the pre-trained MLP model
    model.load_state_dict(
        torch.load(mlp_filepath)
    )  # Load the weights of the pre-trained model
    return model


def make_predictions(model, X_tensor):
    """
    Make predictions using the given PyTorch model and input tensor.

    Parameters:
        model (torch.nn.Module): The PyTorch model.
        X_tensor (torch.Tensor): The input tensor for making predictions.

    Returns:
        torch.Tensor: The predictions.
    """
    model.eval()
    model = model.to("cpu")
    X_tensor = X_tensor.to("cpu")

    with torch.no_grad():
        predictions = model(X_tensor)

    return predictions