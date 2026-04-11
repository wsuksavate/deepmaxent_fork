import torch
import random
import numpy as np
import os
import datetime
import warnings

    
def disable_warnings():
    warnings.filterwarnings("ignore")

def set_seed(global_seed):
    """
    Sets the random seeds for reproducibility.

    Args:
        args (argparse.Namespace): The arguments containing the seed values.

    Note:

    """
    random.seed(global_seed)  # Set the random seed for the random module
    np.random.seed(global_seed)  # Set the random seed for the numpy module
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(global_seed)

    
def get_random_seed_list(nbr):
    return [random.randint(1, 1000) for _ in range(nbr)]

def print_parameters(args):
    """
    Print a configuration file 

    Args:
        args (argparse.Namespace): The arguments containing all input values.

    Note:
        """
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_filename = f"{args.outputdir}/configuration_parameters.txt"
    with open(output_filename, "w") as output_file:
        output_file.write(f"# Configuration Parameters - Date: {current_date}\n")
        output_file.write(f"min_lon={args.min_lon}\n")
        output_file.write(f"min_lat={args.min_lat}\n")
        output_file.write(f"max_lon={args.max_lon}\n")
        output_file.write(f"max_lat={args.max_lat}\n")
        output_file.write(f"diag_variance={args.diag_variance}\n")
        output_file.write(f"seuil_correlation={args.seuil_correlation}\n")
        output_file.write(f"nspecies={args.nspecies}\n")
        output_file.write(f"nobservation_by_species={args.nobservation_by_species}\n")
        output_file.write(f"nbr_observation_required={args.nbr_observation_required}\n")
        output_file.write(f"n_components={args.n_components}\n")
        output_file.write(f"savefigures={args.savefigures}\n")
        output_file.write(f"closefigures={args.closefigures}\n")
        output_file.write(f"noms_colonnes_to_drop={args.noms_colonnes_to_drop}\n")
    
    print(f"Configuration parameters have been saved to {output_filename}")



    
def make_results_directory(args):   
    
    if not os.path.exists(f'{args.outputdir}'):
        os.mkdir(f'{args.outputdir}')
    if not os.path.exists(f'{args.outputdir}/models'):    
        os.mkdir(f'{args.outputdir}/models')
    if not os.path.exists(f'{args.outputdir}/AUC_by_species'):    
        os.mkdir(f'{args.outputdir}/AUC_by_species')
        

def compute_auc(model, X, y, device):
    """
    Compute mean AUC across all species with sufficient data.
    
    Args:
        model: trained DeepMaxent model
        X: input features tensor
        y: target occurrence tensor
        device: computation device
    
    Returns:
        mean_auc: average AUC across species
        valid_aucs: list of AUC values for each valid species
    """
    model.eval()
    with torch.no_grad():
        X_dev = X.to(device)
        predictions = model(X_dev).cpu()
        # Apply softmax to get probabilities
        probs = torch.softmax(predictions, dim=0).numpy()
    
    y_np = y.numpy()
    
    # Convert to binary (presence/absence)
    y_binary = (y_np > 0).astype(int)
    
    valid_aucs = []
    for sp_idx in range(y_binary.shape[1]):
        # Only compute AUC if species has both presences and absences
        if y_binary[:, sp_idx].sum() > 0 and y_binary[:, sp_idx].sum() < len(y_binary):
            try:
                auc = roc_auc_score(y_binary[:, sp_idx], probs[:, sp_idx])
                valid_aucs.append(auc)
            except:
                pass
    
    mean_auc = np.mean(valid_aucs) if valid_aucs else 0.0
    return mean_auc, valid_aucs
        