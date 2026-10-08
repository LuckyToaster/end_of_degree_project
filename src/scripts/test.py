import argparse
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader

from src.constants import CSV_PATH
from src.dataset import FoodDataset
from src.ml import three_way_split, standardize, dataloader_args
from src.models import get_Swin_V2_S, get_EfficientNet_B3, get_EfficientNet_V2_S, get_MobileNet_V3_L

MODEL_REGISTRY = {
    'swin_v2_s': get_Swin_V2_S,
    'efficientnet_b3': get_EfficientNet_B3,
    'efficientnet_v2_s': get_EfficientNet_V2_S,
    'mobilenet_v3_l': get_MobileNet_V3_L
}

def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained model checkpoint on the holdout test set.")
    parser.add_argument("model_name", choices=list(MODEL_REGISTRY.keys()), help="The model architecture to load.")
    parser.add_argument("checkpoint", help="Path to the trained model checkpoint (.pt file).")
    parser.add_argument("--targets", nargs='+', default=['fat_g', 'carb_g', 'prot_g', 'kcal'], help="Target columns the model was trained on.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed used for data splitting (to ensure identical test set).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference.")
    return parser.parse_args()

def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        sys.exit(1)

    print("Preparing dataset (extracting holdout test set)...")
    _, _, test_df = three_way_split(CSV_PATH, args.seed, stratify_target='kcal')
    _, _, test_df, means, stds = standardize(None, None, test_df, args.targets) # Only need to standardize test set here

    print(f"Initializing model: {args.model_name}...")
    model_getter = MODEL_REGISTRY[args.model_name]
    
    getter_result = model_getter(n_targets=len(args.targets), verbose=False)
    # Handle varying return lengths based on models.py state
    val_transforms = getter_result[-1] 
    model = getter_result[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading weights from {checkpoint_path}...")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
         print(f"Error loading checkpoint: {e}")
         print("Make sure the architecture and n_targets match the checkpoint exactly.")
         sys.exit(1)
         
    model = model.to(device)
    model.eval()

    test_loader = DataLoader(FoodDataset(test_df, val_transforms, 'img_path', args.targets), batch_size=args.batch_size, shuffle=False, **dataloader_args)

    criterion = torch.nn.L1Loss(reduction='sum') # Using L1 (MAE) for intuitive metric reporting
    total_loss = 0.0
    total_samples = 0

    print("Starting evaluation on test set...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_samples += targets.size(0)

    # Note: This is reporting MAE on the STANDARDIZED targets. 
    # To report real-world MAE (e.g., actual grams off), you would need to un-standardize the predictions here using the `stds` array.
    avg_mae = total_loss / total_samples
    
    print("-" * 30)
    print("Test Set Evaluation Results")
    print("-" * 30)
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {checkpoint_path.name}")
    print(f"Samples Evaluated: {total_samples}")
    print(f"Standardized Mean Absolute Error (MAE): {avg_mae:.4f}")
    
    # Optional: Quick logic to un-standardize and show real MAE if stds are available
    if stds is not None:
         real_mae_per_target = avg_mae * stds 
         print("\nEstimated Real-World Average Error:")
         for target_name, error, std in zip(args.targets, real_mae_per_target, stds):
             print(f"  {target_name}: ~{error:.2f} (std scale: {std:.2f})")

if __name__ == "__main__":
    main()
