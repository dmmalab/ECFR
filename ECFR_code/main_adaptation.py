import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.build_dataset import StreamingMedicalDataset
from models.memory_bank import DynamicMemoryBank
from methods.ecfr import DualMinMaxAdaptation, configure_ln_parameters
from methods.grata_adapted import GradientAlignmentAdaptation
from utils.metrics import compute_accuracy, compute_brier_score, compute_ece
from utils.visualization import plot_ecfr_2d_dynamics

def parse_args():
    parser = argparse.ArgumentParser(description="Test-Time Adaptation for Medical FMs")
    parser.add_argument('--dataset_csv', type=str, required=True, help="Path to test dataset CSV")
    parser.add_argument('--method', type=str, choices=['ecfr', 'grata'], default='ecfr')
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for LayerNorm affine parameters")
    parser.add_argument('--capacity', type=int, default=128, help="Memory bank capacity")
    parser.add_argument('--quantile', type=float, default=0.5, help="Quantile for dynamic thresholding")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing {args.method.upper()} adaptation on {device}...")

    # 1. Initialize Dataset & Dataloader (Batch Size = 1 for continuous streaming)
    dataset = StreamingMedicalDataset(csv_path=args.dataset_csv)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 2. Initialize Model (Mocking the FM initialization for demonstration)
    # In practice, load Ark+ or DermLIP here.
    import torchvision.models as models
    model = models.resnet50(pretrained=True).to(device) 
    
    # 3. Configure Parameter-Efficient Fine-Tuning (LayerNorm only)
    opt_params = configure_ln_parameters(model)
    optimizer = optim.SGD(opt_params, lr=args.lr, momentum=0.9)

    # 4. Initialize Adaptation Method & Memory Bank
    if args.method == 'ecfr':
        method = DualMinMaxAdaptation(model, optimizer)
        memory_bank = DynamicMemoryBank(capacity=args.capacity, quantile=args.quantile)
    elif args.method == 'grata':
        method = GradientAlignmentAdaptation(model, optimizer)
    
    # 5. Streaming Metrics Tracking
    all_targets, all_probs = [], []
    all_entropies, all_consistencies = [], []
    
    # 6. Online Adaptation Loop
    model.train() # TTA requires train mode for LN updates
    
    for x_clean, x_weak, x_strong, target, _ in tqdm(dataloader, desc="Adapting"):
        x_clean, x_weak, x_strong = x_clean.to(device), x_weak.to(device), x_strong.to(device)
        
        if args.method == 'ecfr':
            logits, ent, cons = method.forward_and_adapt(x_clean, x_weak, x_strong, memory_bank)
            all_entropies.extend(ent.cpu().numpy())
            all_consistencies.extend(cons.cpu().numpy())
        elif args.method == 'grata':
            logits = method.forward_and_adapt(x_weak, x_strong)
            
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.detach().cpu())
        all_targets.extend(target.numpy())

    # 7. Evaluate Metrics
    final_probs = torch.cat(all_probs, dim=0)
    final_targets = torch.tensor(all_targets)
    
    acc = compute_accuracy(final_probs, final_targets)
    brier = compute_brier_score(final_probs, final_targets, num_classes=final_probs.shape[1])
    ece = compute_ece(final_probs, final_targets)
    
    print("\n" + "="*40)
    print(f"Results for {args.method.upper()}:")
    print(f"Accuracy: {acc:.4f} | Brier: {brier:.4f} | ECE: {ece:.4f}")
    print("="*40)

    # 8. Visualization (EAFR only)
    if args.method == 'ecfr' and len(all_entropies) > 0:
        ent_thresh, cons_thresh = memory_bank.get_thresholds()
        import numpy as np
        
        # One-Hot Brier Score
        targets_one_hot = torch.nn.functional.one_hot(final_targets, num_classes=final_probs.shape[1]).numpy()
        brier_solos = np.sum((final_probs.numpy() - targets_one_hot)**2, axis=1)
        
        plot_ecfr_2d_dynamics(
            entropies=np.array(all_entropies), 
            cons_losses=np.array(all_consistencies), 
            brier_solo_values=brier_solos, 
            entropy_thresh=ent_thresh, 
            cons_thresh=cons_thresh,
            save_path=f"ECFR_Quadrant_Flow.png"
        )
        print("Generated Flow Regulation Dynamics Plot.")    

if __name__ == "__main__":
    main()
