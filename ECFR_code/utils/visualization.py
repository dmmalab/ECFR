import matplotlib.pyplot as plt
import numpy as np

def plot_ecfr_2d_dynamics(entropies, cons_losses, brier_solo_values, 
                          entropy_thresh, cons_thresh, 
                          x_max_fixed=0.693, y_max_fixed=0.693, 
                          save_path="ECFR_2D_Dynamics.png"):#Adjust according to the zero-shot results
    """
    Plots the 2D Quadrant Flow visualization using Piecewise Hybrid Scale.
    Accepts pre-calculated brier_solo_values to strictly support multi-class plotting.
    """
    try:
        plt.ioff()
        
        x_min, x_max = 0.0, x_max_fixed
        y_min, y_max = 0.0, y_max_fixed
        
        valid_mask = (entropies >= x_min) & (entropies <= x_max) & \
                     (cons_losses >= y_min) & (cons_losses <= y_max)
        
        x_filtered = entropies[valid_mask]
        y_filtered = cons_losses[valid_mask]
        brier_filtered = brier_solo_values[valid_mask] 
        
        def piecewise_hybrid_y(vals, v_thr, v_max):
            res = np.zeros_like(vals, dtype=float)
            mask_low = vals <= v_thr
            res[mask_low] = (vals[mask_low] / v_thr) * (1/3)
            v_thr_log = np.log10(v_thr + 1e-9)
            v_max_log = np.log10(v_max + 1e-9)
            res[~mask_low] = (1/3) + ((np.log10(vals[~mask_low] + 1e-9) - v_thr_log) / (v_max_log - v_thr_log)) * (2/3)
            return res

        y_mapped = piecewise_hybrid_y(y_filtered, cons_thresh, y_max)
        
        fig, ax = plt.subplots(figsize=(11, 9))
        background_colors = [(0.8, 1.0, 0.8), (0.8, 0.8, 1.0), (0.9, 0.9, 0.9), (0.95, 0.8, 0.95)]
        
        ax.add_patch(plt.Rectangle((x_min, 0), entropy_thresh - x_min, 1/3, facecolor=background_colors[0], alpha=0.3))
        ax.add_patch(plt.Rectangle((x_min, 1/3), entropy_thresh - x_min, 1 - 1/3, facecolor=background_colors[1], alpha=0.3))
        ax.add_patch(plt.Rectangle((entropy_thresh, 0), x_max - entropy_thresh, 1/3, facecolor=background_colors[2], alpha=0.3))
        ax.add_patch(plt.Rectangle((entropy_thresh, 1/3), x_max - entropy_thresh, 1 - 1/3, facecolor=background_colors[3], alpha=0.3))
        
        ax.axvline(x=entropy_thresh, color='gray', linestyle='--', alpha=0.7)
        ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.7)
        
        scatter = ax.scatter(x_filtered, y_mapped, c=brier_filtered, cmap='YlOrRd', s=30, alpha=0.8, edgecolors='black', linewidths=0.3)
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Squared Prediction Error', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Entropy (Linear Scale)', fontsize=16, fontweight='bold', labelpad=10)
        ax.set_ylabel('Consistency JS Divergence (Hybrid Scale)', fontsize=16, fontweight='bold', labelpad=10)
        
        ax.set_xlim([x_min, x_max])
        ax.set_xticks([0.0, entropy_thresh, x_max])
        ax.set_xticklabels(['0.0', f'{entropy_thresh:.4f}', f'{x_max:.3f}'], fontsize=14, fontweight='bold')
        
        ax.set_ylim([0, 1])
        ax.set_yticks([0.0, 1/3, 1.0])
        ax.set_yticklabels(['0.0', f'{cons_thresh:.5f}', f'{y_max:.6f}'], fontsize=14, fontweight='bold')
        
        plt.title(f'ECFR Flow Regulation Dynamics', fontsize=16, fontweight='bold')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Visualization Error: {e}")
