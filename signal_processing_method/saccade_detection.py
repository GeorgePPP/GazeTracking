import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from icecream import ic

class SaccadeDetector:
    def __init__(self, config):
        self.config = config

    def nonlinear_denoising(self, y, alpha, beta, max_iterations=100):
        N = len(y)
        D1 = sparse.diags([-1, 1], [0, 1], shape=(N-1, N))
        D3 = sparse.diags([-1, 3, -3, 1], [0, 1, 2, 3], shape=(N-3, N))
        
        x = y.copy()
        for _ in range(max_iterations):
            A1 = sparse.diags(1 / (np.abs((D1 @ x)) + 1e-6))
            A3 = sparse.diags(1 / (np.abs((D3 @ x)) + 1e-6))
            M = sparse.eye(N) + alpha * D1.T @ A1 @ D1 + beta * D3.T @ A3 @ D3
            x_new = spsolve(M, y)
            if np.allclose(x, x_new, rtol=1e-6):
                break
            x = x_new
        return x

    def calculate_velocity(self, x, fps):
        # Convert pixels to degrees
        pixels_per_degree = self.config['pixels_per_degree']
        x_deg = x / pixels_per_degree
        
        # Calculate velocity in degrees per second using central differences
        velocity = np.diff(x_deg, prepend=x_deg[0], append=x_deg[-1]) * fps / 2 / 0.033
        
        return velocity

    def detect_saccades(self, position, fps):
        # Step 1: Nonlinear Denoising
        alpha = self.config['alpha']
        beta = self.config['beta']
        ic(np.min(position), np.max(position))
        denoised_position = self.nonlinear_denoising(position, alpha, beta)
        ic(np.min(denoised_position), np.max(denoised_position))

        # Step 2: Velocity Thresholding
        velocity = self.calculate_velocity(denoised_position, fps)
        ic(np.min(velocity), np.max(velocity), np.mean(np.abs(velocity)))

        onset_threshold = self.config['saccade_onset_velocity']
        offset_threshold = self.config['saccade_offset_velocity']

        saccade_onset = np.where(np.abs(velocity) > onset_threshold)[0]
        saccade_offset = np.where(np.abs(velocity) < offset_threshold)[0]

        saccades = []
        for start in saccade_onset:
            end = saccade_offset[saccade_offset > start]
            if len(end) > 0:
                saccades.append((start, end[0]))

        # Print number of saccades before post-processing
        print(f"Number of saccades detected before post-processing: {len(saccades)}")

        # Post-processing
        min_saccade_duration = int(self.config['min_saccade_duration'] * fps)
        min_intersaccadic_interval = int(self.config['min_intersaccadic_interval'] * fps)
        max_velocity = self.config['max_velocity']

        filtered_saccades = []
        removed_saccades = []
        for i, (start, end) in enumerate(saccades):
            duration = end - start
            peak_velocity = np.max(np.abs(velocity[start:end]))
            
            if duration >= min_saccade_duration and peak_velocity <= max_velocity:
                if i > 0:
                    prev_end = filtered_saccades[-1][1]
                    if start - prev_end >= min_intersaccadic_interval:
                        filtered_saccades.append((start, end))
                    else:
                        removed_saccades.append((start, end, "insufficient intersaccadic interval"))
                else:
                    filtered_saccades.append((start, end))
            else:
                if duration < min_saccade_duration:
                    removed_saccades.append((start, end, "insufficient duration"))
                elif peak_velocity > max_velocity:
                    removed_saccades.append((start, end, "excessive peak velocity"))

        # Print number of saccades after post-processing
        print(f"Number of saccades after post-processing: {len(filtered_saccades)}")

        # Print information about removed saccades
        print("Removed saccades:")
        for start, end, reason in removed_saccades:
            print(f"  Saccade from frame {start} to {end} removed due to {reason}")
            print(f"    Duration: {end - start} frames")
            print(f"    Peak velocity: {np.max(np.abs(velocity[start:end])):.2f}")

        labeled_saccades = np.zeros_like(position, dtype=int)
        for i, (start, end) in enumerate(filtered_saccades, 1):
            labeled_saccades[start:end] = i

        return labeled_saccades, denoised_position

def load_config(config_path):
    import json
    with open(config_path, 'r') as f:
        return json.load(f)