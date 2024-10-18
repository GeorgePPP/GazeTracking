import matplotlib.pyplot as plt
import numpy as np

def generate_visualizations(t, position, denoised_position, labeled_saccades, saccade_threshold, config):
    plt.figure(figsize=(12, 8))

    # Plot original and denoised position
    plt.subplot(2, 1, 1)
    plt.plot(t, position, label='Original')
    plt.plot(t, denoised_position, label='Denoised')
    plt.title('Eye Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend()

    # Plot velocity and saccades
    plt.subplot(2, 1, 2)
    velocity = np.diff(denoised_position) / np.diff(t)
    velocity = np.insert(velocity, 0, velocity[0])
    
    plt.plot(t, velocity, label='Velocity')
    plt.axhline(y=saccade_threshold, color='r', linestyle='--', label='Saccade Threshold')
    plt.axhline(y=-saccade_threshold, color='r', linestyle='--')
    
    saccade_starts = np.where(np.diff(labeled_saccades) == 1)[0]
    saccade_ends = np.where(np.diff(labeled_saccades) == -1)[0]
    
    for start, end in zip(saccade_starts, saccade_ends):
        plt.axvspan(t[start], t[end], alpha=0.2, color='red')

    plt.title('Eye Velocity and Detected Saccades')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (degrees/s)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('saccade_detection_visualization.png')
    plt.close()