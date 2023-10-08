import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def draw_color_strip(value, filename):

    value = max(-1, min(1, value))  # Ensure value is within the range [-1, 1]
    gradient = np.linspace(-1, 1, 256).reshape(1, -1)
    gradient = np.vstack((gradient, gradient))
    fig, ax = plt.subplots(figsize=(8, 2))

    ax.imshow(gradient, aspect='auto', cmap='viridis', extent=[-1, 1, 0, 1])
    ax.axvline(value, color='red', linestyle='--')
    ax.text(-1, -0.1, '-1 (Negative)', color='black', ha='center', size=10)
    ax.text(1, -0.1, '1 (Positive)', color='black', ha='center', size=10)
    ax.text(0, -0.1, '(Neutral)', color='black', ha='center', size=10)
    ax.text(value, 1.1, f'{value:.2f}', color='red', ha='center', size=10)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    plt.title(filename)
    plt.savefig(f'static/images/{filename}.png', dpi=300, bbox_inches='tight')
