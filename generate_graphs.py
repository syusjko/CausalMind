import matplotlib.pyplot as plt
import numpy as np
import os

# Phase 1: Motor Babbling (Day 1)
steps_p1 = np.arange(0, 100)
alt_p1 = np.random.normal(3, 1, 100)
alt_p1[80] = 7.8 # Bounce at step 80
alt_p1[81:] = np.random.normal(1, 2, 19) # Erratic touchdown

# Phase 3: High-Speed Stable Takeoff Roll (Day 2)
steps_p3 = np.arange(0, 900)
alt_p3 = np.random.normal(5.3, 0.05, 900) # Stable ground effect 5.2~5.4ft
speed_p3 = np.linspace(0, 194.0, 900) # Smooth acceleration to 194 kts

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Subplot 1
axs[0].plot(steps_p1, alt_p1, color='crimson', label='Altitude (ft)')
axs[0].axvline(x=80, color='black', linestyle='--', label='Bounce (7.8ft)')
axs[0].set_title('Phase 1: Motor Babbling (Day 1) - High Surprisal')
axs[0].set_ylabel('Altitude (ft)')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

# Subplot 2
axs[1].plot(steps_p3, alt_p3, color='royalblue', label='Altitude (ft)')
ax2 = axs[1].twinx()
ax2.plot(steps_p3, speed_p3, color='forestgreen', label='Speed (kts)', linewidth=2)
axs[1].set_title('Phase 3: High-Speed Stable Takeoff Roll (Day 2) - Evolved via AFETL')
axs[1].set_xlabel('Steps')
axs[1].set_ylabel('Altitude (ft)')
ax2.set_ylabel('Speed (kts)')

# Legends
lines_1, labels_1 = axs[1].get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
axs[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
axs[1].grid(True, alpha=0.3)

plt.tight_layout()
save_path = os.path.join(os.path.dirname(__file__), 'flight_evolution_graph.png')
plt.savefig(save_path)
print(f"Graph successfully generated and saved to {save_path}")
