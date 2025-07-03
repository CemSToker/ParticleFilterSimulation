import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Particle Filter Simulation for a point moving in a noisy 2D grid (Real-Time Visualization)
# -------------------------------------------------------------
# This example demonstrates a simple particle filter (Monte Carlo Localization)
# to estimate the position of a point moving in a 2D grid with noisy motion and noisy observations.

# Parameters
NUM_PARTICLES = 500      # Number of particles
NUM_STEPS = 30           # Number of time steps
GRID_SIZE = 100          # Size of the grid (GRID_SIZE x GRID_SIZE)
MOVE_STD = 2.0           # Standard deviation of motion noise
OBS_STD = 5.0            # Standard deviation of observation noise

# True motion model: the point moves right and up by 1 unit per step
MOVE = np.array([1, 1])

# Initialize true position
true_pos = np.array([20.0, 20.0])

# Initialize particles randomly in the grid
particles = np.random.uniform(0, GRID_SIZE, size=(NUM_PARTICLES, 2))

# Initialize weights uniformly
weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES

# For plotting
true_positions = [true_pos.copy()]
estimated_positions = [np.mean(particles, axis=0)]

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
sc_particles = ax.scatter(particles[:, 0], particles[:, 1], color='b', alpha=0.2, label='Particles')
true_line, = ax.plot([], [], 'g-', label='True Position')
est_line, = ax.plot([], [], 'r--', label='Estimated Position (PF)')
true_dot, = ax.plot([], [], 'go', markersize=10, label='True Current')
est_dot, = ax.plot([], [], 'rx', markersize=10, label='Estimated Current')
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Particle Filter Simulation in Noisy 2D Grid (Real-Time)')
ax.legend()
ax.grid(True)

# Animation update function
def update(frame):
    global true_pos, particles, weights
    # 1. Move the true position (with motion noise)
    true_pos = true_pos + MOVE + np.random.normal(0, MOVE_STD, size=2)
    true_positions.append(true_pos.copy())

    # 2. Simulate a noisy observation of the true position
    observation = true_pos + np.random.normal(0, OBS_STD, size=2)

    # 3. Move all particles according to the motion model (with noise)
    particles += MOVE + np.random.normal(0, MOVE_STD, size=(NUM_PARTICLES, 2))

    # 4. Compute weights based on how likely each particle is given the observation
    distances = np.linalg.norm(particles - observation, axis=1)
    weights = np.exp(-0.5 * (distances / OBS_STD) ** 2)
    weights += 1e-300  # avoid zeros
    weights /= np.sum(weights)  # normalize

    # 5. Resample particles according to their weights (systematic resampling)
    indices = np.zeros(NUM_PARTICLES, dtype=int)
    cumsum = np.cumsum(weights)
    step = 1.0 / NUM_PARTICLES
    u0 = np.random.uniform(0, step)
    j = 0
    for i in range(NUM_PARTICLES):
        u = u0 + i * step
        while u > cumsum[j]:
            j += 1
        indices[i] = j
    particles = particles[indices]
    weights = np.ones(NUM_PARTICLES) / NUM_PARTICLES  # reset weights

    # 6. Estimate position as the mean of the particles
    estimated_pos = np.mean(particles, axis=0)
    estimated_positions.append(estimated_pos)

    # Update plot data
    sc_particles.set_offsets(particles)
    true_line.set_data(np.array(true_positions)[:, 0], np.array(true_positions)[:, 1])
    est_line.set_data(np.array(estimated_positions)[:, 0], np.array(estimated_positions)[:, 1])
    true_dot.set_data(true_pos[0], true_pos[1])
    est_dot.set_data(estimated_pos[0], estimated_pos[1])
    return sc_particles, true_line, est_line, true_dot, est_dot

ani = FuncAnimation(fig, update, frames=NUM_STEPS, interval=300, blit=True, repeat=False)
plt.show()

# -----------------------------
# Algorithm Steps Explained:
# 1. Initialize particles randomly in the grid.
# 2. At each time step:
#    a. Move the true position (with noise).
#    b. Simulate a noisy observation.
#    c. Move all particles according to the motion model (with noise).
#    d. Update particle weights based on observation likelihood.
#    e. Resample particles according to their weights.
#    f. Estimate the position as the mean of the particles.
# 3. Plot the true and estimated positions over time.
# -----------------------------
