import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Particle Filter Simulation for a point moving in a noisy 2D grid (Real-Time Visualization, Multi-Sensor, Adaptive Resampling)
# --------------------------------------------------------------------------------------
# This example demonstrates a particle filter with multiple sensors (each with its own uncertainty)
# and adaptive resampling based on the effective sample size (ESS) threshold.

# Parameters
NUM_PARTICLES = 500      # Number of particles
NUM_STEPS = 30           # Number of time steps
GRID_SIZE = 100          # Size of the grid (GRID_SIZE x GRID_SIZE)
MOVE_STD = 2.0           # Standard deviation of motion noise

# Sensor parameters: each sensor has its own observation noise (std)
SENSOR_STDS = [5.0, 10.0, 3.0]  # Example: 3 sensors with different uncertainties
NUM_SENSORS = len(SENSOR_STDS)

# Adaptive resampling threshold (as a fraction of NUM_PARTICLES)
ESS_THRESHOLD = 0.5  # Resample if effective sample size < 50% of NUM_PARTICLES

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
rmse_list = []  # To store RMSE at each step

# Set up the plot
fig, ax = plt.subplots(figsize=(8, 8))
sc_particles = ax.scatter(particles[:, 0], particles[:, 1], color='b', alpha=0.2, label='Particles')
true_line, = ax.plot([], [], 'g-', label='True Position')
est_line, = ax.plot([], [], 'r--', label='Estimated Position (PF)')
true_dot, = ax.plot([], [], 'go', markersize=10, label='True Current')
est_dot, = ax.plot([], [], 'rx', markersize=10, label='Estimated Current')
rmse_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Particle Filter Simulation in Noisy 2D Grid (Multi-Sensor, Adaptive Resampling)')
ax.legend()
ax.grid(True)

# Animation update function
def update(frame):
    global true_pos, particles, weights
    # 1. Move the true position (with motion noise)
    true_pos = true_pos + MOVE + np.random.normal(0, MOVE_STD, size=2)
    true_positions.append(true_pos.copy())

    # 2. Simulate multiple noisy observations from different sensors
    observations = [true_pos + np.random.normal(0, std, size=2) for std in SENSOR_STDS]

    # 3. Move all particles according to the motion model (with noise)
    particles += MOVE + np.random.normal(0, MOVE_STD, size=(NUM_PARTICLES, 2))

    # 4. Compute weights based on all sensor observations (multiply likelihoods)
    weights = np.ones(NUM_PARTICLES)
    for obs, std in zip(observations, SENSOR_STDS):
        distances = np.linalg.norm(particles - obs, axis=1)
        weights *= np.exp(-0.5 * (distances / std) ** 2) / (2 * np.pi * std**2)
    weights += 1e-300  # avoid zeros
    weights /= np.sum(weights)  # normalize

    # 5. Compute effective sample size (ESS)
    ess = 1.0 / np.sum(weights ** 2)

    # 6. Adaptive resampling: only resample if ESS is below threshold
    if ess < ESS_THRESHOLD * NUM_PARTICLES:
        # Systematic resampling
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

    # 7. Estimate position as the mean of the particles
    estimated_pos = np.mean(particles, axis=0)
    estimated_positions.append(estimated_pos)

    # 8. Compute RMSE (Root Mean Squared Error) between estimated and true position
    rmse = np.linalg.norm(estimated_pos - true_pos)
    rmse_list.append(rmse)

    # Update plot data
    sc_particles.set_offsets(particles)
    true_line.set_data(np.array(true_positions)[:, 0], np.array(true_positions)[:, 1])
    est_line.set_data(np.array(estimated_positions)[:, 0], np.array(estimated_positions)[:, 1])
    true_dot.set_data(true_pos[0], true_pos[1])
    est_dot.set_data(estimated_pos[0], estimated_pos[1])
    rmse_text.set_text(f'RMSE: {rmse:.2f}')
    return sc_particles, true_line, est_line, true_dot, est_dot, rmse_text

ani = FuncAnimation(fig, update, frames=NUM_STEPS, interval=300, blit=True, repeat=False)
plt.show()

# Print final average RMSE
if rmse_list:
    avg_rmse = np.mean(rmse_list)
    print(f'Average RMSE over {NUM_STEPS} steps: {avg_rmse:.2f}')

# -----------------------------
# Algorithm Steps Explained:
# 1. Initialize particles randomly in the grid.
# 2. At each time step:
#    a. Move the true position (with noise).
#    b. Simulate multiple noisy observations from different sensors.
#    c. Move all particles according to the motion model (with noise).
#    d. Update particle weights by multiplying likelihoods from all sensors.
#    e. Compute effective sample size (ESS) and resample only if ESS is low.
#    f. Estimate the position as the mean of the particles.
#    g. Compute and display RMSE between estimated and true position.
# 3. Plot the true and estimated positions over time.
# -----------------------------
