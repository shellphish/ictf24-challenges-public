import numpy as np
import matplotlib.pyplot as plt

def generate_points(n_points=2000, noise_level=0.1, a=1):
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x0 = a * np.sin(theta)
    y0 = a * np.sin(theta) * np.cos(theta)
    x_noise = x0 + np.random.normal(0, noise_level, n_points)
    y_noise = y0 + np.random.normal(0, noise_level, n_points)
    return x_noise, y_noise

def plot_points(x, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=1, color='blue')
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == "__main__":
    x, y = generate_points()
    plot_points(x, y)
