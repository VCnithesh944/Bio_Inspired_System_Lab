import random

# Objective function
def f(x):
    return -(x**2) + 5*x + 20

# Particle class
class Particle:
    def __init__(self, bounds):
        self.position = random.uniform(bounds[0], bounds[1])
        self.velocity = 0.0
        self.best_position = self.position
        self.best_value = f(self.position)

    def update_velocity(self, gbest, w=0.5, c1=1.5, c2=1.5):
        r1, r2 = random.random(), random.random()
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (gbest - self.position)
        self.velocity = w*self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        # Keep within bounds
        self.position = max(bounds[0], min(self.position, bounds[1]))
        value = f(self.position)
        if value > self.best_value:
            self.best_value = value
            self.best_position = self.position

# PSO implementation
def pso(num_particles=9, iterations=20, bounds=(-10, 10)):
    # Initialize particles
    particles = [Particle(bounds) for _ in range(num_particles)]
    gbest = max(particles, key=lambda p: p.best_value)
    gbest_position = gbest.best_position
    gbest_value = gbest.best_value

    for it in range(iterations):
        for p in particles:
            p.update_velocity(gbest_position)
            p.update_position(bounds)

        # Update global best
        candidate = max(particles, key=lambda p: p.best_value)
        if candidate.best_value > gbest_value:
            gbest_value = candidate.best_value
            gbest_position = candidate.best_position

        print(f"Iteration {it+1}: Best x = {gbest_position:.4f}, Best f(x) = {gbest_value:.4f}")

    return gbest_position, gbest_value

# Run PSO
best_x, best_val = pso()
print("\nFinal Best Solution:")
print("x =", round(best_x, 4))
print("f(x) =", round(best_val, 4))
