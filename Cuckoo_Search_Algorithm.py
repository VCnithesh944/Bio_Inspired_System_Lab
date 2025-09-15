import numpy as np
import random

# Problem Data
values = [60, 100, 120, 80, 30]     # item values
weights = [10, 20, 30, 40, 50]      # item weights
capacity = 100                        # knapsack capacity
num_items = len(values)

# CSA Parameters
n_nests = 10                          # number of nests
Pa = 0.25                              # discovery probability
max_iter = 100                         # maximum generations
beta = 1.5                              # for Levy flight

# Fitness evaluation
def fitness(solution):
    total_value = 0
    total_weight = 0
    for i in range(num_items):
        if solution[i] == 1:
            total_value += values[i]
            total_weight += weights[i]
    if total_weight > capacity:
        return 0   # penalize overweight solutions
    return total_value

# Generate a random solution
def random_solution():
    sol = np.random.randint(0, 2, num_items)
    return sol

# Levy flight simulated by flipping bits
def levy_flight(solution):
    new_sol = solution.copy()
    for i in range(num_items):
        if random.random() < 0.3:   # probability to flip each bit
            new_sol[i] = 1 - new_sol[i]
    return new_sol

# Main algorithm
def cuckoo_search():
    # Initialize nests
    nests = [random_solution() for _ in range(n_nests)]
    fitnesses = [fitness(sol) for sol in nests]

    best_idx = np.argmax(fitnesses)
    best_sol = nests[best_idx]
    best_fit = fitnesses[best_idx]

    for gen in range(max_iter):
        for i in range(n_nests):
            # Generate new solution
            new_sol = levy_flight(nests[i])
            new_fit = fitness(new_sol)

            # Replace if better
            if new_fit > fitnesses[i]:
                nests[i] = new_sol
                fitnesses[i] = new_fit

                # Update global best
                if new_fit > best_fit:
                    best_fit = new_fit
                    best_sol = new_sol

        # Abandon some nests with probability Pa
        for i in range(n_nests):
            if random.random() < Pa:
                nests[i] = random_solution()
                fitnesses[i] = fitness(nests[i])

        # Track best solution
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > best_fit:
            best_fit = fitnesses[best_idx]
            best_sol = nests[best_idx]

        print(f"Generation {gen+1}: Best Fitness = {best_fit}")

    return best_sol, best_fit

# Run the algorithm
best_solution, best_value = cuckoo_search()
print("\nBest Solution:", best_solution)
print("Total Value:", best_value)
total_weight = sum(weights[i] for i in range(num_items) if best_solution[i] == 1)
print("Total Weight:", total_weight)
