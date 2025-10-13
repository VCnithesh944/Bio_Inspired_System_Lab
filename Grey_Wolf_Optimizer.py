import numpy as np

# --- Fitness function (replace with any objective) ---
def sphere(x):
    return np.sum(x**2)  # minimum = 0 at [0,0,...,0]

# --- Grey Wolf Optimizer ---
def GWO(obj_func, dim, n_wolves=20, max_iter=100, lb=-10, ub=10):
    # Initialize wolves (population)
    wolves = np.random.uniform(lb, ub, (n_wolves, dim))

    # Initialize leaders
    alpha, beta, delta = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    alpha_score, beta_score, delta_score = float("inf"), float("inf"), float("inf")

    for t in range(max_iter):
        # --- Evaluate wolves ---
        for i in range(n_wolves):
            fitness = obj_func(wolves[i])

            # Update alpha, beta, delta
            if fitness < alpha_score:
                alpha_score, alpha = fitness, wolves[i].copy()
            elif fitness < beta_score:
                beta_score, beta = fitness, wolves[i].copy()
            elif fitness < delta_score:
                delta_score, delta = fitness, wolves[i].copy()

        # --- Identify omegas (all other wolves) ---
        omega_indices = [i for i in range(n_wolves) 
                         if not (np.array_equal(wolves[i], alpha) or 
                                 np.array_equal(wolves[i], beta) or 
                                 np.array_equal(wolves[i], delta))]

        # --- Update positions ---
        a = 2 - 2 * (t / max_iter)  # decreases from 2 → 0

        for i in range(n_wolves):
            # Each wolf (including omegas) moves guided by α, β, δ
            for d in range(dim):
                # α influence
                r1, r2 = np.random.rand(), np.random.rand()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha[d] - wolves[i][d])
                X1 = alpha[d] - A1 * D_alpha

                # β influence
                r1, r2 = np.random.rand(), np.random.rand()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * beta[d] - wolves[i][d])
                X2 = beta[d] - A2 * D_beta

                # δ influence
                r1, r2 = np.random.rand(), np.random.rand()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * delta[d] - wolves[i][d])
                X3 = delta[d] - A3 * D_delta

                # New position = average of α, β, δ influences
                wolves[i][d] = (X1 + X2 + X3) / 3

            # Keep inside search space
            wolves[i] = np.clip(wolves[i], lb, ub)

        # --- Debug (optional): print iteration status ---
        print(f"Iter {t+1}/{max_iter} | Alpha fitness: {alpha_score:.6f}")

    return alpha_score, alpha, beta_score, beta, delta_score, delta, wolves[omega_indices]

# --- Run Example ---
best_score, best_pos, beta_score, beta_pos, delta_score, delta_pos, omegas = GWO(sphere, dim=3)

print("\nBest (Alpha):", best_pos, "| Fitness:", best_score)
print("Second (Beta):", beta_pos, "| Fitness:", beta_score)
print("Third (Delta):", delta_pos, "| Fitness:", delta_score)
print("Number of Omegas:", len(omegas))
