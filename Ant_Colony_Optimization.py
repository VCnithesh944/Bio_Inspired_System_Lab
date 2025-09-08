import numpy as np
import random

class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=2):
        """
        distances   : 2D numpy array with distances between cities
        n_ants      : number of ants per iteration
        n_best      : number of best ants who deposit pheromone
        n_iterations: number of iterations
        decay       : pheromone evaporation rate
        alpha       : importance of pheromone
        beta        : importance of heuristic (1/distance)
        """
        self.distances  = distances
        self.pheromone  = np.ones(self.distances.shape) / len(distances)
        self.all_inds   = range(len(distances))
        self.n_ants     = n_ants
        self.n_best     = n_best
        self.n_iterations = n_iterations
        self.decay      = decay
        self.alpha      = alpha
        self.beta       = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        
        for i in range(self.n_iterations):
            all_paths = self.construct_paths()
            self.spread_pheromone(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            
            self.pheromone *=(1-self.decay)  # evaporation
        
        return all_time_shortest_path

    def construct_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = self.gen_path(0)  # start at city 0
            all_paths.append((path, self.path_distance(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for _ in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))  # return to start
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * ((1.0/dist) ** self.beta)
        norm_row = row / row.sum()
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move

    def spread_pheromone(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def path_distance(self, path):
        total_dist = 0
        for move in path:
            total_dist += self.distances[move]
        return total_dist


if __name__ == "__main__":
    # Example with 5 cities
    cities = np.array([
        (0,0), (1,5), (5,2), (6,6), (8,3)
    ])
    
    distances = np.sqrt((np.square(cities[:, np.newaxis] - cities).sum(axis=2)))
    
    ant_colony = AntColony(distances, n_ants=10, n_best=5, n_iterations=100, decay=0.5, alpha=1, beta=2)
    shortest_path = ant_colony.run()
    print("Best Path:", shortest_path)
