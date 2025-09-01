import random

# Example setup
tasks = [3, 2, 4, 1, 2]      # Resource demand of each task
resources = [5, 5, 5]        # Capacity of each resource
num_tasks = len(tasks)
num_resources = len(resources)
population_size = 10
generations = 50

# Generate initial population
def init_population():
    return [[random.randint(0, num_resources - 1) for _ in range(num_tasks)]
            for _ in range(population_size)]

# Fitness function
def fitness(chromosome):
    load = [0] * num_resources
    for task, res in enumerate(chromosome):
        load[res] += tasks[task]
    # Penalize overload
    penalty = sum(max(0, load[i] - resources[i]) for i in range(num_resources))
    return -penalty  # Higher is better (less penalty)

# Selection
def selection(pop):
    pop.sort(key=fitness, reverse=True)
    return pop[:population_size//2]

# Crossover
def crossover(parent1, parent2):
    point = random.randint(1, num_tasks-1)
    return parent1[:point] + parent2[point:]

# Mutation
def mutate(chromosome):
    if random.random() < 0.2:  # mutation rate
        i = random.randint(0, num_tasks-1)
        chromosome[i] = random.randint(0, num_resources-1)
    return chromosome

# GA main loop
population = init_population()
for g in range(generations):
    selected = selection(population)
    children = []
    for i in range(len(selected)//2):
        parent1, parent2 = selected[2*i], selected[2*i+1]
        child = crossover(parent1, parent2)
        children.append(mutate(child))
    population = selected + children

# Best solution
best = max(population, key=fitness)
print("Best Allocation:", best)
print("Fitness:", fitness(best))
