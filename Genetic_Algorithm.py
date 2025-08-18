import math

def binary_to_int(bin_str):
    return int(bin_str, 2)

def fitness(x):
    return x ** 2  # You can replace this with a DNA-specific fitness later

def print_table(population, fitness_values, prob_values, expected_output, actual_count):
    print(f"{'DNA Seq No':<10} | {'DNA Sequence (bin)':<18} | {'x Value':<7} | {'Fitness (DNA match)':<20} | {'% Prob':<7} | {'Expected Count':<15} | {'Actual Count':<12}")
    print("-"*105)
    for i, (chrom, x, fit, prob, exp, act) in enumerate(zip(population, 
                                                            [binary_to_int(c) for c in population], 
                                                            fitness_values, prob_values, expected_output, actual_count)):
        print(f"{i:<10} | {chrom:<18} | {x:<7} | {fit:<20} | {prob*100:<7.2f} | {exp:<15.2f} | {act:<12}")

def selection(population):
    x_values = [binary_to_int(ch) for ch in population]
    fitness_values = [fitness(x) for x in x_values]
    
    total_fitness = sum(fitness_values)
    max_fitness = max(fitness_values)
    avg_fitness = total_fitness / len(fitness_values)
    
    # Probability for each DNA chromosome: f(x)/sum(f(x))
    prob_values = [f/total_fitness for f in fitness_values]
    
    # Expected output: f(x)*N / sum(f(x))
    expected_output = [(f * len(population)) / total_fitness for f in fitness_values]
    
    # Actual count: closest integer to expected output (standard rounding)
    actual_count = [round(val) for val in expected_output]
    
    print("\nDNA Matching Table:")
    print_table(population, fitness_values, prob_values, expected_output, actual_count)
    print(f"\nSum of Fitness (DNA matches): {total_fitness}, Max Fitness: {max_fitness}, Avg Fitness: {avg_fitness:.2f}")
    return prob_values, expected_output, actual_count, max_fitness

def print_mating_pool(pool, title):
    print(f"\n{title}:")
    print(f"{'DNA Seq No':<10} | {'DNA Sequence (bin)':<18}")
    print("-"*30)
    for i, chrom in enumerate(pool):
        print(f"{i:<10} | {chrom:<18}")

def mating_pool_selection(population, pool_size, bit_pos):
    mating_pool = population[:pool_size]
    
    print_mating_pool(mating_pool, f"Mating Pool before crossover at bit {bit_pos} (DNA sequence bits)")
    
    for i in range(0, pool_size-1, 2):
        chrom1 = list(mating_pool[i])
        chrom2 = list(mating_pool[i+1])
        
        chrom1_bit = chrom1[bit_pos]
        chrom2_bit = chrom2[bit_pos]
        
        chrom1[bit_pos], chrom2[bit_pos] = chrom2_bit, chrom1_bit
        
        mating_pool[i] = "".join(chrom1)
        mating_pool[i+1] = "".join(chrom2)
    
    print_mating_pool(mating_pool, f"Mating Pool after crossover at bit {bit_pos} (DNA sequence bits)")
    
    new_population = population.copy()
    new_population[:pool_size] = mating_pool
    return new_population

def mutation(population, mutation_masks):
    print(f"\nMutation masks applied to DNA sequences:")
    for i, mask in enumerate(mutation_masks):
        print(f"DNA Chromosome {i}: {mask}")
    
    new_population = []
    for chrom, mutation_value in zip(population, mutation_masks):
        chrom_list = list(chrom)
        mutation_mask = list(mutation_value)
        for i, m in enumerate(mutation_mask):
            if m == '1':
                # Flip the bit (1's complement)
                chrom_list[i] = '1' if chrom_list[i] == '0' else '0'
        new_population.append("".join(chrom_list))
    
    print(f"Population of DNA sequences after mutation: {new_population}")
    return new_population

def main():
    print("=== DNA Sequence Matching Genetic Algorithm ===")
    num_chromosomes = int(input("Enter number of DNA sequences (chromosomes): "))
    length_chromosome = int(input("Enter length of each DNA sequence (in bits): "))
    
    population = []
    print("Enter DNA sequences as binary strings (e.g. 1100...):")
    for _ in range(num_chromosomes):
        while True:
            chrom = input()
            if len(chrom) == length_chromosome and set(chrom).issubset({'0','1'}):
                population.append(chrom)
                break
            else:
                print(f"Invalid input. Enter binary string of length {length_chromosome}.")
    
    max_fitness_old = -1
    
    while True:
        prob_values, expected_output, actual_count, max_fitness = selection(population)
        
        if max_fitness == max_fitness_old or abs(max_fitness - max_fitness_old) < 1e-5:
            print("\nDNA matching fitness stabilized, stopping evolution.")
            break
        
        max_fitness_old = max_fitness
        
        # Input mating pool size with validation
        while True:
            pool_size = int(input(f"\nEnter mating pool size (even number ≤ DNA sequence bit length {length_chromosome} and ≤ population size {len(population)}): "))
            if pool_size <= length_chromosome and pool_size <= len(population) and pool_size > 0 and pool_size % 2 == 0:
                break
            else:
                print("Invalid input. Mating pool size must be an even number, > 0, and ≤ DNA sequence length and population size.")
        
        # Input bit position for crossover
        while True:
            bit_pos = int(input(f"Enter bit position to swap (0 to {length_chromosome-1}): "))
            if 0 <= bit_pos < length_chromosome:
                break
            else:
                print(f"Invalid bit position. Must be between 0 and {length_chromosome-1}.")
        
        population = mating_pool_selection(population, pool_size, bit_pos)
        
        # Mutation input: one mask per chromosome
        mutation_masks = []
        print(f"\nEnter mutation binary strings (one per DNA sequence, length {length_chromosome}):")
        for i in range(len(population)):
            while True:
                mask = input(f"Mutation mask for DNA chromosome {i}: ")
                if len(mask) == length_chromosome and set(mask).issubset({'0','1'}):
                    mutation_masks.append(mask)
                    break
                else:
                    print(f"Invalid input. Enter binary string of length {length_chromosome}.")
        
        population = mutation(population, mutation_masks)
        
if __name__ == "__main__":
