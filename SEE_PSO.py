import numpy as np
import random


KEY_LENGTH = 16      # bits in key (toy; real ciphers are much larger)
STREAM_LENGTH = 64   # number of keystream bits we use for attack

# --- PSO parameters ---
NUM_PARTICLES = 30
MAX_ITER = 1000
W = 0.7              # inertia
C1 = 1.5             # cognitive
C2 = 1.5             # social


def int_to_bits(x, length):
    """
    Convert integer x into a numpy array of bits (0/1) of given length.
    """
    return np.array([(x >> i) & 1 for i in range(length)][::-1], dtype=np.int8)


def bits_to_int(bits):
    """
    Convert numpy array of bits (0/1) to integer.
    """
    x = 0
    for b in bits:
        x = (x << 1) | int(b)
    return x



def nonlinear_feedback(state):
    """
    Example nonlinear Boolean function over the state bits.
    This is intentionally simple and artificial:
        f(x) = x0 ^ (x1 & x3) ^ (x2 & x4) ^ x5
    where xi are some bits of the state.
    """
    x0 = state[0]
    x1 = state[1]
    x2 = state[2]
    x3 = state[3]
    x4 = state[4]
    x5 = state[5]
    return x0 ^ (x1 & x3) ^ (x2 & x4) ^ x5


def keystream_generator(key_bits, length):
    """
    Toy stream cipher:
      - State is initialized as "key_bits" (padded/truncated).
      - Each step:
          z_t = nonlinear_feedback(state)
          output z_t as keystream bit
          shift state and insert z_t at the end
    """
    # For this toy example, state size = KEY_LENGTH
    state = key_bits.copy()
    stream = np.zeros(length, dtype=np.int8)

    for t in range(length):
        z = nonlinear_feedback(state)
        stream[t] = z

        # Shift left and insert new bit at the end
        state[:-1] = state[1:]
        state[-1] = z

    return stream


def encrypt_stream(plaintext_bits, keystream_bits):
    """
    XOR plaintext with keystream to get ciphertext.
    """
    return plaintext_bits ^ keystream_bits


def decrypt_stream(ciphertext_bits, keystream_bits):
    """
    XOR ciphertext with keystream to get plaintext.
    """
    return ciphertext_bits ^ keystream_bits



def generate_known_plaintext_instance():
    """
    Simulate one known-plaintext scenario:
      - Choose a random secret key.
      - Generate keystream.
      - Encrypt a random plaintext.
      - Return: secret_key_bits, plaintext, ciphertext, observed_keystream
    """
    true_key_bits = np.array(
        [0,1,0,0,1,1,0,0,1,1,0,1,1,1,1,1],
        dtype=np.int8
    )

    # Random plaintext (attacker knows this)
    plaintext_bits = np.random.randint(0, 2, STREAM_LENGTH, dtype=np.int8)

    # Generate keystream and ciphertext
    keystream_bits = keystream_generator(true_key_bits, STREAM_LENGTH)
    ciphertext_bits = encrypt_stream(plaintext_bits, keystream_bits)

    # Attacker computes observed keystream
    observed_keystream = ciphertext_bits ^ plaintext_bits

    return true_key_bits, plaintext_bits, ciphertext_bits, observed_keystream



def fitness(candidate_key_bits, observed_keystream):
    """
    Fitness = number of matching keystream bits between:
        keystream(candidate_key) and observed_keystream.
    The higher the fitness, the closer the candidate key is to real key.
    """
    generated_keystream = keystream_generator(candidate_key_bits, len(observed_keystream))
    matches = np.sum(generated_keystream == observed_keystream)
    return matches  # can normalize if needed


class Particle:
    def __init__(self, observed_keystream):
        self.observed_keystream = observed_keystream

        # Binary representation of key: position in {0,1}^KEY_LENGTH
        self.position = np.random.randint(0, 2, KEY_LENGTH)
        self.velocity = np.random.uniform(-1, 1, KEY_LENGTH)

        self.best_position = self.position.copy()
        self.best_score = fitness(self.position, self.observed_keystream)

    def update_velocity(self, global_best):
        r1 = np.random.rand(KEY_LENGTH)
        r2 = np.random.rand(KEY_LENGTH)

        cognitive = C1 * r1 * (self.best_position - self.position)
        social = C2 * r2 * (global_best - self.position)

        self.velocity = W * self.velocity + cognitive + social

    def update_position(self):
        # Sigmoid to convert velocity → probability
        sigmoid = 1 / (1 + np.exp(-self.velocity))
        rand_vals = np.random.rand(KEY_LENGTH)

        # Binary PSO: update bits based on probability
        self.position = np.where(rand_vals < sigmoid, 1, 0)

        # Update personal best
        score = fitness(self.position, self.observed_keystream)
        if score > self.best_score:
            self.best_score = score
            self.best_position = self.position.copy()


def pso_key_recovery(observed_keystream, num_particles=NUM_PARTICLES, max_iter=MAX_ITER):
    # Initialize swarm
    swarm = [Particle(observed_keystream) for _ in range(num_particles)]

    # Initialize global best
    global_best = swarm[0].best_position.copy()
    global_best_score = swarm[0].best_score

    for iteration in range(max_iter):
        for particle in swarm:
            particle.update_velocity(global_best)
            particle.update_position()

            # Update global best
            if particle.best_score > global_best_score:
                global_best_score = particle.best_score
                global_best = particle.best_position.copy()
        if iteration % 4 == 0:

            print(f"Iteration {iteration+1}/{max_iter} | Best matches = {global_best_score}/{len(observed_keystream)}")
            if global_best_score == 64:
                break

    return global_best, global_best_score


# ============================================================
# 8. EXPERIMENT: RUN THE ATTACK
# ============================================================

if __name__ == "__main__":
    # Generate a random scenario
    true_key_bits, P, C, observed_keystream = generate_known_plaintext_instance()

    print("True key bits:     ", true_key_bits, " (int:", bits_to_int(true_key_bits), ")")

    # Run PSO-based cryptanalysis
    recovered_key_bits, recovered_score = pso_key_recovery(observed_keystream)

    print("\nRecovered key bits:", recovered_key_bits, " (int:", bits_to_int(recovered_key_bits), ")")
    print("Recovered fitness (matching keystream bits):", recovered_score, "/", STREAM_LENGTH)

    # Check if recovered key is correct
    is_correct = np.array_equal(true_key_bits, recovered_key_bits)
    print("\nKey recovered correctly?", is_correct)
