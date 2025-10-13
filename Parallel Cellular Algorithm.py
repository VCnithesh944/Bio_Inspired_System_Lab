import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# --- Functions (create_sample_image and update_cell_state) are unchanged ---

def create_sample_image(size=(100, 100), tumor_pos=(40, 60), tumor_size=15):
    """Creates a simple grayscale image with a background and a brighter tumor region."""
    image = np.random.normal(loc=50, scale=5, size=size).astype(np.uint8)
    x_start, x_end = tumor_pos[0], tumor_pos[0] + tumor_size
    y_start, y_end = tumor_pos[1], tumor_pos[1] + tumor_size
    image[x_start:x_end, y_start:y_end] = np.random.normal(loc=150, scale=10, size=(tumor_size, tumor_size))
    return np.clip(image, 0, 255)

def update_cell_state(args):
    """This function represents the rule for a single cell."""
    row_idx, col_idx, grid, neighborhood_size, threshold = args
    half_size = neighborhood_size // 2
    row_min = max(0, row_idx - half_size)
    row_max = min(grid.shape[0], row_idx + half_size + 1)
    col_min = max(0, col_idx - half_size)
    col_max = min(grid.shape[1], col_idx + half_size + 1)
    neighborhood = grid[row_min:row_max, col_min:col_max]
    neighbor_avg_intensity = np.mean(neighborhood)
    cell_intensity = grid[row_idx, col_idx]
    intensity_difference = abs(cell_intensity - neighbor_avg_intensity)
    if intensity_difference > threshold:
        return 1
    else:
        return 0

# --- Main Algorithm Execution ---

if __name__ == '__main__':
    # Initialize Parameters
    GRID_SIZE = (100, 100)
    NEIGHBORHOOD_SIZE = 3

    # --- NEW: User Input ---
    try:
        # Prompt the user to set the detection sensitivity
        user_threshold = int(input("Enter the detection threshold (e.g., 25): "))
        INTENSITY_THRESHOLD = user_threshold
    except ValueError:
        print("Invalid input. Using default threshold of 25.")
        INTENSITY_THRESHOLD = 25

    # Initialize Population
    original_image = create_sample_image(size=GRID_SIZE)
    
    # Prepare arguments for parallel processing
    tasks = [(r, c, original_image, NEIGHBORHOOD_SIZE, INTENSITY_THRESHOLD) 
             for r in range(GRID_SIZE[0]) for c in range(GRID_SIZE[1])]

    print(f"\nStarting parallel processing on {cpu_count()} CPU cores...")

    # Execute the process once
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(update_cell_state, tasks)
    
    processed_grid = np.array(results).reshape(GRID_SIZE)
    
    print("Processing complete.")

    # --- NEW: Numerical Output ---
    # Calculate the area of the detected region by counting the '1's
    detected_area_pixels = np.sum(processed_grid)
    total_pixels = GRID_SIZE[0] * GRID_SIZE[1]
    percentage_of_image = (detected_area_pixels / total_pixels) * 100

    print("\n--- Numerical Diagnosis Results ---")
    print(f"Detected Anomaly Area: {detected_area_pixels} pixels")
    print(f"Percentage of Image Affected: {percentage_of_image:.2f}%")
    print("---------------------------------\n")
    print("Displaying visual results...")

    # --- Visual Output (unchanged) ---
    overlay_image = np.stack([original_image]*3, axis=-1)
    overlay_image[processed_grid == 1] = [255, 0, 0] # Red

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Medical Image')
    axes[0].axis('off')

    axes[1].imshow(processed_grid, cmap='hot')
    axes[1].set_title('Processed Grid (Detected Region)')
    axes[1].axis('off')
    
    axes[2].imshow(overlay_image)
    axes[2].set_title('Detection Overlay on Original Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
