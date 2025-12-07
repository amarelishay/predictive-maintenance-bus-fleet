
# Pipeline wrapper that executes the original training logic as a module

from src.model import original_logic

def run():
    print("▶ Running full predictive‑maintenance training pipeline...")
    # original_logic executes everything internally
    print("✔ Finished")

if __name__ == "__main__":
    run()
