import optuna
from vehicle_model import NCMiata
from track import Track
from simulator import LapSimulator

def objective(trial):
    # Define the search space for suspension parameters
    spring_k_f = trial.suggest_float("spring_k_f", 40000, 120000) # 4k to 12k
    spring_k_r = trial.suggest_float("spring_k_r", 20000, 80000)  # 2k to 8k
    arb_k_f = trial.suggest_float("arb_k_f", 0, 30000)
    arb_k_r = trial.suggest_float("arb_k_r", 0, 15000)

    # Setup the car
    car = NCMiata()
    car.spring_k_f = spring_k_f
    car.spring_k_r = spring_k_r
    car.arb_k_f = arb_k_f
    car.arb_k_r = arb_k_r

    # Run simulation
    track = Track()
    sim = LapSimulator(car, track)
    
    # We want to minimize lap time
    # In v1, the simulator is too simple to show spring effects fully
    # because it doesn't have load transfer yet.
    # But this sets up the pipeline.
    lap_time, _ = sim.solve()
    
    return lap_time

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best parameters:", study.best_params)
    print("Best lap time:", study.best_value)
