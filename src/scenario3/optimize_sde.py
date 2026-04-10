import random
from evaluation_function import evaluate_model
from prettytable import PrettyTable



def sample_params():
    cnn_weight = random.uniform(0.2, 0.8)

    return {
        "shake_threshold": random.uniform(0.1, 0.5),
        "entropy_threshold": random.uniform(0.3, 1),
        "cnn_weight": cnn_weight,
        "transformer_weight": 1 - cnn_weight,
        "min_steady_timesteps": random.randint(5, 20)
    }


def compute_score(results):
    return (
        + 2.0 * results["f1"]
        + 1.5 * results["transition_recall"]
        + 1.0 * results["transition_precision"]
        + 1.0 * results["accuracy"]
        - 1.5 * results["fn_ratio"]
        - 1.0 * results["fp_ratio"]
        - 0.01 * results["mean_delay"]
        - 0.01 * results["mean_duration"]
    )


def optimize(n_trials=50):

    best_score = -float("inf")
    best_params = None
    best_results = None
    optimized_parameters = PrettyTable(
        ['Shake Threshold', 'Entropy Threshold', 'CNN eight', 'Transformer weight', 'Min Steady Timesteps'])
    optimized_metrics = PrettyTable(
        ['Accuracy', 'Precision', 'Recall', 'F1', 'Transition Precision','Transition Recall','Mean Delay' , 'Mean Duration', 'FN ratio', 'FP ratio'])
    for i in range(n_trials):
        print(f"\nTrial {i+1}/{n_trials}")

        params = sample_params()
        results = evaluate_model(params)

        score = compute_score(results)

        print(f"Score: {score:.4f}")
        print(results)

        if score > best_score:
            best_score = score
            best_params = params
            best_results = results

    print("\n==============================")
    print("BEST RESULT")
    print("==============================")
    print("Score:", best_score)
    print("\nParams:")
    optimized_parameters.add_row(
        [round(best_params["shake_threshold"],4), round(best_params["entropy_threshold"],4), round(best_params["cnn_weight"],4),
         round(best_params["transformer_weight"],4), best_params["min_steady_timesteps"]])
    print(optimized_parameters)

    print("\nMetrics:")
    optimized_metrics.add_row(
        [round(best_results["accuracy"],4), round(best_results["precision"],4), round(best_results["recall"],4), round(best_results["f1"],4),
         best_results["transition_precision"], best_results["transition_recall"], round(best_results["mean_delay"],4), round(best_results["mean_duration"],4),
         best_results["fn_ratio"], best_results["fp_ratio"]])
    print(optimized_metrics)



if __name__ == "__main__":
    optimize(n_trials=100)