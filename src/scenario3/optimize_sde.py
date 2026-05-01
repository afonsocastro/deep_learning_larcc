import optuna
from evaluation_function import evaluate_model
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import numpy as np


from itertools import combinations


import numpy as np
import matplotlib.pyplot as plt


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def compute_cost(pareto_points, w_fp=2.0):
    fp = normalize(pareto_points[:, 1])
    delay = normalize(pareto_points[:, 2])
    duration = normalize(pareto_points[:, 3])

    return w_fp * fp + delay + duration


def plot_final_figure(all_points, pareto_points, knees):
    """
    all_points: (N,4)
    pareto_points: (M,4)
    knees: lista de dicts [{"w_fp":..., "knee": [...]}]
    """

    all_points = np.array(all_points)
    pareto_points = np.array(pareto_points)

    # separar knees
    knee_low = knees[0]["knee"]        # w=1.0
    knee_stable = knees[1]["knee"]     # w>=1.5 (todos iguais)
    knee_final = knee_stable           # escolha final

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # ---------------------------
    # (1) Accuracy vs FP
    # ---------------------------
    ax = axs[0, 0]
    ax.scatter(all_points[:, 1], all_points[:, 0], alpha=0.3)
    ax.scatter(pareto_points[:, 1], pareto_points[:, 0])

    ax.scatter(knee_low[1], knee_low[0], marker='X', s=200, label="Knee w=1.0")
    ax.scatter(knee_stable[1], knee_stable[0], marker='X', s=200, label="Knee w≥1.5")
    ax.scatter(knee_final[1], knee_final[0], marker='*', s=300, label="Selected")

    ax.set_xlabel("FP")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs False Positives")
    ax.grid()

    # ---------------------------
    # (2) Accuracy vs Delay
    # ---------------------------
    ax = axs[0, 1]
    ax.scatter(all_points[:, 2], all_points[:, 0], alpha=0.3)
    ax.scatter(pareto_points[:, 2], pareto_points[:, 0])

    ax.scatter(knee_low[2], knee_low[0], marker='X', s=200)
    ax.scatter(knee_stable[2], knee_stable[0], marker='X', s=200)
    ax.scatter(knee_final[2], knee_final[0], marker='*', s=300)

    ax.set_xlabel("Delay")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Delay")
    ax.grid()

    # ---------------------------
    # (3) Accuracy vs Duration
    # ---------------------------
    ax = axs[1, 0]
    ax.scatter(all_points[:, 3], all_points[:, 0], alpha=0.3)
    ax.scatter(pareto_points[:, 3], pareto_points[:, 0])

    ax.scatter(knee_low[3], knee_low[0], marker='X', s=200)
    ax.scatter(knee_stable[3], knee_stable[0], marker='X', s=200)
    ax.scatter(knee_final[3], knee_final[0], marker='*', s=300)

    ax.set_xlabel("Duration")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Duration")
    ax.grid()

    # ---------------------------
    # (4) Cost vs Accuracy
    # ---------------------------
    ax = axs[1, 1]

    cost_all = compute_cost(all_points)
    cost_pareto = compute_cost(pareto_points)

    ax.scatter(cost_all, all_points[:, 0], alpha=0.3)
    ax.scatter(cost_pareto, pareto_points[:, 0])

    # custos dos knees
    knee_low_cost = compute_cost(np.array([knee_low]))[0]
    knee_stable_cost = compute_cost(np.array([knee_stable]))[0]

    ax.scatter(knee_low_cost, knee_low[0], marker='X', s=200)
    ax.scatter(knee_stable_cost, knee_stable[0], marker='X', s=200)
    ax.scatter(knee_stable_cost, knee_stable[0], marker='*', s=300)

    ax.set_xlabel("Cost")
    ax.set_ylabel("Accuracy")
    ax.set_title("Knee Selection (Accuracy vs Cost)")
    ax.grid()

    # legenda global
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def compute_knee_distances(knees):
    """
    Calcula distâncias euclidianas entre todos os pares de knee points
    """

    # extrair só os pontos
    points = np.array([k["knee"] for k in knees], dtype=float)
    points = normalize(points)
    distances = []

    for (i, p1), (j, p2) in combinations(enumerate(points), 2):

        dist = np.linalg.norm(p1 - p2)

        distances.append({
            "pair": (i, j),
            "w_fp_pair": (knees[i]["w_fp"], knees[j]["w_fp"]),
            "distance": dist
        })

    return distances


def find_knee(pareto_points, cost):
    acc = pareto_points[:, 0]

    # normalizar
    acc_n = normalize(acc)
    cost_n = normalize(cost)

    dist = np.sqrt((1 - acc_n)**2 + (cost_n)**2)

    idx = np.argmin(dist)

    return pareto_points[idx], idx


def weight_stability_analysis(pareto_points):

    weights = [1.0, 1.5, 2.0, 2.5, 3.0]

    knees = []

    for w in weights:
        cost = compute_cost(pareto_points, w_fp=w)
        knee, idx = find_knee(pareto_points, cost)

        knees.append({
            "w_fp": w,
            "knee": knee,
            "idx": idx
        })

    return knees


def plot_knee(pareto_points, knee_point):
    acc = pareto_points[:, 0]
    cost = compute_cost(pareto_points)

    plt.figure(figsize=(6,5))

    plt.scatter(cost, acc, alpha=0.5, label="Pareto front")

    plt.scatter(
        cost[idx],
        knee_point[0],
        color="green",
        s=150,
        marker="X",
        label="Knee point"
    )

    plt.xlabel("Cost (FP + Delay + Duration)")
    plt.ylabel("Accuracy")
    plt.title("Knee Point Selection")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()





def plot_knee_variation(knees):

    weights = [k["w_fp"] for k in knees]
    accs = [k["knee"][0] for k in knees]
    fps = [k["knee"][1] for k in knees]

    plt.figure(figsize=(6,4))

    plt.plot(weights, accs, marker='o', label="Accuracy")
    plt.plot(weights, fps, marker='o', label="FP")

    plt.xlabel("Weight of FP")
    plt.ylabel("Metric value")
    plt.title("Knee point stability vs FP weight")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def extract_trials(study):
    all_points = []
    pareto_points = []

    for t in study.trials:
        if t.values is not None:
            all_points.append(t.values)

    for t in study.best_trials:
        pareto_points.append(t.values)

    all_points = np.array(all_points)
    pareto_points = np.array(pareto_points)

    return all_points, pareto_points


def plot_pareto_grid(all_points, pareto_points, selected_point=None):

    acc = all_points[:, 0]
    fp = all_points[:, 1]
    delay = all_points[:, 2]
    duration = all_points[:, 3]

    acc_p = pareto_points[:, 0]
    fp_p = pareto_points[:, 1]
    delay_p = pareto_points[:, 2]
    duration_p = pareto_points[:, 3]

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    def scatter(ax, x, y, x_p, y_p, xlabel, ylabel):

        ax.scatter(x, y, alpha=0.25, s=20, label="All trials")
        ax.scatter(x_p, y_p, s=40, label="Pareto front")

        idx = np.argsort(x_p)
        ax.plot(x_p[idx], y_p[idx])

        # highlight selected
        if selected_point is not None:
            ax.scatter(
                selected_point[x_idx(xlabel)],
                selected_point[y_idx(ylabel)],
                color="green",
                s=120,
                marker="X",
                label="Selected"
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    # -------------------------
    # helper mapping
    def x_idx(name):
        return {
            "False Positives": 1,
            "Mean Delay": 2,
            "Mean Duration": 3,
            "Accuracy": 0
        }[name]

    def y_idx(name):
        return x_idx(name)

    # -------------------------
    # (1) Accuracy vs FP
    scatter(
        axs[0, 0],
        fp, acc,
        fp_p, acc_p,
        "False Positives", "Accuracy"
    )

    # (2) Accuracy vs Delay
    scatter(
        axs[0, 1],
        delay, acc,
        delay_p, acc_p,
        "Mean Delay", "Accuracy"
    )

    # -------------------------
    # (3) Accuracy vs Duration  ← ALTERADO
    scatter(
        axs[1, 0],
        duration, acc,
        duration_p, acc_p,
        "Mean Duration", "Accuracy"
    )

    # -------------------------
    # (4) FP vs Duration
    scatter(
        axs[1, 1],
        duration, fp,
        duration_p, fp_p,
        "Mean Duration", "False Positives"
    )

    # legenda global
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("pareto_grid_updated.pdf")
    plt.show()


def objective(trial):

    cnn_weight = trial.suggest_float("cnn_weight", 0.1, 0.9)

    params = {
        "shake_threshold": trial.suggest_float("shake_threshold", 0.1, 0.8),
        "entropy_threshold": trial.suggest_float("entropy_threshold", 0.1, 1.2),
        "cnn_weight": cnn_weight,
        "transformer_weight": 1 - cnn_weight,
        "min_steady_timesteps": trial.suggest_int("min_steady_timesteps", 3, 20)
    }

    results = evaluate_model(params)

    return (
        results["accuracy"],              # maximize
        results["fp_transitions"],        # minimize
        results["mean_delay"],            # minimize
        results["mean_duration"]          # minimize
    )

if __name__ == "__main__":
    sampler = optuna.samplers.NSGAIISampler(population_size=10)
    study = optuna.create_study(directions=["maximize", "minimize", "minimize", "minimize"], sampler=sampler)
    study.optimize(objective, n_trials=30)
    pareto_trials = study.best_trials

    # all_points = np.array([t.values for t in study.trials if t.values is not None])
    # pareto_points = np.array([t.values for t in pareto_trials])
    #
    # knees = weight_stability_analysis(pareto_points)
    # plot_final_figure(all_points, pareto_points, knees)
    #
    # accs = [k["knee"][0] for k in knees]
    # fps = [k["knee"][1] for k in knees]
    # delays = [k["knee"][2] for k in knees]
    # durations = [k["knee"][3] for k in knees]
    #
    # print("Accuracy variation:", max(accs) - min(accs))
    # print("FP variation:", max(fps) - min(fps))
    # print("Delay variation:", max(delays) - min(delays))
    # print("Duration variation:", max(durations) - min(durations))
    #
    # plot_knee_variation(knees)
    # distances =compute_knee_distances(knees)
    # dists = np.array([d["distance"] for d in distances])
    #
    # print("\nKnee distance statistics:")
    # print(f"Mean distance: {dists.mean():.4f}")
    # print(f"Max distance:  {dists.max():.4f}")
    # print(f"Min distance:  {dists.min():.4f}")
    # print(f"Std dev:       {dists.std():.4f}")
    #
    # print("\nPairwise distances between knees:")
    # for d in distances:
    #     print(f"weights {d['w_fp_pair']} → distance = {d['distance']:.4f}")
    # exit(0)
    pareto_sorted = sorted(pareto_trials, key=lambda t: t.values[0], reverse=True)
    top_k = pareto_sorted[:20]

    t = PrettyTable(
        ['Person', 'Generation', 'Accuracy', 'False Positives', 'Mean Delay', 'Mean Duration', 'Shake Threshold',
         'Entropy Threshold', "CNN weight", "Transformer weight", "Min Steady Timesteps"])

    for bt in top_k:
        t.add_row([bt.number, bt.system_attrs.get("NSGAIISampler:generation"), round(bt.values[0],4), round(bt.values[1],4), round(bt.values[2],4),
                   round(bt.values[3],4), round(bt.params["shake_threshold"],4), round(bt.params["entropy_threshold"],4), round(bt.params["cnn_weight"],4),
                   round(1 - bt.params["cnn_weight"],4), bt.params["min_steady_timesteps"]])

        #
        # print("\nbest_trial\n")
        # print(best_trial)
        #
        # print("\n")
        # optimized_parameters = PrettyTable(
        #     ['Shake Threshold', 'Entropy Threshold', 'CNN eight', 'Transformer weight', 'Min Steady Timesteps'])
        #
        # optimized_parameters.add_row(
        #     [])
        # print(optimized_parameters)
        #
        # print("\n")
        # optimized_metrics = PrettyTable(
        #     ['Accuracy', 'False Positives', 'Mean Delay', 'Mean Duration'])
        # optimized_metrics.add_row(
        #     [])
        # print(optimized_metrics)

    print("\n")

    print(
        "+---------------------+---------------------------------------------------------+----------------------------------------------------------------------------------------------+")
    print(
        "|     Population      |                         Metrics                         |                                          Parameters                                          |")
    print(t)
    print("\n")
    best_trial = min(top_k, key=lambda t: t.values[1])

    all_points, pareto_points = extract_trials(study)

    best = best_trial.values

    plot_pareto_grid(all_points, pareto_points, selected_point=best)


