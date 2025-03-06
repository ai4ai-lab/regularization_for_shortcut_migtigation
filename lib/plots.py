import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


### plot the weights assigned to the shortcut variable for each model
### only for synthetic experiment
def plot_weights_for_synthetic_experiment(experiment_model, reg_strength, ax):
    labels = {"without_reg": "No regularization", "l1": "L1", "l2":"L2", "eye":"EYE", "causal":"Causal"}
    color_dict = {"without_reg": "#2878b5", "l1": "#9ac9db", "l2": "#f8ac8c", "eye": "#c82423", "causal": "#999999"}

    bar_width = 0.15
    weight = np.array([4,-0.5,1,2,0]) ### corresponding to the weights set before

    idx = np.arange(0, 5)
    start_idx = idx - bar_width*6/2 + bar_width
    plt.bar(start_idx, weight, bar_width, label='True weights', color="black")
    for i, (name, model) in enumerate(experiment_model.items()):
        para = model.model.weight[0].detach().numpy()
        plt.bar(start_idx + (i+1)*bar_width, para, bar_width, label=labels[name], color=color_dict[name])

    plt.legend(prop = {'family':'times new roman','size':8, 'weight':'bold'})
    plt.xticks(fontsize=10, fontfamily='times new roman', fontweight="bold")
    plt.yticks(fontsize=10, fontfamily='times new roman', fontweight="bold")
    ax.set_xticks(idx + bar_width/2)
    ax.set_xticklabels(["Concept 1", "Concept 2", "Unknown 1", "Unknown 2", "Shortcut"])
    plt.ylabel("Trained weight of the shortcut variable", fontsize=14, fontfamily='times new roman', fontweight="bold")
    plt.title(f"Weight comparison (regularization strength = {reg_strength})", fontsize=13, fontweight='bold', fontfamily="times new roman")
    plt.grid(True)


### plot the treatment effect of the shortcut variable for each regularization method
def plot_treatment_effect(experiment_model):

    model_dict = {"No regularization": experiment_model["without_reg"], 
              "L1": experiment_model["l1"], 
              "L2": experiment_model["l2"], 
              "EYE": experiment_model["eye"], 
              "Causal": experiment_model["causal"]}
    
    color_dict = {"No regularization": "#2878b5", "L1": "#9ac9db", "L2": "#f8ac8c", "EYE": "#c82423", "Causal":"#999999"}

    for key, value in model_dict.items():

        val_te = value.test_treatment_effect
        plt.plot(range(0, len(val_te)), val_te, label=key, color=color_dict[key], lw=2.5)

    plt.grid(True)
    plt.legend(prop = {'family':'times new roman','size':11, 'weight':'bold'})
    plt.xticks(fontsize=12, fontfamily='times new roman', fontweight="bold")
    plt.yticks(fontsize=12, fontfamily='times new roman', fontweight="bold")
    plt.xlabel("Epoch", fontsize=13, fontfamily='times new roman', fontweight="bold")
    plt.ylabel("Estimated shortcut treatment effect", fontsize=11, fontweight='bold', font="times new roman")