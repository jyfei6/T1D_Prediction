import pandas as pd
import matplotlib.pyplot as plt
'''
This plot is used to plot the final prediction triple bar plots for batch4
'''
def plot_proportional_bars(data_values, labels, sample_name, ax, primary_color='#EDEB7A', secondary_color='#70E0E0'):
    """
    Plot a horizontal bar chart with proportional color fills.

    Parameters:
        data_values (list of floats): The proportions of the primary color for each bar.
        labels (list of str): The labels for each bar.
        primary_color (str): Hex code or color name for the primary fill.
        secondary_color (str): Hex code or color name for the remaining fill.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    # Calculate the remaining proportions
    data_proportions = [(value, 1 - value) for value in data_values]

    # Create a figure
    # fig, ax = plt.subplots(figsize=(8, 4))

    # Plot each bar with two colored segments using the provided colors
    for i, (proportions, label) in enumerate(zip(data_proportions, labels)):
        # Plot the proportion of the data value
        ax.barh(label, proportions[0], color=primary_color, edgecolor='white')
        # Plot the remaining proportion to fill up to 1
        ax.barh(label, proportions[1], left=proportions[0], color=secondary_color, edgecolor='white')

    # Adding text annotations for clarity
    for i, value in enumerate(data_values):
        ax.text(value / 2, i, f'{value:.3f}', va='center', ha='center', color='black', fontsize=10, fontweight='bold')
        ax.text(value + (1 - value) / 2, i, f'{1 - value:.3f}', va='center', ha='center', color='white', fontsize=10, fontweight='bold')

    # Setting titles and labels
    ax.set_title("Probability of being Progressive for" + sample_name)
    ax.set_xlabel('Proportion')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)


# Example usage:
if __name__ == "__main__":
    data_values = [0.2, 0.4, 0.2]
    labels = ['Anova', 'Var_fil', 'Var_unfil']
    fig = plot_proportional_bars(data_values, labels)
    plt.show()