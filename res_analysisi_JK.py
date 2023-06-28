import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from distribution_sets import distribution_sets
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler


def read_data_from_files():
    # Initialize a list to store individual data frames.
    dfs = []

    # Get a list of all csv files in the directory
    csv_files = glob.glob('./results/*.csv')

    # Loop over the list of files
    for file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file, skiprows=1, names=["Best Score", "Mean_best" , "Std_best" ,"Mean_f" , "Mean_acc" , "Std_scc" , "Max_acc", "Max Iteration" , "Time"])

        # Remove brackets from "Best Score" column and convert to numeric
        df["Best Score"] = df["Best Score"].str.replace('[', '', regex=True).str.replace(']', '', regex=True)
        df["Best Score"] = pd.to_numeric(df["Best Score"])

        # Split the filename to get the test_Name, swarnDistribution, phipDistribution, phihgDistribution, experimentType
        base = os.path.basename(file)
        name = os.path.splitext(base)[0]
        test_Name, swarmDistribution, phipDistribution, phihgDistribution, experimentType = name.split('_')

        # Add these as columns to the DataFrame
        df['test_Name'] = test_Name
        df['swarmDistribution'] = swarmDistribution
        df['phipDistribution'] = phipDistribution
        df['phihgDistribution'] = phihgDistribution
        df['experimentType'] = experimentType

        # Append this DataFrame to the list
        dfs.append(df)

    # Concatenate all the dataframes in the list.
    df_all = pd.concat(dfs, ignore_index=True)

    # Create new column "max_FE"
    df_all["max_FE"] = df_all["Max Iteration"] * 20

    # Print the final DataFrame
    print(df_all)
    df_all.to_pickle("./df_all.pkl")

def swarm_plots():
    # Get list of unique test_Names
    test_names = df_all['test_Name'].unique()

    # Define the columns and their names
    column = ['Best Score', 'Max_acc', 'max_FE']
    column_name = ['mB', 'mACC', 'mFE']

    # Determine the number of unique swarm distributions
    num_distributions = df_all['swarmDistribution'].nunique()


    # Iterate over each column
    for i_column in range(len(column)):
        # Create a figure with subplots arranged in a 1xN grid
        fig, axes = plt.subplots(1, len(test_names), figsize=(4*len(test_names), 4),  constrained_layout =True)

        # Adjust the margins and the spacing between subplots
        # fig.subplots_adjust(left=0.1, right=None, bottom=0.4, top=None, wspace=0.4, hspace=None)

        # Iterate over each test name
        for i_test, test_name in enumerate(test_names):
            # Filter data for the current test_name
            data = df_all[df_all['test_Name'] == test_name]

            # Create the boxplot on the appropriate subplot
            data.boxplot(column=column[i_column], by='swarmDistribution', ax=axes[i_test], fontsize=14)
            axes[i_test].set_title(f'{test_name}', fontsize=18)
            axes[i_test].set_ylabel(column_name[i_column], fontsize=18)
            axes[i_test].set_xlabel('Distribution', fontsize=18)
            # axes[i_test].set_ylabel(column_name[i_column], fontsize=18)
            # axes[i_test].set_xlabel('Distribution', fontsize=18)

            # Set the xticks to be numbers from 1 to the number of unique distributions
            axes[i_test].set_xticklabels(np.arange(1, num_distributions + 1))
        # Set common labels
        # fig.text(0.5, 0.04, 'Distributions', ha='center', va='center', fontsize=18)
        # fig.text(0.02, 0.5, column_name[i_column], ha='center', va='center', rotation='vertical', fontsize=18)

        fig.suptitle('')  # Suppress the automatic pandas-generated title
        # Save the figure to a PNG file
        fig.savefig(f'swarmDistributions_{column_name[i_column]}.png')


def phi_plots():

    generator_set = distribution_sets()
    phi_dist_all = []
    for phi in generator_set['phi_p']:
        phi_dist_all.append(phi.dist.name)

    # Get list of unique test_Names
    test_names = df_all['test_Name'].unique()

    # Define the columns and their names
    column = ['Best Score', 'Max_acc', 'max_FE']
    column_name = ['mB', 'mACC', 'mFE']


    # Create an empty dictionary to store the statistics
    statistics = {}

    # Get the unique distributions
    # distributions = df_all['phipDistribution'].unique()
    distributions = phi_dist_all
    distributions[distributions.index('weibull_min')] = 'weibull'

    # Iterate over each test name
    for i_test, test_name in enumerate(test_names):
        # Iterate over each measure:
        for i_column, col in enumerate(column):
            # Create a new figure for each test and  measure
            fig, ax = plt.subplots(figsize=(21, 3))
            # Adjust the margins and the spacing between subplots
            fig.subplots_adjust(left=0.1, right=0.9, bottom=0.20, top=0.9)

            boxplot_data = pd.DataFrame(columns=distributions)
            # Iterate over each distribution combination
            for distribution in distributions:
                # Filter data for the current test_name and distribution combination
                data_combined = df_all.loc[(df_all['test_Name'] == test_name) &
                                       ((df_all['phipDistribution'] == distribution) |
                                       (df_all['phihgDistribution'] == distribution))]
                # Append the column data to the boxplot_data list
                # print(distribution)
                boxplot_data[distribution] = data_combined[col].tolist()


            # print(boxplot_data.shape)
            colors = ['maroon', 'teal', 'purple']
            # plot. Set color of marker edge
            flierprops = dict(marker='o', markerfacecolor='whitesmoke', markersize=5,
                              linestyle='none', markeredgecolor='lightgrey')
            ax = boxplot_data.boxplot(fontsize=14, #patch_artist=True, #boxprops=dict(facecolor=colors[i_test]),
                                      showmeans=True,
                                      flierprops=flierprops, #showfliers=False,
                                      color=colors[i_test])
            # ax = boxplot_data.boxplot(fontsize=14)

            # Set the xticks to be numbers from 1 to the number of unique distributions
            ax.set_xticklabels(np.arange(1, len(distributions) + 1))

            # Set the title for the figure
            ax.set_title(f'{test_name}', fontsize=18)
            # Set the common labels
            ax.set_xlabel('Distribution', fontsize=14)
            ax.set_ylabel(column_name[i_column], fontsize=14)
            # Save the figure to a PNG file
            # Set the color of the outlier points to yellow

            fig.savefig(f'jointDistributions_{test_name}_{column_name[i_column]}.png')
            plt.close(fig)  # Close the figure to release memory resources

            # Calculate statistics for the boxplot data
            stats_mean = np.mean(boxplot_data, axis=0)
            stats_std = np.std(boxplot_data, axis=0)

            # Store the statistics in the dictionary
            statistics[(test_name, column_name[i_column])] = {'Mean': stats_mean, 'Std': stats_std}

    # Create a DataFrame from the statistics dictionary
    for test_name in test_names:
        for col in column_name:
            df_statistics = pd.DataFrame(statistics[(test_name, col)])
            # Convert the DataFrame to a LaTeX table
            if col == 'mACC':
                df_sorted = df_statistics.sort_values(by='Mean', ascending=False)
            else:
                df_sorted = df_statistics.sort_values(by='Mean', ascending=True)
            print('#===============')
            print(f"function: {test_name}, measure: {col}")
            pprint(df_sorted.to_dict())

            latex_table = df_sorted.head(5).to_latex()
            # Save the LaTeX table to a .tex file
            with open(f'latex_{test_name}_{col}.tex', 'w') as f:
                f.write(latex_table)

def each_phi_plot():
    generator_set = distribution_sets()
    phi_dist_all = []
    for phi in generator_set['phi_p']:
        phi_dist_all.append(phi.dist.name)

    # Define the columns and their names
    column = ['Best Score', 'Max_acc', 'max_FE']
    column_name = ['mB', 'mACC', 'mFE']
    column_norm = ['Best Score']

    # Create an empty dictionary to store the statistics
    statistics = {}

    # Get the unique distributions
    # distributions = df_all['phipDistribution'].unique()
    distributions = phi_dist_all
    distributions[distributions.index('weibull_min')] = 'weibull'

    # Initialize a MinMaxScaler
    scaler = MinMaxScaler()
    # Apply the normalization to each group
    for col in column_norm:
        df_all[col] = df_all.groupby('test_Name')[col].transform(
            lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).reshape(-1))

    params = ['phipDistribution', 'phihgDistribution']
    # Iterate over each measure:
    for i_column, col in enumerate(column):
        for i_phi, phi in enumerate(params):
            # Create a new figure for each test and  measure
            fig, ax = plt.subplots(figsize=(21, 3))
            # Adjust the margins and the spacing between subplots
            # fig.subplots_adjust(left=0.1, right=0.9, bottom=0.20, top=0.9)

            # Create a boxplot for each distribution
            # sns.boxplot(x='phipDistribution', y=col, data=df_filtered)
            colors = ['sienna', 'green']
            # plot. Set color of marker edge
            flierprops = dict(marker='o', markerfacecolor='gainsboro', markersize=5,
                              linestyle='none', markeredgecolor='silver')
            ax = df_all.boxplot(column=col, by=phi, ax=ax, fontsize=14, #patch_artist=True, #boxprops=dict(facecolor=colors[i_test]),
                                      showmeans=True,
                                      flierprops=flierprops, #showfliers=False,
                                      color=colors[i_phi])
            # ax = boxplot_data.boxplot(fontsize=14)


            if phi == 'phipDistribution':
                name = 'phi_p'
            else:
                name = 'phi_g'

            # Set the xticks to be numbers from 1 to the number of unique distributions
            ax.set_xticklabels(np.arange(1, len(distributions) + 1))
            plt.suptitle('')
            # Set the title for the figure
            ax.set_title(f'{name}', fontsize=14)
            # Set the common labels
            ax.set_xlabel('Distribution', fontsize=14, wrap=True)
            ax.set_ylabel(column_name[i_column], fontsize=14)
            plt.subplots_adjust(left=0.1, right=0.9, bottom=0.20, top=0.9)
            # Save the figure to a PNG file
            fig.savefig(f'{name}_{column_name[i_column]}.png')
            plt.close(fig)  # Close the figure to release memory resources

if __name__ == "__main__":
    # Check if the pickle file exists
    if not os.path.isfile("./df_all.pkl"):
        # If the pickle file does not exist, run the function to read the data from the files
        read_data_from_files()

    # Read the data from the pickle file
    df_all = pd.read_pickle("./df_all.pkl")
    print(df_all.head())
    df_all["max_FE"] = df_all["Max Iteration"] * 20 / 1000 # because of the large range numbers we need to shorten
    # swarm_plots()


    # phi_plots()

    each_phi_plot()





