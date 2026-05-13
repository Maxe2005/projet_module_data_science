from matplotlib import pyplot as plt


def histogram(data, column):
    """Génère un histogramme pour une colonne spécifique du DataFrame."""
    plt.hist(data[column], bins=20, color="blue", edgecolor="black")
    plt.title(f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


def overview_missing_data(data, print_each_line=False):
    if data is not None:
        missing_values = data.isna()
        nb_missing_values = missing_values.sum().sum()
        print(f"Number of missing values: {nb_missing_values}")
        nb_lines_with_missing_values = missing_values.any(axis=1).sum()
        print(f"Number of lines with missing values: {nb_lines_with_missing_values}")
        nb_missing_values_per_column = missing_values.sum()
        print("\nNumber of missing values per column:")
        print(nb_missing_values_per_column)

        lines_with_more_than_one_missing_value = missing_values.sum(axis=1) > 1
        print("\n")
        if print_each_line:
            print("Lines with more than one missing value:")
        count = 0
        for index, row in missing_values.iterrows():
            if lines_with_more_than_one_missing_value.at[index]:
                if print_each_line:
                    missing_columns = row[row].index.tolist()
                    print(
                        f"Line {index} has missing values in columns: {missing_columns}"
                    )
                count += 1
        print(f"Number of lines with more than one missing value: {count}")
    else:
        print("Data is None, cannot overview missing data.")
