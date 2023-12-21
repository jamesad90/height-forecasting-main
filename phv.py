import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('Berkeley_EB_4.csv')
def plot_median_iqr_phv(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    # Calculate yearly height growth
    data['height_growth'] = data.groupby('child_id')['height_cm'].diff()

    # Determine age at PHV for each child
    phv_age = data.loc[data.groupby('child_id')['height_growth'].idxmax()]

    # Define criteria for early and late maturing (adjust as needed)
    early_maturing_age = 12
    late_maturing_age = 14

    # Categorize each child
    phv_age['maturation'] = pd.cut(phv_age['age_decimalyears'], 
                                   bins=[0, early_maturing_age, late_maturing_age, float('inf')],
                                   labels=['Early', 'Normal', 'Late'])

    # Merge the maturation category back into the original data
    data = data.merge(phv_age[['child_id', 'maturation']], on='child_id', how='left')

    # Filter for only early and late maturers
    data_filtered = data[data['maturation'].isin(['Early', 'Late'])]

    # Group by age and maturation category to calculate median and IQR
    grouped = data_filtered.groupby(['maturation', 'age_decimalyears'])
    median = grouped['height_growth'].median()
    iqr = grouped['height_growth'].quantile(0.75) - grouped['height_growth'].quantile(0.25)

    fig, ax = plt.subplots(figsize=(10, 6))
    for maturation in ['Early', 'Late']:
        ax.plot(median.xs(maturation).index, median.xs(maturation).values, label=f'{maturation} Median')
        ax.fill_between(median.xs(maturation).index, 
                        (median - iqr).xs(maturation).values, 
                        (median + iqr).xs(maturation).values, 
                        alpha=0.2, label=f'{maturation} IQR')

    ax.set_title('Median and IQR of PHV Curves for Early and Late Maturers')
    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Height Growth (cm/year)')
    ax.legend()
    plt.show()
    return data_filtered
# Example usage
plot_median_iqr_phv(data)


def plot_phv_curves_facet(data):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    # Calculate yearly height growth
    data['height_growth'] = data.groupby('child_id')['height_cm'].diff()

    # Determine age at PHV for each child
    phv_age = data.loc[data.groupby('child_id')['height_growth'].idxmax()]

    # Define criteria for early and late maturing (adjust as needed)
    early_maturing_age = 12
    late_maturing_age = 14

    # Categorize each child
    phv_age['maturation'] = pd.cut(phv_age['age_decimalyears'], 
                                   bins=[0, early_maturing_age, late_maturing_age, float('inf')],
                                   labels=['Early', 'Normal', 'Late'])

    # Merge the maturation category back into the original data
    data = data.merge(phv_age[['child_id', 'maturation']], on='child_id', how='left')

    # Filter for only early and late maturers
    data_filtered = data[data['maturation'].isin(['Early', 'Normal', 'Late'])]

    # Create facet grid
    g = sns.FacetGrid(data_filtered, col="maturation", hue="child_id", col_wrap=4, height=4)
    g = g.map(sns.lineplot, "age_decimalyears", "height_growth", marker="o")

    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('PHV Curves for Early and Late Maturers')
    g.add_legend()
    g.set_axis_labels("Age (years)", "Height Growth (cm/year)")

    plt.show()

# Example usage
plot_phv_curves_facet(data)

def categorize_maturation(test_data):
    # Ensure the data is a DataFrame
    if not isinstance(test_data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame")

    # Calculate yearly height growth
    test_data['height_growth'] = test_data.groupby('child_id')['height_cm'].diff()

    # Determine age at PHV for each child
    phv_age = test_data.loc[test_data.groupby('child_id')['height_growth'].idxmax()]

    # Define criteria for early and late maturing (adjust as needed)
    early_maturing_age = 12
    late_maturing_age = 14

    # Categorize each child
    phv_age['maturation'] = pd.cut(phv_age['age_decimalyears'], 
                                   bins=[0, early_maturing_age, late_maturing_age, float('inf')],
                                   labels=['Early', 'Normal', 'Late'])

    # Merge the maturation category back into the original data
    result = test_data.merge(phv_age[['child_id', 'maturation']], on='child_id', how='left')

    return result

categorized_data = categorize_maturation(data)
print(categorized_data)