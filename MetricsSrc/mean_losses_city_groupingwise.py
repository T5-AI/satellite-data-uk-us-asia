import pandas as pd
data = pd.read_csv("./urban_data/tencities/GenAI_density/output_results.csv")
data.head()
data_noNA = data.dropna().copy()
cities = [
    'Singapore',
    'HongKong',
    'Munich',
    'Stockholm',
    'Chicago',
    'Orlando',
    'Kinshasa',
    'SaoPaulo',
    'Mexico',
    'Kigali'
]

# Ensure data_noNA is a full copy before modifying it
data_noNA = data_noNA.copy()

# Compute mean across the 3 sub-columns for each city
for city in cities:
    col_0 = f"{city}____0"
    col_1 = f"{city}____1"
    col_2 = f"{city}____2"

    # Assign the mean while using .loc to avoid SettingWithCopyWarning
    data_noNA.loc[:, f"{city}_mean"] = data_noNA[[col_0, col_1, col_2]].mean(axis=1)

data_noNA.to_csv("./urban_data/tencities/GenAI_density/all_col_losses114.csv", index=False)

mean_cols = [f"{city}_mean" for city in cities]
df_final = data_noNA[['GroupBase', 'OriginalFilename'] + mean_cols]

# 3) Save results
df_final.to_csv("./urban_data/tencities/GenAI_density/mean_losses.csv", index=False)