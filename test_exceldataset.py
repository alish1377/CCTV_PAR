import pandas as pd

df = pd.read_excel("data/NATIVE/Native_dataset.xlsx")
# Your target column list
target_columns = [
    'Age_young', 'Age_adult', 'Age_old', 'Gender_female', 'Hair_short', 'Hair_long', 'Hair_bald',
    'Upper_short', 'U_black', 'U_blue', 'U_brown', 'U_green', 'U_grey', 'U_orange', 'U_pink',
    'U_purple', 'U_red', 'U_white', 'U_yellow', '////U_Others////', 'L_black', 'L_blue', 'L_brown',
    'L_green', 'L_grey', 'L_orange', 'L_pink', 'L_purple', 'L_red', 'L_white', 'L_yellow',
    '////L_Others////', 'Backpack', 'Bag', 'Glasses', 'Sunglasses', 'Hat'
]


# Assuming df is your DataFrame
# Step 1: Filter only columns that are both in DataFrame and in target_columns
available_columns = [col for col in target_columns if col in df.columns]

# Step 2: Check how many of these have all values as NaN
all_nan_count = df[available_columns].isna().all().sum()

print(f"Total available columns from list: {len(available_columns)}")
print(f"Columns with all NaN values: {all_nan_count}")