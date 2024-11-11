import pandas as pd

# Read the CSV file
df = pd.read_csv('/Users/gongjingyi/Desktop/Later/Northeastern Image Data.csv')

# Find the midpoint
midpoint = len(df) // 2

# Split the dataframe into two
df1 = df.iloc[:midpoint]
df2 = df.iloc[midpoint:]
df1.drop_duplicates(inplace=True)
df2.drop_duplicates(inplace=True)
# Save each part to a new CSV file
df1.to_csv('data/Northeastern_Image_Data_1_copy.csv', index=False)
df2.to_csv('data/Northeastern_Image_Data_2_copy.csv', index=False)


# # read in data
# df = pd.read_csv('data/Northeastern_Image_Data_2.csv')
# print(df.shape)

# # drop duplicates
# df_no_duplicates = df.drop_duplicates()
# print(df_no_duplicates.shape)
