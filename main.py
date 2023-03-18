import pandas as pd
import matplotlib as plt
import numpy as np

df = pd.read_csv("abalone.data", header=None)
df.columns = ["sex", "length", "diameter", "height", "whole_height", "shucked_weight",
              "viscera_weight", "shell_weight", "rings"]

# we get acquire the necessary statistics for the z - score normalization
length_mean = df["length"].mean()
length_dev = df["length"].std()
# print out our stats
print(f"Mean: {length_mean}")
print(f"Standard deviation: {length_dev}")

# now we need to add a column with this new z-scred value for length
df["length_z"] = ((df["length"] - length_mean) / length_dev)
print(df)

# lets now test our length_z (we want mean = 0 and std = 1)
print(f"Mean is: {df['length_z'].mean()}")
print(f"Std is: {df['length_z'].std()}")
# our stats match out so we are all g :D


# ------------------- Question 2 -------------------------------

diameter_col = df["diameter"]
print(diameter_col)
print(len(diameter_col))


# -------------------- Question 3 ----------------------

df["sex_one_hot"] = pd.get_dummies(df["sex"]).values.tolist()
print(df)

# -------------------- Question 4 ----------------------

correlation_list = []
for i in range(1, 8):
    correlation_list.append((df.columns[i], df["rings"].corr(df[df.columns[i]])))

correlation_list.sort(key=lambda x: x[1])
correlation_list.reverse()

counter = 0

print(correlation_list[2])
print("correlation ranked - ")
for item in correlation_list:
    print(f"{counter + 1}) {correlation_list[counter][0]} - {correlation_list[counter][1]}")
    counter = counter + 1

# ------------------------------------ Question 5 ----------------------------------

df["ring_shell_corr"] = df["rings"].corr(df["shell_weight"])
df = df.assign(ring_shell_corr3 = lambda x: x.rings.corr(x.shell_weight))
print(df)




