from statistics import mean, median
import matplotlib.pyplot as plt

rust_row_1 = [0.1686275, 0.19215691, 0.2313726, 0.30980396, 0.3803922, 0.4901961, 0.56078434, 0.5058824, 0.45098042, 0.4666667, 0.4039216, 0.5294118, 0.5058824, 0.47450984, 0.43529415, 0.427451, 0.427451, 0.4431373, 0.41960788, 0.43529415, 0.43529415, 0.37254906, 0.3411765, 0.27058828, 0.15294123, 0.11372554, 0.17647064, 0.22352946, 0.19215691, 0.254902]


python_row_1 = [0.16862750053405762, 0.19215691089630127, 0.22352945804595947, 0.30980396270751953, 0.3803921937942505, 0.4901961088180542, 0.5607843399047852, 0.5058823823928833, 0.45098042488098145, 0.458823561668396, 0.3960784673690796, 0.529411792755127, 0.5058823823928833, 0.4745098352432251, 0.4274510145187378, 0.41960787773132324, 0.4274510145187378, 0.4431372880935669, 0.41960787773132324, 0.43529415130615234, 0.43529415130615234, 0.37254905700683594, 0.34117650985717773, 0.2705882787704468, 0.14509809017181396, 0.11372554302215576, 0.17647063732147217, 0.22352945804595947, 0.18431377410888672, 0.2549020051956177]


rust_row_1 = [x*255 for x in rust_row_1]
python_row_1 = [x*255 for x in python_row_1]

print(len(rust_row_1))
print(len(python_row_1))

diffs = []
for x in range(len(rust_row_1)):
    rust_v = rust_row_1[x]
    py_v = python_row_1[x]
    diff = rust_v - py_v
    diffs.append(diff)

print("Diffs", diffs)
print("Max diff found", max(diffs))
print("Mean difference", mean(diffs))
print("Median difference", median(diffs))

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot([x for x in rust_row_1], label="Rust % 1", marker='o')
plt.plot([x for x in python_row_1], label="Python % 1", marker='x')
plt.title("Rust vs Python Values")
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(diffs, label="Differences", color='red', marker='.')
plt.title("Differences")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()