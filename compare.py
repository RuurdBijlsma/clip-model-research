from statistics import mean, median

rust_embeddings = [0.011462627, -0.009205772, 0.018827433, -0.017202549, 0.008893525]

py_embeddings = [0.011323810555040836, -0.009267876856029034, 0.01880050078034401, -0.017182862386107445, 0.009121607057750225]



diffs = []
for x in range(len(py_embeddings)):
    rust_v = rust_embeddings[x]
    py_v = py_embeddings[x]
    diff = abs(rust_v - py_v)
    diffs.append(diff)

print("Diffs", diffs)
print("Max diff found", max(diffs))
print("Mean difference", mean(diffs))
print("Median difference", median(diffs))
