import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def get_result(lines):
    count = 0
    want = ""
    for i in lines:
        if "strict" in i or "nodup_list_all" in i:
            count += 1
        if count == 10:
            want = want + i
        if count == 11:
            break
    want = want.replace("\n", " ").replace("\t", " ").replace("[", "").replace("]", "")
    want = [float(i) for i in want.split() if i != "" and i != "strict"]
    return want


def get(path, save_name="a.png"):
    dirs = [i for i in os.listdir(path) if "checkpoint_" in i and ".pt" not in i]
    idx = list(range(1, 11))
    x = {}
    for i in dirs:
        iters = i.split("_")[-1]
        if iters == "best" or iters == "last":
            continue
        path_new = os.path.join(path, i, "score")
        if not os.path.exists(path_new):
            path_new = os.path.join(path, i, "score.txt")

        if os.path.exists(path_new):
            with open(path_new, "r") as f:
                lines = f.readlines()
                x[int(iters)] = get_result(lines)
    x = sorted(x.items(), key = lambda kv:(kv[1], kv[0]))
    plt.title("retro")
    [print(i) for i in x]
    all = [i[1] for i in x]
    x = [i[0] for i in x]
    for i in range(0, len(x)):
        if x[i] % 10000 == 0:
            plt.plot(idx, all[i], label=str(x[i]))
    plt.xlabel("top K")
    plt.ylim(0, 1)
    plt.ylabel("hit percentage")
    plt.grid()
    plt.yticks(np.arange(0, 1.0, 0.05))
    plt.legend()
    plt.savefig(save_name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise "ERROR"
    path = sys.argv[1]
    get(path, save_name=path + "/a.png")
