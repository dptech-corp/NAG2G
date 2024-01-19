from utils.G2G_cal import *

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise "ERROR"
    if ".txt" in sys.argv[1]:
        smi_path = sys.argv[1]
        N_beam_search = 10
        if len(sys.argv) >= 3:
            N_beam_search = int(sys.argv[2])
        if "--if_full" in sys.argv:
            if_full = True
        else:
            if_full = False
        if len(sys.argv) >= 5:
            score_name = sys.argv[4]
            save_path = smi_path.replace(smi_path.split("/")[-1], score_name)
        else:
            # score_name = "score"
            score_name = smi_path.split("/")[-1].replace("smi", "score")
            save_path = smi_path.replace(smi_path.split("/")[-1], score_name)
        # save_path = None
        run(smi_path, save_path, N_beam_search=N_beam_search, if_full=if_full)
    else:
        path = sys.argv[1]
        # txt_name = "smi_lp0.0_t0_10_b256.txt"
        dirs = [
            os.path.join(path, i)
            for i in os.listdir(path)
            if "checkpoint_" in i and ".pt" not in i
        ]
        for i in dirs:
            dirs2 = [os.path.join(i, j) for j in os.listdir(i) if ".txt" in j and "smi" in j]
            for j in dirs2:
                orders = "python G2G_cal.py {} &".format(j)
                print(orders)
                os.system(orders)