import numpy as np
import pandas as pd
from evaluate_tracker import evaluate_tracker
from calculate_measures import tracking_analysis

class Params():
    def __init__(self):
        pass
def opt(p):
    params = Params()

    params.sigma = p[0]
    params.lmba = p[1]
    params.alfa = p[2]
    params.enlarge_factor = p[3]
    evaluate_tracker("workspace-dir", "MESSO", params)
    out = tracking_analysis("workspace-dir", "MESSO")

    return out['total_failures'], out['average_overlap'], out['average_speed']

#df = pd.DataFrame(columns=["sigma", "lmba", "alfa", "enlarge_factor", "fails", "overlap", "fps"])
df = pd.read_csv("rez.csv")
for lmba in [0.001, 0.01]:
    for sigma in [1.0, 2.0, 4.0, 6.0, 8.0]:
        for alfa in [0.05, 0.1, 0.15, 0.2]:
            for enlarge_factor in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:

                vals = df[["sigma", "lmba", "alfa", "enlarge_factor"]].values
                calc = True
                for v in vals:
                    if ([sigma, lmba, alfa, enlarge_factor] == v).all():
                        calc = False
                if calc:
                    try:
                        fails, overlap, fps = opt([sigma, lmba, alfa, enlarge_factor])
                        df.loc[len(df.index)] = [sigma, lmba, alfa, enlarge_factor, fails, overlap, fps]
                        df.to_csv("rez.csv", index=False)
                    except:
                        print("FAIL")
                else:
                    print("Skip")