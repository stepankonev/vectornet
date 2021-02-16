import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from IDX import IDX
from vn_config import vn_config

def draw(data0, attn, step, pred):
    from l5kit.geometry import transform_points
    
    pred = np.array([transform_points(
        pred[:,i,:,:].cpu().detach().numpy(),
        data0.raster_from_agent.cpu().detach().numpy()
    ) for i in range(3)]).transpose(1,0,2,3)
     
    
    pts = transform_points(
        data0.target_positions.cpu().detach().numpy(),
        data0.raster_from_agent.cpu().detach().numpy()
    )
    plt.figure(figsize = (10,20))
    
    for i, batch in enumerate(range(18)):
        plt.subplot(6,3,i+1)
        A = attn[batch]
        A = (A - A.min()) / (A.max() - A.min()) * 0.9 + 0.1
        df = pd.DataFrame(data0.x[data0.batch == batch].cpu().detach().numpy())
        df["pid"] = data0.i[data0.batch == batch].cpu().detach().numpy()
        for pid, pdf in df.groupby("pid"):
            color = "gray"
            if pdf.iloc[0, IDX["ego"]] == 1:
                color = "green"
            if pdf.iloc[0, IDX["default"]] == 1:
                color = "black"
            if pdf.iloc[0, IDX["yellow"]] == 1:
                color = "yellow"
            if pdf.iloc[0, IDX["red"]] == 1:
                color = "red"
            if pdf.iloc[0, IDX["green"]] == 1:
                color = "green"
            if pdf.iloc[0, IDX["crosswalk"]] == 1:
                color = "orange"
            if pdf.iloc[0, IDX["other"]] == 1:
                color = "blue"
            plt.plot(pdf[0], pdf[1], color = color, alpha = A[pid].detach().item())
        plt.plot(pts[batch,:,0], pts[batch,:,1], linewidth = 3, color = "purple")
        plt.plot(pred[batch,0,:,0], pred[batch,0,:,1], linewidth = 3, alpha = 0.7)
        plt.plot(pred[batch,1,:,0], pred[batch,1,:,1], linewidth = 3, alpha = 0.7)
        plt.plot(pred[batch,2,:,0], pred[batch,2,:,1], linewidth = 3, alpha = 0.7)
        plt.xlim([0, 224])
        plt.ylim([0, 224])
    plt.tight_layout()
    plt.savefig(f"{vn_config['plots_path']}/p_{step}.png")
    plt.close()


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)