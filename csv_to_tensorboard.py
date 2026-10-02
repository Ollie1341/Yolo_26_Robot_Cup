import pandas as pd
from torch.utils.tensorboard import SummaryWriter

CSV = "runs/detect/train-3/results.csv"
LOG_DIR = "runs/tensorboard/train-3"

df = pd.read_csv(CSV)

writer = SummaryWriter(LOG_DIR)

for _, row in df.iterrows():
    epoch = int(row["epoch"])

    metrics = {
        "Loss/Train Box": row["train/box_loss"],
        "Loss/Train Classification": row["train/cls_loss"],
        "Loss/Train L1": row["train/l1_loss"],
        "Loss/Validation Box": row["val/box_loss"],
        "Loss/Validation Classification": row["val/cls_loss"],
        "Loss/Validation L1": row["val/l1_loss"],
        "Metrics/Precision": row["metrics/precision(B)"],
        "Metrics/Recall": row["metrics/recall(B)"],
        "Metrics/mAP50": row["metrics/mAP50(B)"],
        "Metrics/mAP50-95": row["metrics/mAP50-95(B)"],
        "Learning Rate/pg0": row["lr/pg0"],
        "Learning Rate/pg1": row["lr/pg1"],
        "Learning Rate/pg2": row["lr/pg2"],
    }

    for name, value in metrics.items():
        writer.add_scalar(name, value, epoch)

writer.close()

print(f"TensorBoard logs created at {LOG_DIR}")

# source .venv/bin/activate                            zsh   100  21:11:59 
# tensorboard --logdir runs/tensorboard