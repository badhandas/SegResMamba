import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.layers import Norm
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
from monai.utils import set_determinism
import torch
from dataset_brats import get_loader
import numpy as np

print_config()


#from model.segmamba import SegMamba
from model.segresmamba import SegResMamba
from monai.networks.nets import UNet, SwinUNETR, UNETR,DynUNet
import psutil

directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)
set_determinism(seed=0)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)



dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

train_loader, val_loader, train_ds_len, val_ds_len = get_loader()

max_epochs = 200
val_interval = 1
VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")



model = SegResMamba(in_chans=4,
                 out_chans=3,
                 depths=[2,2,2,2],
                 feat_size=[48, 96, 192, 384]).to(device)

# model = SwinUNETR(
#     img_size=(128,128,128),
#     in_channels=4,
#     out_channels=3,
#     feature_size=48,
#     drop_rate=0.0,
#     attn_drop_rate=0.0,
#     dropout_path_rate=0.0,
#     use_checkpoint=True,
# ).to(device)

# model = UNETR(
#     in_channels=4,
#     out_channels=3,
#     img_size=(128, 128, 128),
#     feature_size=16,
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=12,
#     res_block=True,
#     dropout_rate=0.0,
# ).to(device)

# print(out.shape)

#model = model.to(device)


num_params = sum(p.numel() for p in model.parameters())

# Print the number of parameters in millions
print(f"Number of parameters: {num_params / 1e6:.2f} million")


loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


# define inference method
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(128, 128, 128),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)


# use amp to accelerate training
scaler = torch.cuda.amp.GradScaler()
# enable cuDNN benchmark
torch.backends.cudnn.benchmark = True

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []
mean_dice_max = 0.0

total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(
            f"{step}/{train_ds_len // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}"
            f", step time: {(time.time() - step_start):.4f}"
        )
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:

        val_start = time.time()
        model.eval()
        dice_tc_list = []
        dice_et_list = []
        dice_wt_list = []
        run_acc = AverageMeter()
        v_step = 0
        all_dice = []
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                
                dice_acc.reset()
                dice_acc(y_pred = val_outputs, y = val_labels )
                acc, not_nans = dice_acc.aggregate()
                run_acc.update(acc.cpu().numpy(), n = not_nans.cpu().numpy())
                dice_tc = run_acc.avg[0]
                dice_wt = run_acc.avg[1]
                dice_et = run_acc.avg[2]
                #val_avg_acc = np.mean(run_acc)
                print(
                    "Val {}/{} {}/{}".format(epoch, max_epochs, v_step, len(val_loader)),
                    ", dice_tc:",
                    dice_tc,
                    ", dice_wt:",
                    dice_wt,
                    ", dice_et:",
                    dice_et,
                )
                v_step += 1
                dice_tc_list.append(dice_tc)
                dice_wt_list.append(dice_wt)
                dice_et_list.append(dice_et)
            mean_dice = (dice_tc_list[-1]+dice_wt_list[-1]+dice_et_list[-1])/3
            val_time = time.time() - val_start
            print(f'CPU usage: {psutil.cpu_percent()}%')
            print(f'Memory usage: {psutil.virtual_memory().percent}%')
            print('Epoch: ' + str(epoch + 1), 'Mean Dice Score: ' + str(mean_dice)+'inference time'+str(val_time))
            
            if mean_dice > mean_dice_max:
                mean_dice_max = mean_dice
                torch.save(model.state_dict(), 'saved_model/' + 'epoch_' + str(epoch) + '_dice_' + str(mean_dice_max) + '.pth')
                print('Checkpoint_' + 'epoch_' + str(epoch) + '_dice_' + str(mean_dice_max) + '.pth' + ' saved')

    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
