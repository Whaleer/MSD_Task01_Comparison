import os
import torch
from tqdm import tqdm
import json
import nibabel as nib
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from medpy.metric.binary import hd95
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
from monai.apps import DecathlonDataset
import numpy as np
from monai.metrics import DiceMetric
from monai.handlers.utils import from_engine

# Reuse the same transforms and dataset setup from training
root_dir = "/root/autodl-tmp"
save_dir = "/root/lbx/Task01_BrainTumor_comparison/unet/test_outputs"
os.makedirs(save_dir, exist_ok=True)

# Reuse the validation transform for testing
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d


val_org_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

# 在 DiceMetric 定义等指标的下面添加 HD95 计算函数
def compute_hd95(gt, pred, voxelspacing=(1.0, 1.0, 1.0)):
    """
    计算 Hausdorff Distance 的 95th percentile (HD95)。
    如果分割和真实标签均为空，则返回 0.0；
    如果其中一个为空，则返回 np.inf。
    """
    gt = gt.astype(np.bool_)
    pred = pred.astype(np.bool_)
    if np.sum(gt) == 0 and np.sum(pred) == 0:
        return 0.0
    if np.sum(gt) == 0 or np.sum(pred) == 0:
        return np.inf
    return hd95(pred, gt, voxelspacing=voxelspacing)



val_org_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_org_transforms,
    section="validation",
    download=False,
    num_workers=4, 
    cache_num=0,
)
val_org_loader = DataLoader(val_org_ds, batch_size=1, shuffle=False, num_workers=4)

post_transforms = Compose(
    [
        Invertd(
            keys="pred",
            transform=val_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cpu",
        ),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold=0.5),
    ]
)


dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

# 在开始 for 循环之前初始化存储各病例 HD95 的列表
hd95_tc_list = []
hd95_wt_list = []
hd95_et_list = []

# define inference method
VAL_AMP = True
def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3,
    in_channels=4,
    out_channels=3,
    channels=[16,32,64,128],
    strides=[2,2,2]
).to(device)


try:
    model_path = os.path.join(save_dir, "best_metric_model.pth")
    model.load_state_dict(torch.load(model_path))
    print(f"Successfully loaded model from {model_path}")
except Exception as e:
    print(f"Error loading model from {model_path}: {str(e)}")
    raise  # 重新抛出
model.eval()

print(len(val_org_loader))
with torch.no_grad():
    for val_data in tqdm(val_org_loader, total=len(val_org_loader), desc="Processing cases"):
        
        img_name = val_data["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
        if img_name == "BRATS_226.nii.gz" or img_name == "BRATS_030.nii.gz":
            continue
        print(f"Processing case: {img_name}")
        # print(f"val_data type:{type(val_data)}")
        print(val_data.keys()) 
        
        val_inputs = val_data["image"].to(device)
        val_data["pred"] = inference(val_inputs)
        val_data = [post_transforms(i) for i in decollate_batch(val_data)]
        val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
        dice_metric(y_pred=val_outputs, y=val_labels)
        dice_metric_batch(y_pred=val_outputs, y=val_labels)
        val_outputs = torch.stack(val_outputs).squeeze(0)
        val_labels = torch.stack(val_labels).squeeze(0)
        print(f"val_inputs shape:{val_inputs.shape}")    # (1, 4, 240, 240, 155)
        print(f"val_outputs shape:{val_outputs.shape}")  # (3, 240, 240, 155)
        print(f"val_labels shape:{val_labels.shape}")    # (3, 240, 240, 155)

        
        # input_np = val_inputs.cpu().numpy()
        # val_outputs_2 = torch.softmax(val_outputs, 0).cpu().numpy()
        # val_outputs_2 = np.argmax(val_outputs_2, axis=0).astype(np.uint8)
        # val_labels_2 = val_labels.cpu().numpy()[0, :, :, :]
        
        
        input_np = val_inputs.cpu().numpy()
        
        pred_np = torch.sigmoid(val_outputs).cpu().numpy()  # 使用sigmoid而不是softmax
        final_pred = np.zeros(pred_np.shape[1:], dtype=np.uint8)  
        final_pred[(pred_np[0] > 0.5) | (pred_np[1] > 0.5) | (pred_np[2] > 0.5)] = 1  # WT
        final_pred[(pred_np[0] > 0.5) | (pred_np[2] > 0.5)] = 2  # TC
        final_pred[pred_np[2] > 0.5] = 4  # ET
        
        label_np = val_labels.cpu().numpy()  
        final_label = np.zeros(label_np.shape[1:], dtype=np.uint8)
        final_label[(label_np[1] == 1)] = 1  # WT
        final_label[(label_np[0] == 1) | (label_np[2] == 1)] = 2  # TC
        final_label[label_np[2] == 1] = 4  # ET
        
        
        print(f"input_np shape:{input_np.shape}")
        print(f"final_pred shape:{final_pred.shape}")     # (240, 240, 155)
        print(f"final_label shape:{final_label.shape}")   # (240, 240, 155)
        
        
        # 添加 HD95 的计算代码：针对 WT (值1)、TC (值2) 和 ET (值4)
        hd95_wt = compute_hd95(final_label == 1, final_pred == 1)
        hd95_tc = compute_hd95(final_label == 2, final_pred == 2)
        hd95_et = compute_hd95(final_label == 4, final_pred == 4)
        print(f"hd95 for {img_name}: WT: {hd95_wt:.4f}, TC: {hd95_tc:.4f}, ET: {hd95_et:.4f}")
        hd95_wt_list.append(hd95_wt)
        hd95_tc_list.append(hd95_tc)
        hd95_et_list.append(hd95_et)
        
        input_img = nib.Nifti1Image(input_np[0, 2], np.eye(4))
        input_filename = os.path.join(save_dir, f"{img_name}_input.nii.gz")
        nib.save(input_img, input_filename)
        print(f"Saved input image: {input_filename}")
        
        output_img = nib.Nifti1Image(final_pred, np.eye(4))
        output_filename = os.path.join(save_dir, f"{img_name}_output.nii.gz")
        nib.save(output_img, output_filename)
        print(f"Saved output image: {output_filename}")
        
        label_img = nib.Nifti1Image(final_label, np.eye(4))
        label_filename = os.path.join(save_dir, f"{img_name}_label.nii.gz")
        nib.save(label_img, label_filename)
        print(f"Saved label image: {label_filename}")
        
   
    metric_org = dice_metric.aggregate().item()
    metric_batch_org = dice_metric_batch.aggregate()

    dice_metric.reset()
    dice_metric_batch.reset()

metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()
avg_hd95_wt = np.mean(hd95_wt_list)
avg_hd95_tc = np.mean(hd95_tc_list)
avg_hd95_et = np.mean(hd95_et_list)

# 添加平均值计算
avg_dice = (metric_tc + metric_wt + metric_et) / 3
avg_hd95 = (avg_hd95_tc + avg_hd95_wt + avg_hd95_et) / 3

print("Metric on original image spacing: ", metric_org)
print(f"metric_tc: {metric_tc:.4f}")
print(f"metric_wt: {metric_wt:.4f}")
print(f"metric_et: {metric_et:.4f}")
print(f"Average Dice: {avg_dice:.4f}")  # 新增平均Dice
print(f"hd95_tc: {avg_hd95_tc:.4f}")
print(f"hd95_wt: {avg_hd95_wt:.4f}")
print(f"hd95_et: {avg_hd95_et:.4f}")
print(f"Average HD95: {avg_hd95:.4f}")  # 新增平均HD95

# with torch.no_grad():
#     for val_data in tqdm(val_org_loader, total=len(val_org_loader), desc="Processing cases"):
#         val_inputs = val_data["image"].to(device)
#         val_data["pred"] = inference(val_inputs)
#         val_data = [post_transforms(i) for i in decollate_batch(val_data)]
#         val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
#         dice_metric(y_pred=val_outputs, y=val_labels)
#         dice_metric_batch(y_pred=val_outputs, y=val_labels)

#     metric_org = dice_metric.aggregate().item()
#     metric_batch_org = dice_metric_batch.aggregate()

#     dice_metric.reset()
#     dice_metric_batch.reset()

# metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()

# print("Metric on original image spacing: ", metric_org)
# print(f"metric_tc: {metric_tc:.4f}")
# print(f"metric_wt: {metric_wt:.4f}")
# print(f"metric_et: {metric_et:.4f}")






# for single_data in val_data:
#     # 获取病例ID
#     print(f"single_data type:{type(single_data)};single_data len:{len(single_data)}")
#     print("single_data keys:", single_data.keys())
#     try:
#         # 获取并验证文件名
#         img_name = single_data["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
#         filename_or_obj = single_data["image_meta_dict"]["filename_or_obj"]
#         if not filename_or_obj or not isinstance(filename_or_obj[0], str):
#             raise ValueError("Invalid filename format in image_meta_dict")
        
#         # img_name = filename_or_obj[0].split("/")[-1]
#         case_id = os.path.splitext(img_name)[0]
        
#         print(f"Processing case: {img_name}")
#         print(f"Case ID: {case_id}")
        
#     except KeyError:
#         raise RuntimeError("Missing 'filename_or_obj' in image_meta_dict") from None
#     except IndexError:
#         raise RuntimeError("Empty filename_or_obj list in image_meta_dict") from None
#     except Exception as e:
#         raise RuntimeError(f"Error parsing filename: {str(e)}") from e
    
#     # 获取数据和标签
#     val_outputs = single_data["pred"].cpu().numpy()  # (3, H, W, D)
#     val_labels = single_data["label"].cpu().numpy()   # (3, H, W, D)
#     val_input = single_data["image"].cpu().numpy()   # (4, H, W, D)
    
#     # 转换预测结果为BraTS格式
#     final_pred = np.zeros(val_outputs.shape[1:], dtype=np.uint8)
#     final_pred[val_outputs[0] == 1] = 2  # TC
#     final_pred[val_outputs[1] == 1] = 1  # WT
#     final_pred[val_outputs[2] == 1] = 4  # ET
    
#     # 转换标签为BraTS格式
#     final_label = np.zeros(val_labels.shape[1:], dtype=np.uint8)
#     final_label[val_labels[0] == 1] = 2  # TC
#     final_label[val_labels[1] == 1] = 1  # WT
#     final_label[val_labels[2] == 1] = 4  # ET

#     # 保存为nii.gz文件
#     affine = single_data["image_meta_dict"]["affine"]
#     nib.save(nib.Nifti1Image(val_input.transpose(1,2,3,0), np.eye(4)), 
#             os.path.join(save_dir, f"{img_name}_input.nii.gz"))
#     nib.save(nib.Nifti1Image(final_pred, np.eye(4)), 
#             os.path.join(save_dir, f"{img_name}_pred.nii.gz"))
#     nib.save(nib.Nifti1Image(final_label, np.eye(4)), 
#             os.path.join(save_dir, f"{img_name}_label.nii.gz"))
    
#     # 计算当前样本的Dice
#     dice_metric.reset()
#     dice_metric(y_pred=torch.tensor(val_outputs[np.newaxis]), 
#                 y=torch.tensor(val_labels[np.newaxis]))
#     sample_metric = dice_metric.aggregate().item()
    
#     dice_metric_batch.reset()
#     dice_metric_batch(y_pred=torch.tensor(val_outputs[np.newaxis]),
#                     y=torch.tensor(val_labels[np.newaxis]))
#     sample_batch = dice_metric_batch.aggregate()

#     print(f"Case: {img_name}")
#     print(f"  TC: {sample_batch[0].item():.4f}")
#     print(f"  WT: {sample_batch[1].item():.4f}") 
#     print(f"  ET: {sample_batch[2].item():.4f}")
#     print(f"  Average Dice: {sample_metric:.4f}\n")
