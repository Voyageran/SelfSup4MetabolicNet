import os
import pydicom
import numpy as np
from datetime import datetime

BASE_DIR = r"D:\DATA\szwj_1185"
SKIP = {...}  # 保留之前的跳过列表

def parse_time(tm_str):
    fmt = "%H%M%S.%f" if "." in tm_str else "%H%M%S"
    return datetime.strptime(tm_str, fmt)

uptake = []

for series in os.listdir(BASE_DIR):
    if series in SKIP: continue
    series_path = os.path.join(BASE_DIR, series)
    if not os.path.isdir(series_path): continue

    # 找到第一个 DICOM
    dcm_file = None
    for root, _, files in os.walk(series_path):
        for f in files:
            if f.lower().endswith(".dcm"):
                dcm_file = os.path.join(root, f)
                break
        if dcm_file: break
    if not dcm_file:
        continue

    ds = pydicom.dcmread(dcm_file, stop_before_pixels=True)

    # 1) 尝试从 Radiopharmaceutical Information Sequence 里拿 RadiopharmaceuticalStartTime
    start_val = None
    seq = ds.get("RadiopharmaceuticalInformationSequence", None)
    if seq and len(seq) > 0:
        start_val = seq[0].get("RadiopharmaceuticalStartTime", None)

    # 2) 如果 1) 失败，再退回去读顶层 (0018,1072)
    if not start_val:
        start_val = ds.get((0x0018, 0x1072), None)

    # 3) 采集时间直接读 AcquisitionTime 顶层
    acq_val = ds.get("AcquisitionTime", None) or ds.get((0x0008, 0x0032), None)

    if not start_val or not acq_val:
        # 跳过无效
        continue

    # 如果值是 bytes，先 decode
    start_str = start_val.decode() if isinstance(start_val, bytes) else str(start_val)
    acq_str   = acq_val.decode()   if isinstance(acq_val, bytes)   else str(acq_val)

    # 解析并计算 Uptake Time（分钟）
    try:
        t0 = parse_time(start_str)
        t1 = parse_time(acq_str)
    except:
        continue

    uptake.append((t1 - t0).total_seconds() / 60.0)

# 输出统计
if uptake:
    arr = np.array(uptake)
    print(f"共处理 {len(arr)} 个序列")
    print(f"平均 Uptake Time = {arr.mean():.1f} 分钟 ± {arr.std(ddof=1):.1f} 分钟")
else:
    print("无有效 Uptake Time 数据可统计")
