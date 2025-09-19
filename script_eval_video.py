import matplotlib.pyplot as plt
import os
import utils

root_path = './'

data_name = 'UCSD_P1_256'
data_path = os.path.join(root_path, 'dataset', data_name)

res_root = os.path.join(root_path, 'video-proj_results')
# Model setting name should include dataset for clarity
if data_name == 'UCSD_P1_256':
    model_setting = 'MemAE_Conv3DSpar_UCSD_P1_256_MemDim2000_EntW0.0002_ShrThres0.0025_Non'
    # model_setting = 'MemAE_Conv3DSpar_UCSD_P1_256_MemDim2000_EntW0.0002_ShrThres0.0025_Seed1_Non
elif data_name == 'UCSD_P2_256':
    model_setting = 'MemAE_Conv3DSpar_UCSD_P2_256_MemDim2000_EntW0.0002_ShrThres0.0025_Non'
elif data_name == 'Avenue':
    model_setting = 'MemAE_Conv3DSpar_Avenue_MemDim2000_EntW0.0002_ShrThres0.0025_Non'
else:
    raise ValueError(f"Unsupported dataset: {data_name}")
# model_setting = 'MemAE_Conv3DSpar_UCSD_P2_256_MemDim2000_EntW0.0002_ShrThres0.0025_Non'
res_path = os.path.join(res_root, 'res_'+model_setting)

auc = utils.eval_video(data_path, res_path, is_show=False)
