import torch

config = {
  #common
  'output': 'out',
  'resnet_type':'50',#[50,101]
  'rsn_score_threshold':0.65,
  'ocr_out_dim':-1,
  'input_dim': 3,
  'input_max_length':512,
  'data_label_dim':6,
  'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
  'detection_box_scaling':1, #decode
  'label_scaling':2, #encode
  'identification_list':'configs/identification_list.txt',
  'wh_add':1/0.75*1.0,
  'test_model': 'ckpt/ckpt_epoch_final.pth',
}
