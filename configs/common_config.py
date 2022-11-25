import torch

config = {
  #common
  'output': 'out',
  'resnet_type':'50',#[50,101]
  'rsn_score_threshold':0.75,
  'ocr_out_dim':-1,
  'input_dim': 3,
  'input_max_length':512,
  'data_label_dim':6,
  'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
  'detection_box_scaling':1, #decode
  'label_scaling':2, #encode
  'identification_list':'configs/identification_list.txt',

  #ocr-training
  'ocr_train_dataset': ['dataset/ctw1500_ocr/train','dataset/iam_hand/lines','dataset/iam_hand/sentences','dataset/pre_train_single','dataset/ctw1500_ocr/train'],
  #'ocr_train_dataset': ['dataset/pre_train_single'],
  #'ocr_train_dataset': ['dataset/ctw1500_ocr/train'],
  'ocr_train_data_label':['image_label_list.txt','image_label_list.txt','image_label_list.txt','image_label_list.txt','image_label_list.txt','image_label_list.txt','image_label_list.txt'],
  'ocr_train_ckpt_save':'ckpt',

  #train
  'train_dataset': ['dataset/mtwi_train','dataset/scut_ctw1500/train'],
  'train_image':['image_train','image'],
  'train_label':['txt_train','ctw1500_e2e_train_label'],
  'train_label_point_num':[4,14],
  'train_label_point_style':[[0,1],[0,1]], #0:x,1:y
  'train_label_point_order':[{0:0,3:1,2:2,1:3},{0:0,1:6,2:7,3:13}], #[up_start]-0:p,[up_end]-1:p,[bottom_end]-2:p,[bottom_start]-3:p
  'train_max_image_num':[3000,1000],

  'train_ckpt_save':'ckpt',
  'train_ocr_pre_ckpt':'ckpt/ocr_ckpt_epoch_final.pth',
  'train_start_ckpt':'ckpt/ckpt_epoch_.pth',
  'train_start_epoch':0,
  'train_batch_size':4,
  'train_epochs':3,
  'train_learning_rate':1e-4,
  'train_min_learning_rate':1e-7,
  'train_save_epoch':2,

  #ocr-test
  'ocr_test_dataset': ['dataset/ctw1500_ocr/test'],
  #'ocr_test_dataset': ['dataset/pre_train_single'],
  'ocr_test_data_label':['image_label_list.txt'],

  #test
  'test_dataset': ['dataset/scut_ctw1500/test'],
  'test_image': ['image'],
  'test_label': ['ctw1500_e2e_test_label'],
  'test_label_point_num':[14],
  'test_max_image_num':[1000],
  'test_label_point_style':[[0,1]],
  'test_label_point_order':[{0:0,1:6,2:7,3:13}],
  'test_model': 'ckpt/ckpt_epoch_2.pth',
}
