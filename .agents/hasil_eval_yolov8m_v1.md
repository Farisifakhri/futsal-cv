==================================================
🚀 [Step 1/3] Memulai Preprocessing Dataset Lokal
📌 Source : /content/futsal-cv/data
📌 Target : /content/futsal-cv/data_processed
==================================================
🔄 Processing split 'train': 256 gambar...
🔄 Processing split 'valid': 73 gambar...
🔄 Processing split 'test': 36 gambar...
==================================================
✅ Preprocessing selesai! Total 365 gambar diproses.
==================================================
==================================================
🚀 [Step 2/3] Memulai Data Augmentation Offline Lokal
📌 Source : /content/futsal-cv/data_processed
📌 Target : /content/futsal-cv/data_augmented
==================================================
🔄 Processing & Augmenting split 'train': 256 original images...
🔄 Processing & Augmenting split 'valid': 73 original images...
🔄 Processing & Augmenting split 'test': 36 original images...
==================================================
✅ Data Augmentation selesai! Total 1133 gambar dalam dataset akhir.
==================================================
Config created at: /content/futsal-cv/data_augmented/data.yaml
==================================================
[Step 3/3] Memulai Training Model YOLOV8M (imgsz=1024)
==================================================
New https://pypi.org/project/ultralytics/8.4.104 available 😃 Update with 'pip install -U ultralytics'
Ultralytics YOLOv8.1.0 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: task=detect, mode=train, model=models/yolov8m.pt, data=data_augmented/data.yaml, epochs=100, time=None, patience=15, batch=8, imgsz=1024, save=True, save_period=-1, cache=False, device=0, workers=8, project=futsal, name=yolov8m_futsal_pipeline, exist_ok=True, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.15, copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=futsal/yolov8m_futsal_pipeline
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1      1392  ultralytics.nn.modules.conv.Conv             [3, 48, 3, 2]                 
  1                  -1  1     41664  ultralytics.nn.modules.conv.Conv             [48, 96, 3, 2]                
  2                  -1  2    111360  ultralytics.nn.modules.block.C2f             [96, 96, 2, True]             
  3                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  4                  -1  4    813312  ultralytics.nn.modules.block.C2f             [192, 192, 4, True]           
  5                  -1  1    664320  ultralytics.nn.modules.conv.Conv             [192, 384, 3, 2]              
  6                  -1  4   3248640  ultralytics.nn.modules.block.C2f             [384, 384, 4, True]           
  7                  -1  1   1991808  ultralytics.nn.modules.conv.Conv             [384, 576, 3, 2]              
  8                  -1  2   3985920  ultralytics.nn.modules.block.C2f             [576, 576, 2, True]           
  9                  -1  1    831168  ultralytics.nn.modules.block.SPPF            [576, 576, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  2   1993728  ultralytics.nn.modules.block.C2f             [960, 384, 2]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  2    517632  ultralytics.nn.modules.block.C2f             [576, 192, 2]                 
 16                  -1  1    332160  ultralytics.nn.modules.conv.Conv             [192, 192, 3, 2]              
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  2   1846272  ultralytics.nn.modules.block.C2f             [576, 384, 2]                 
 19                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  2   4207104  ultralytics.nn.modules.block.C2f             [960, 576, 2]                 
 22        [15, 18, 21]  1   3778012  ultralytics.nn.modules.head.Detect           [4, [192, 384, 576]]          
Model summary: 295 layers, 25858636 parameters, 25858620 gradients, 79.1 GFLOPs

Transferred 469/475 items from pretrained weights
TensorBoard: Start with 'tensorboard --logdir futsal/yolov8m_futsal_pipeline', view at http://localhost:6006/
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 3
wandb: You chose "Don't visualize my results"
wandb: Using W&B in offline mode.
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.28.0
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Run data is saved locally in /content/futsal-cv/wandb/offline-run-20260722_021603-9k1v3kss
wandb: View this run in the terminal with `wandb leet`
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
/usr/local/lib/python3.12/dist-packages/ultralytics/utils/checks.py:638: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(True):
AMP: checks passed ✅
/usr/local/lib/python3.12/dist-packages/ultralytics/engine/trainer.py:271: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
train: Scanning /content/futsal-cv/data_augmented/train/labels... 1024 images, 0 backgrounds, 0 corrupt: 100% 1024/1024 [00:00<00:00, 1987.62it/s]
train: New cache created: /content/futsal-cv/data_augmented/train/labels.cache
/usr/local/lib/python3.12/dist-packages/ultralytics/data/augment.py:847: UserWarning: Argument(s) 'quality_lower' are not valid for transform ImageCompression
  A.ImageCompression(quality_lower=75, p=0.0),
/usr/local/lib/python3.12/dist-packages/albumentations/core/composition.py:331: UserWarning: Got processor for bboxes, but no transform to process it.
  self._set_keys()
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Scanning /content/futsal-cv/data_augmented/valid/labels... 73 images, 0 backgrounds, 0 corrupt: 100% 73/73 [00:00<00:00, 940.61it/s]
val: New cache created: /content/futsal-cv/data_augmented/valid/labels.cache
Plotting labels to futsal/yolov8m_futsal_pipeline/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 77 weight(decay=0.0), 84 weight(decay=0.0005), 83 bias(decay=0.0)
100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      8.62G      1.984      2.366      2.446        229       1024: 100% 128/128 [01:29<00:00,  1.44it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:04<00:00,  1.15it/s]
                   all         73        815      0.672      0.386      0.431      0.173

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      2/100      8.85G      1.645        1.9      2.201        233       1024: 100% 128/128 [01:34<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.80it/s]
                   all         73        815       0.81      0.371      0.449      0.226

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      3/100      8.53G      1.603      1.833      2.172        126       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.74it/s]
                   all         73        815      0.745       0.45      0.478      0.216

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      4/100      8.41G       1.53      1.778      2.113        182       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.71it/s]
                   all         73        815      0.786      0.522      0.575      0.308

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      5/100      8.76G       1.45      1.668      2.042        237       1024: 100% 128/128 [01:33<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.71it/s]
                   all         73        815      0.558      0.563      0.611      0.335

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      6/100      8.88G      1.428      1.653      2.028        284       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.86it/s]
                   all         73        815      0.475      0.537       0.56      0.324

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      7/100      8.87G      1.369        1.6      1.968        190       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.85it/s]
                   all         73        815      0.713      0.653      0.676      0.438

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      8/100      8.88G      1.334      1.537      1.942        226       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.76it/s]
                   all         73        815      0.712       0.61      0.653      0.415

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      9/100      8.36G      1.298      1.475      1.906        205       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.86it/s]
                   all         73        815      0.749       0.69      0.707      0.474

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     10/100      8.37G      1.293      1.492      1.905        195       1024: 100% 128/128 [01:34<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.84it/s]
                   all         73        815       0.78      0.728      0.746      0.485

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     11/100      8.59G      1.251      1.445      1.854        192       1024: 100% 128/128 [01:33<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.76it/s]
                   all         73        815      0.785      0.731      0.755       0.54

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     12/100      8.45G      1.235      1.417      1.852        216       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.87it/s]
                   all         73        815      0.862      0.705      0.772      0.531

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     13/100      8.39G      1.231       1.42      1.834        166       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.87it/s]
                   all         73        815      0.833      0.755      0.784      0.534

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     14/100      8.41G      1.235      1.391      1.852        214       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.74it/s]
                   all         73        815      0.816      0.806      0.807      0.569

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     15/100      8.59G        1.2      1.378      1.816        276       1024: 100% 128/128 [01:34<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.75it/s]
                   all         73        815      0.801      0.765      0.774      0.564

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     16/100      8.48G      1.176      1.356      1.787        227       1024: 100% 128/128 [01:33<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.72it/s]
                   all         73        815      0.853       0.78        0.8      0.579

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     17/100      8.57G      1.151      1.321      1.768        268       1024: 100% 128/128 [01:34<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.88it/s]
                   all         73        815      0.891      0.761      0.811      0.582

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     18/100      8.94G      1.153      1.314       1.77        181       1024: 100% 128/128 [01:33<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.75it/s]
                   all         73        815      0.851      0.795      0.815      0.612

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     19/100      8.86G      1.144      1.326      1.766        198       1024: 100% 128/128 [01:34<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.88it/s]
                   all         73        815      0.819      0.789      0.795      0.617

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     20/100      8.89G      1.111      1.284       1.72        246       1024: 100% 128/128 [01:33<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.82it/s]
                   all         73        815      0.842       0.81      0.813      0.614

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     21/100      8.91G      1.107      1.285      1.728        183       1024: 100% 128/128 [01:33<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.86it/s]
                   all         73        815      0.859      0.789       0.82       0.64

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     22/100      8.59G      1.112      1.264      1.726        167       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.84it/s]
                   all         73        815      0.866      0.798      0.833      0.641

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     23/100      8.39G      1.089      1.216      1.708        157       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.82it/s]
                   all         73        815      0.877      0.801      0.824      0.651

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     24/100      8.41G      1.072      1.243      1.685        204       1024: 100% 128/128 [01:34<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.90it/s]
                   all         73        815      0.879      0.808      0.815      0.645

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     25/100      8.59G      1.063      1.226      1.682        149       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.86it/s]
                   all         73        815      0.837      0.765      0.799      0.602

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     26/100      8.74G      1.047      1.211      1.672        168       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.78it/s]
                   all         73        815      0.879      0.806      0.823      0.635

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     27/100      8.62G      1.034        1.2      1.654        194       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.88it/s]
                   all         73        815      0.865       0.81      0.821      0.625

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     28/100      8.57G      1.028       1.19      1.653        233       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.88it/s]
                   all         73        815      0.861       0.83      0.841      0.647

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     29/100      8.56G       1.02       1.16      1.634        202       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.86it/s]
                   all         73        815      0.832      0.822      0.821      0.644

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     30/100      8.59G      1.013      1.156      1.637        169       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.77it/s]
                   all         73        815       0.85      0.805      0.829       0.65

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     31/100       8.6G     0.9993      1.162      1.626        257       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.83it/s]
                   all         73        815      0.857      0.807      0.851      0.662

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     32/100      8.86G      1.006      1.145      1.635        173       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.83it/s]
                   all         73        815      0.862      0.795      0.816      0.626

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     33/100      8.56G      1.001      1.163      1.625        212       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.88it/s]
                   all         73        815      0.849      0.796      0.822      0.639

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     34/100      8.71G     0.9899      1.132      1.613        166       1024: 100% 128/128 [01:34<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.89it/s]
                   all         73        815       0.87      0.819      0.832      0.652

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     35/100      8.64G     0.9626      1.105      1.585        188       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.87it/s]
                   all         73        815      0.858      0.829      0.828      0.652

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     36/100      8.55G      0.987      1.129      1.611        232       1024: 100% 128/128 [01:34<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.89it/s]
                   all         73        815      0.821      0.826      0.822       0.66

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     37/100      8.69G     0.9692      1.122      1.598        191       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.89it/s]
                   all         73        815      0.869      0.806       0.82      0.657

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     38/100      8.59G     0.9595      1.087      1.579        143       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.71it/s]
                   all         73        815      0.869      0.811      0.835       0.67

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     39/100      8.55G      0.967       1.11      1.593        197       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.87it/s]
                   all         73        815      0.885      0.825      0.836      0.658

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     40/100      8.38G     0.9575      1.117      1.583        237       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.86it/s]
                   all         73        815      0.855      0.838      0.826      0.656

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     41/100      8.72G     0.9495      1.071      1.582        182       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.87it/s]
                   all         73        815      0.866      0.832      0.854      0.682

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     42/100      8.55G     0.9466      1.096      1.574        182       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.72it/s]
                   all         73        815      0.886      0.813      0.832      0.662

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     43/100      8.58G     0.9372      1.083       1.56        166       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.76it/s]
                   all         73        815      0.873      0.832      0.837       0.67

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     44/100      8.61G     0.9353      1.063      1.558        156       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.86it/s]
                   all         73        815      0.892      0.829      0.864      0.696

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     45/100      8.59G     0.9225      1.055      1.545        174       1024: 100% 128/128 [01:34<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.88it/s]
                   all         73        815      0.864      0.829      0.833      0.679

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     46/100      8.59G     0.9204       1.05      1.557        193       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.88it/s]
                   all         73        815      0.881      0.833      0.844       0.68

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     47/100      8.54G     0.9183      1.045      1.544        237       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.85it/s]
                   all         73        815      0.878      0.821      0.843      0.681

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     48/100       8.6G     0.9155       1.05      1.541        264       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.74it/s]
                   all         73        815      0.881      0.812      0.837      0.678

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     49/100      8.66G     0.9081      1.036      1.537        194       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.85it/s]
                   all         73        815      0.894      0.821      0.852      0.702

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     50/100      8.57G     0.8855      1.014      1.515        227       1024: 100% 128/128 [01:33<00:00,  1.36it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.89it/s]
                   all         73        815      0.875      0.826      0.846      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     51/100      8.64G     0.8957      1.024      1.518        116       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.87it/s]
                   all         73        815      0.895      0.812      0.853      0.695

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     52/100      8.38G     0.8824      1.019      1.515        167       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.88it/s]
                   all         73        815      0.872      0.832      0.854      0.694

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     53/100       8.7G     0.8866      1.011      1.508        160       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.88it/s]
                   all         73        815      0.893      0.813      0.841      0.698

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     54/100      8.59G     0.8904      1.009      1.515        155       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.87it/s]
                   all         73        815      0.902      0.831      0.859      0.707

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     55/100      8.72G     0.8776      1.013      1.507        164       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.86it/s]
                   all         73        815      0.892      0.818       0.84       0.68

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     56/100      8.72G      0.878      1.003      1.507        228       1024: 100% 128/128 [01:33<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.84it/s]
                   all         73        815       0.87      0.835      0.832      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     57/100      8.57G     0.8815     0.9929      1.514        156       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.73it/s]
                   all         73        815      0.881      0.815      0.844      0.699

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     58/100      8.73G      0.874     0.9981      1.503        131       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.74it/s]
                   all         73        815      0.878      0.832      0.842      0.686

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     59/100       8.6G     0.8606     0.9709        1.5        254       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.85it/s]
                   all         73        815      0.882      0.816      0.848      0.701

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     60/100      8.65G     0.8709     0.9967      1.501        179       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.90it/s]
                   all         73        815      0.875       0.82      0.847      0.688

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     61/100      8.59G     0.8394     0.9576       1.47        260       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.79it/s]
                   all         73        815      0.902      0.807      0.837      0.694

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     62/100      8.68G      0.858     0.9655      1.488        203       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.89it/s]
                   all         73        815      0.875      0.832      0.839      0.691

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     63/100      8.56G      0.841     0.9493      1.468        138       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.86it/s]
                   all         73        815      0.868      0.843      0.839      0.691

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     64/100      8.74G     0.8186     0.9281      1.445        221       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.71it/s]
                   all         73        815       0.89      0.814      0.837       0.69

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     65/100      8.62G     0.8513     0.9449      1.481        275       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.76it/s]
                   all         73        815      0.891      0.819      0.839      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     66/100       8.6G     0.8394     0.9393      1.464        190       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.87it/s]
                   all         73        815      0.898       0.81      0.838      0.696

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     67/100      8.54G     0.8178     0.9315      1.447        297       1024: 100% 128/128 [01:33<00:00,  1.37it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.88it/s]
                   all         73        815      0.896      0.811      0.844        0.7

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     68/100      8.59G     0.8132     0.9167      1.442        200       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.87it/s]
                   all         73        815      0.877      0.832      0.826      0.692

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     69/100      8.67G     0.8099     0.9068      1.442        171       1024: 100% 128/128 [01:32<00:00,  1.38it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:02<00:00,  1.84it/s]
                   all         73        815      0.881      0.828      0.835      0.692
Stopping training early as no improvement observed in last 15 epochs. Best results observed at epoch 54, best model saved as best.pt.
To update EarlyStopping(patience=15) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.

69 epochs completed in 1.997 hours.
Optimizer stripped from futsal/yolov8m_futsal_pipeline/weights/last.pt, 52.1MB
Optimizer stripped from futsal/yolov8m_futsal_pipeline/weights/best.pt, 52.1MB

Validating futsal/yolov8m_futsal_pipeline/weights/best.pt...
Ultralytics YOLOv8.1.0 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
Model summary (fused): 218 layers, 25842076 parameters, 0 gradients, 78.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 5/5 [00:05<00:00,  1.11s/it]
                   all         73        815      0.902      0.831      0.859      0.707
                  ball         73         51      0.797      0.725      0.742      0.632
            goalkeeper         73         56      0.962      0.911      0.925      0.742
                player         73        525      0.921      0.803      0.866      0.721
               referee         73        183      0.926      0.885      0.904      0.733
Speed: 0.8ms preprocess, 34.5ms inference, 0.0ms loss, 7.2ms postprocess per image
Results saved to futsal/yolov8m_futsal_pipeline
wandb: 
wandb: Run history:
wandb:                  lr/pg0 ▁▅███▇▇▇▇▇▇▆▆▆▆▆▅▅▅▅▅▄▄▄▄▃▃▃▃▃▃▂▂▂▂▂▂▁▁▁
wandb:                  lr/pg1 ▁▅███▇▇▇▇▆▆▆▆▆▆▅▅▅▅▅▄▄▄▄▄▄▄▃▃▃▃▃▂▂▂▂▂▁▁▁
wandb:                  lr/pg2 ▅████▇▇▇▇▇▇▆▆▆▆▆▆▅▅▅▅▅▅▄▄▄▄▄▄▃▃▃▃▂▂▂▂▂▁▁
wandb:        metrics/mAP50(B) ▁▁▂▃▄▅▅▆▇▇▇▇▇▇▇▇█▇▇█▇▇█▇████████▇███████
wandb:     metrics/mAP50-95(B) ▁▂▂▃▃▅▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇█▇████████████████
wandb:    metrics/precision(B) ▁▃▂▃▄▆▅▅▆█▆▇▇▇▇▇▆▇▇▆▇▇▇▇▇█▇▇█▇▇██▇▇▇█▇██
wandb:       metrics/recall(B) ▁▂▄▃▅▆▆▆▇▇▇▇▇▇▇▇██▇▇▇██▇████████████████
wandb:            model/GFLOPs ▁
wandb:        model/parameters ▁
wandb: model/speed_PyTorch(ms) ▁
wandb:                      +6 ...
wandb: 
wandb: Run summary:
wandb:                  lr/pg0 0.00042
wandb:                  lr/pg1 0.00042
wandb:                  lr/pg2 0.00042
wandb:        metrics/mAP50(B) 0.85923
wandb:     metrics/mAP50-95(B) 0.70703
wandb:    metrics/precision(B) 0.90158
wandb:       metrics/recall(B) 0.83087
wandb:            model/GFLOPs 79.075
wandb:        model/parameters 25858636
wandb: model/speed_PyTorch(ms) 33.024
wandb:                      +6 ...
wandb: 
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /content/futsal-cv/wandb/offline-run-20260722_021603-9k1v3kss
wandb: Find logs at: ./wandb/offline-run-20260722_021603-9k1v3kss/logs
==================================================
Training Pipeline Finished!
==================================================
Model terbaik (yolov8m) berhasil disalin ke: /content/futsal-cv/models/best_futsal.pt
