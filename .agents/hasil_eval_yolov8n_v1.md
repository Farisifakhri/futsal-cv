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
[Step 3/3] Memulai Training Model YOLOv8
==================================================
Downloading https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt to 'models/yolov8n.pt'...
100% 6.23M/6.23M [00:00<00:00, 104MB/s]
New https://pypi.org/project/ultralytics/8.4.104 available 😃 Update with 'pip install -U ultralytics'
Ultralytics YOLOv8.1.0 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
engine/trainer: task=detect, mode=train, model=models/yolov8n.pt, data=data_augmented/data.yaml, epochs=50, time=None, patience=15, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=0, workers=8, project=futsal, name=yolov8n_futsal_pipeline, exist_ok=True, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.15, copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=futsal/yolov8n_futsal_pipeline
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1    752092  ultralytics.nn.modules.head.Detect           [4, [64, 128, 256]]           
Model summary: 225 layers, 3011628 parameters, 3011612 gradients, 8.2 GFLOPs

Transferred 319/355 items from pretrained weights
TensorBoard: Start with 'tensorboard --logdir futsal/yolov8n_futsal_pipeline', view at http://localhost:6006/
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 3
wandb: You chose "Don't visualize my results"
wandb: Using W&B in offline mode.
wandb: W&B API key is configured. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.28.0
wandb: W&B syncing is set to `offline` in this directory. Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
wandb: Run data is saved locally in /content/futsal-cv/wandb/offline-run-20260722_000031-zxgwyxji
wandb: View this run in the terminal with `wandb leet`
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks with YOLOv8n...
/usr/local/lib/python3.12/dist-packages/ultralytics/utils/checks.py:638: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast(True):
AMP: checks passed ✅
/usr/local/lib/python3.12/dist-packages/ultralytics/engine/trainer.py:271: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
train: Scanning /content/futsal-cv/data_augmented/train/labels... 1024 images, 0 backgrounds, 0 corrupt: 100% 1024/1024 [00:00<00:00, 1817.81it/s]
train: New cache created: /content/futsal-cv/data_augmented/train/labels.cache
/usr/local/lib/python3.12/dist-packages/ultralytics/data/augment.py:847: UserWarning: Argument(s) 'quality_lower' are not valid for transform ImageCompression
  A.ImageCompression(quality_lower=75, p=0.0),
/usr/local/lib/python3.12/dist-packages/albumentations/core/composition.py:331: UserWarning: Got processor for bboxes, but no transform to process it.
  self._set_keys()
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))
val: Scanning /content/futsal-cv/data_augmented/valid/labels... 73 images, 0 backgrounds, 0 corrupt: 100% 73/73 [00:00<00:00, 875.10it/s]
val: New cache created: /content/futsal-cv/data_augmented/valid/labels.cache
Plotting labels to futsal/yolov8n_futsal_pipeline/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
50 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/50      2.78G      2.181      2.812      2.437        323        640: 100% 64/64 [00:25<00:00,  2.47it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:02<00:00,  1.30it/s]
                   all         73        815       0.66      0.175      0.227     0.0796

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/50      2.69G      1.723      2.151      2.229        416        640: 100% 64/64 [00:21<00:00,  2.91it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.76it/s]
                   all         73        815      0.619      0.364       0.36      0.162

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/50      2.46G      1.627      1.952      2.148        309        640: 100% 64/64 [00:21<00:00,  2.99it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.96it/s]
                   all         73        815      0.555      0.453      0.403      0.144

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/50      3.07G      1.518      1.834      2.043        409        640: 100% 64/64 [00:22<00:00,  2.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.92it/s]
                   all         73        815      0.801      0.382      0.463      0.251

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/50      2.53G      1.463      1.759      1.983        470        640: 100% 64/64 [00:22<00:00,  2.82it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.63it/s]
                   all         73        815      0.841      0.518      0.593      0.304

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/50      2.69G      1.439      1.719      1.962        492        640: 100% 64/64 [00:21<00:00,  3.02it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.90it/s]
                   all         73        815      0.863      0.559      0.626       0.39

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/50      3.31G      1.395      1.634      1.913        393        640: 100% 64/64 [00:21<00:00,  2.98it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.77it/s]
                   all         73        815      0.861      0.562      0.628      0.389

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/50      3.53G      1.362      1.605      1.883        381        640: 100% 64/64 [00:23<00:00,  2.78it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.05it/s]
                   all         73        815      0.869      0.567      0.651      0.411

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/50      2.75G      1.328       1.59      1.857        416        640: 100% 64/64 [00:21<00:00,  2.95it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.85it/s]
                   all         73        815      0.622      0.583      0.644        0.4

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/50      2.97G      1.302      1.552      1.844        394        640: 100% 64/64 [00:21<00:00,  2.99it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.57it/s]
                   all         73        815      0.883      0.578      0.673      0.456

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/50      2.79G      1.274      1.507      1.807        438        640: 100% 64/64 [00:23<00:00,  2.78it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.77it/s]
                   all         73        815      0.893      0.574      0.672       0.43

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/50      2.78G      1.261      1.488      1.801        442        640: 100% 64/64 [00:23<00:00,  2.78it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.14it/s]
                   all         73        815      0.743       0.63       0.69      0.497

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/50      2.83G      1.244      1.474      1.787        397        640: 100% 64/64 [00:20<00:00,  3.10it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.23it/s]
                   all         73        815      0.714      0.666      0.692      0.487

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/50      3.05G      1.234      1.457      1.776        455        640: 100% 64/64 [00:22<00:00,  2.89it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.65it/s]
                   all         73        815      0.638      0.698      0.685      0.457

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/50      2.75G      1.216      1.424      1.757        553        640: 100% 64/64 [00:23<00:00,  2.78it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.80it/s]
                   all         73        815      0.759      0.673      0.723      0.498

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/50      2.93G      1.202      1.412      1.741        462        640: 100% 64/64 [00:20<00:00,  3.06it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.94it/s]
                   all         73        815      0.734      0.701       0.73      0.532

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/50      3.06G      1.175      1.381      1.719        465        640: 100% 64/64 [00:21<00:00,  2.91it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.39it/s]
                   all         73        815      0.819      0.664      0.723      0.516

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/50      2.62G      1.166      1.376      1.718        418        640: 100% 64/64 [00:22<00:00,  2.83it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.11it/s]
                   all         73        815      0.903      0.667      0.756      0.567

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/50      2.93G      1.171       1.36      1.718        354        640: 100% 64/64 [00:21<00:00,  2.98it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.84it/s]
                   all         73        815      0.863      0.692      0.766      0.564

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/50      2.74G      1.138      1.338      1.683        471        640: 100% 64/64 [00:21<00:00,  2.99it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.53it/s]
                   all         73        815      0.846      0.685      0.757      0.549

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/50       3.1G      1.151      1.337      1.702        326        640: 100% 64/64 [00:23<00:00,  2.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  4.07it/s]
                   all         73        815       0.84      0.695      0.752      0.544

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/50      2.78G      1.118      1.315      1.667        351        640: 100% 64/64 [00:22<00:00,  2.89it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.75it/s]
                   all         73        815      0.841      0.691      0.749      0.546

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/50      2.58G      1.097      1.268       1.65        363        640: 100% 64/64 [00:21<00:00,  3.01it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.63it/s]
                   all         73        815      0.812      0.709      0.754      0.557

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/50       2.8G      1.099      1.293      1.656        388        640: 100% 64/64 [00:22<00:00,  2.79it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.17it/s]
                   all         73        815      0.861      0.717      0.778      0.593

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/50      2.75G      1.112      1.283      1.663        397        640: 100% 64/64 [00:23<00:00,  2.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.74it/s]
                   all         73        815      0.881      0.696      0.767      0.566

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      26/50      2.97G      1.093      1.278      1.642        347        640: 100% 64/64 [00:21<00:00,  2.92it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.69it/s]
                   all         73        815      0.883      0.705      0.773      0.603

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      27/50      2.95G      1.055      1.236      1.619        422        640: 100% 64/64 [00:21<00:00,  2.99it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.77it/s]
                   all         73        815      0.871      0.694      0.778      0.596

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      28/50      2.86G      1.073      1.258      1.634        494        640: 100% 64/64 [00:23<00:00,  2.74it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.11it/s]
                   all         73        815       0.84      0.711      0.768      0.596

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      29/50      2.79G      1.046      1.226      1.607        431        640: 100% 64/64 [00:22<00:00,  2.85it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.41it/s]
                   all         73        815       0.88      0.735      0.779      0.595

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      30/50      3.69G      1.041      1.207      1.599        345        640: 100% 64/64 [00:21<00:00,  2.98it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.62it/s]
                   all         73        815      0.852       0.75      0.782      0.573

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      31/50      2.95G      1.046      1.218      1.603        483        640: 100% 64/64 [00:22<00:00,  2.84it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.42it/s]
                   all         73        815      0.932      0.705      0.791      0.613

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      32/50      3.03G      1.019      1.187      1.582        335        640: 100% 64/64 [00:23<00:00,  2.72it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.93it/s]
                   all         73        815       0.89      0.706      0.788      0.608

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      33/50      2.95G       1.04      1.225      1.595        437        640: 100% 64/64 [00:22<00:00,  2.88it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.74it/s]
                   all         73        815      0.866      0.731      0.792      0.616

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      34/50      3.07G       1.02      1.194      1.582        459        640: 100% 64/64 [00:21<00:00,  2.95it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.00it/s]
                   all         73        815        0.9      0.726      0.796      0.622

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      35/50       2.8G     0.9989      1.186      1.557        324        640: 100% 64/64 [00:22<00:00,  2.81it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.17it/s]
                   all         73        815      0.894      0.723      0.801      0.631

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      36/50      2.74G      1.017      1.188      1.581        397        640: 100% 64/64 [00:23<00:00,  2.74it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.92it/s]
                   all         73        815      0.857      0.745      0.793      0.635

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      37/50      3.21G      1.005      1.165      1.567        414        640: 100% 64/64 [00:21<00:00,  2.93it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.15it/s]
                   all         73        815      0.904      0.718      0.797      0.634

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      38/50      2.69G     0.9857      1.161      1.547        430        640: 100% 64/64 [00:21<00:00,  3.03it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.89it/s]
                   all         73        815      0.878      0.699      0.788      0.622

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      39/50      3.12G     0.9838      1.162      1.545        406        640: 100% 64/64 [00:23<00:00,  2.74it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.92it/s]
                   all         73        815      0.872      0.741      0.798      0.641

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      40/50      2.74G     0.9872      1.164       1.55        409        640: 100% 64/64 [00:23<00:00,  2.73it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.57it/s]
                   all         73        815      0.873      0.737      0.802      0.637
Closing dataloader mosaic
/usr/local/lib/python3.12/dist-packages/ultralytics/data/augment.py:847: UserWarning: Argument(s) 'quality_lower' are not valid for transform ImageCompression
  A.ImageCompression(quality_lower=75, p=0.0),
/usr/local/lib/python3.12/dist-packages/albumentations/core/composition.py:331: UserWarning: Got processor for bboxes, but no transform to process it.
  self._set_keys()
albumentations: Blur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01, method='weighted_average', num_output_channels=3), CLAHE(p=0.01, clip_limit=(1.0, 4.0), tile_grid_size=(8, 8))

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      41/50      2.22G     0.7523     0.8258      1.377        175        640: 100% 64/64 [00:20<00:00,  3.10it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.59it/s]
                   all         73        815      0.856      0.769      0.799      0.646

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      42/50      2.24G     0.7267     0.7563      1.352        175        640: 100% 64/64 [00:19<00:00,  3.24it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.92it/s]
                   all         73        815       0.84      0.751        0.8      0.647

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      43/50      2.24G     0.6956     0.7211       1.32        167        640: 100% 64/64 [00:19<00:00,  3.25it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  1.72it/s]
                   all         73        815      0.836      0.785      0.811       0.67

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      44/50      2.24G     0.6728     0.7072      1.307        178        640: 100% 64/64 [00:19<00:00,  3.32it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.27it/s]
                   all         73        815      0.861      0.752       0.81      0.669

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      45/50      2.22G     0.6621     0.6887        1.3        169        640: 100% 64/64 [00:20<00:00,  3.13it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.27it/s]
                   all         73        815      0.842      0.756      0.806      0.665

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      46/50      2.22G     0.6641     0.6883      1.297        166        640: 100% 64/64 [00:18<00:00,  3.39it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.61it/s]
                   all         73        815      0.862      0.747      0.807      0.674

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      47/50      2.24G     0.6598     0.6877        1.3        183        640: 100% 64/64 [00:20<00:00,  3.08it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:01<00:00,  2.66it/s]
                   all         73        815       0.83      0.768      0.808      0.677

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      48/50      2.24G     0.6459     0.6866      1.281        183        640: 100% 64/64 [00:19<00:00,  3.31it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.96it/s]
                   all         73        815      0.835      0.775      0.811      0.678

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      49/50      2.24G     0.6368     0.6694      1.268        184        640: 100% 64/64 [00:21<00:00,  3.04it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.08it/s]
                   all         73        815      0.852      0.763      0.813      0.677

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      50/50      2.22G     0.6244     0.6623      1.257        169        640: 100% 64/64 [00:19<00:00,  3.30it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:00<00:00,  3.42it/s]
                   all         73        815      0.873      0.761      0.811      0.679

50 epochs completed in 0.339 hours.
Optimizer stripped from futsal/yolov8n_futsal_pipeline/weights/last.pt, 6.3MB
Optimizer stripped from futsal/yolov8n_futsal_pipeline/weights/best.pt, 6.3MB

Validating futsal/yolov8n_futsal_pipeline/weights/best.pt...
Ultralytics YOLOv8.1.0 🚀 Python-3.12.13 torch-2.11.0+cu128 CUDA:0 (Tesla T4, 14913MiB)
Model summary (fused): 168 layers, 3006428 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% 3/3 [00:02<00:00,  1.23it/s]
                   all         73        815      0.872      0.762       0.81      0.679
                  ball         73         51      0.738       0.49      0.549      0.495
            goalkeeper         73         56      0.941      0.856      0.908      0.732
                player         73        525      0.903      0.815      0.876      0.739
               referee         73        183      0.905      0.885      0.908      0.749
Speed: 0.3ms preprocess, 3.0ms inference, 0.0ms loss, 5.5ms postprocess per image
Results saved to futsal/yolov8n_futsal_pipeline
wandb: 
wandb: Run history:
wandb:                  lr/pg0 ▃▆████▇▇▇▇▇▇▆▆▆▆▆▅▅▅▅▅▅▄▄▄▄▄▃▃▃▃▃▃▂▂▂▂▁▁
wandb:                  lr/pg1 ▃▆████▇▇▇▇▇▇▆▆▆▆▆▅▅▅▅▅▅▄▄▄▄▄▃▃▃▃▃▃▂▂▂▂▁▁
wandb:                  lr/pg2 ▃▆████▇▇▇▇▇▆▆▆▆▆▅▅▅▅▅▅▄▄▄▄▄▃▃▃▃▃▂▂▂▂▂▁▁▁
wandb:        metrics/mAP50(B) ▁▂▃▅▅▆▅▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇█████████████████
wandb:     metrics/mAP50-95(B) ▁▂▂▃▄▅▅▅▅▅▆▅▆▆▆▇▆▆▆▇▇▇▇▇▇▇▇▇▇▇██████████
wandb:    metrics/precision(B) ▃▂▁▆▇▇▇▂██▄▃▅▆█▇▇▇▆▇█▇▇█▇█▇███▇▇▇▇▇▇▇▇▇▇
wandb:       metrics/recall(B) ▁▃▄▃▅▅▆▆▆▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇█▇▇▇█▇▇▇████████
wandb:            model/GFLOPs ▁
wandb:        model/parameters ▁
wandb: model/speed_PyTorch(ms) ▁
wandb:                      +6 ...
wandb: 
wandb: Run summary:
wandb:                  lr/pg0 6e-05
wandb:                  lr/pg1 6e-05
wandb:                  lr/pg2 6e-05
wandb:        metrics/mAP50(B) 0.81016
wandb:     metrics/mAP50-95(B) 0.67873
wandb:    metrics/precision(B) 0.8719
wandb:       metrics/recall(B) 0.76177
wandb:            model/GFLOPs 8.197
wandb:        model/parameters 3011628
wandb: model/speed_PyTorch(ms) 12.293
wandb:                      +6 ...
wandb: 
wandb: You can sync this run to the cloud by running:
wandb: wandb sync /content/futsal-cv/wandb/offline-run-20260722_000031-zxgwyxji
wandb: Find logs at: ./wandb/offline-run-20260722_000031-zxgwyxji/logs
==================================================
Training Pipeline Finished!
==================================================
Model terbaik berhasil disalin ke: /content/futsal-cv/models/best_futsal.pt
