# Detr Training:

Repo used: https://github.com/lyuwenyu/RT-DETR/tree/main
Model Used: RT-DETRv2-S
Things changed:
    - Dataloader:
      - Update data location and increase bacthsize to 16
    - Optimizer:
      - Remove LR sceduler
    - Main config:
      - Change epochs to 30 & lower learning rate by 25% 
Files are included in `configs` for re-test
Run command: `python3 tools/train.py -c <coinfig> -t <model_ckpt>`

## Results:

- The final mAP is 32.0, this is lot lower than COCO score of 48.1. Plost are included (images/eval_metrics.png)
- Small object mAP is the biggest bottle neck - 0.141
- Gap between AR@1 vs AR@100 - 0.231 vs 0.495. Due the image has ~15 instances
- Small object AR vs AP - 0.319 vs 0.141. We can find the object but not localize it. 
  
Next steps:
- Increase LR & retrain
- Increase image size - this will increase mAP_small score
- Hyper-param tuning


### Appendix - Train losses

Images to add: (total_loss.png); (loss_giou.png); (loss_bbox.png); (loss_vfl.png)

- Overall loss is not yet platued
- Classification (variable focal loss) is noisy during training
- bbox (L1 regression loss) is similar range to gIoU.
  - Tuning the hyperparms might help with mAP_small
