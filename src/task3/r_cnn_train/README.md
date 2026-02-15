# Faster R-CNN Training on BDD100K

## Training Loss Analysis

<table>
<tr>
<td width="50%">

**Total Loss**
![Total Loss](images/total_loss.png)

</td>
<td width="50%">

**Classification Loss**
![Classification Loss](images/loss_cls.png)

</td>
</tr>
<tr>
<td colspan="2" align="center">

**Regression Loss (Smooth L1)**
![Regression Loss](images/loss_reg.png)

</td>
</tr>
</table>

Loss combines cross-entropy classification and Smooth L1 regression, showing convergence throughout training.

---

## Evaluation Results

### mAP Analysis

![mAP over Epochs](images/mAP_over_epochs.png)

**Final mAP: 29.0** (vs. COCO baseline: 32.0)

**Key Findings:**
- **mAP_small: 12.0** — primary bottleneck significantly impacting overall performance
- Performance scales with class frequency (class imbalance issue)

### Per-Class Performance

![AP per Class](images/AP_per_class_over_epochs.png)

- **Traffic lights:** Poor performance despite high frequency
- **Truck & Bus:** Performing well (~1/5th traffic light frequency)
- **Other classes:** Performance correlates with frequency

### Object Size Breakdown

<table>
<tr>
<td width="33%">

**Small**
![Small Objects AP](images/AP_small_objects_per_class_over_epochs.png)

</td>
<td width="33%">

**Medium**
![Medium Objects AP](images/AP_medium_objects_per_class_over_epochs.png)

</td>
<td width="33%">

**Large**
![Large Objects AP](images/AP_large_objects_per_class_over_epochs.png)

</td>
</tr>
</table>

- **Small:** Traffic signs excel (rectangular, unoccluded); traffic lights perform well but high frequency lowers class mAP
- **Medium/Large:** Performance scales with object frequency; trucks/buses dominate large objects

---

## Training Improvements

- [ ] Use **Focal Loss** or class-weighted cross-entropy (address class imbalance)
- [ ] Increase image resolution, learning rate, and training duration
- [ ] Upgrade backbone (ConvNeXt/Swin Transformer) or use newer architecture
- [x] Used Detectron2 (10x faster than custom implementation)

---

## Qualitative Analysis

> **Note:** GIFs auto-play and loop continuously (no pause available in markdown). Use browser controls to pause if needed.

### Improved Detection

<table>
<tr>
<td width="33%">

**Traffic Lights**
![](gifs/sample_light_pred.gif)

</td>
<td width="33%">

**Bicycles**
![](gifs/image_5_bikes_pred.gif)

</td>
<td width="33%">

**Motorcycles**
![](gifs/sample_motorcycle_pred.gif)

</td>
</tr>
</table>

### Multi-Object Detection

<table>
<tr>
<td width="33%">

**20 Cars**
![](gifs/image_20_cars_pred.gif)

</td>
<td width="33%">

**Cars in Traffic**
![](gifs/sample_car_pred.gif)

</td>
<td width="33%">

**Car + Sign + Light**
![](gifs/sample_car_sign_light_pred.gif)

</td>
</tr>
</table>

### Reduced False Positives

<table>
<tr>
<td width="50%">

**Person Detection**
![](gifs/sample_person_pred.gif)

</td>
<td width="50%">

**Traffic Signs**
![](gifs/sample_sign_pred.gif)

</td>
</tr>
</table>

### No Significant Change

<table>
<tr>
<td width="50%">

**Train Detection**
![](gifs/sample_train_pred.gif)

</td>
<td width="50%">

**Rider Detection**
![](gifs/sample_rider_pred.gif)

</td>
</tr>
</table>
