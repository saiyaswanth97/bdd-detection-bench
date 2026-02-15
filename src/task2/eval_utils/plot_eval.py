# Copyright (c) 2026 Sai Yaswanth. All rights reserved.

import os
import json
import numpy as np
import matplotlib.pyplot as plt
# import tensorboard

TB_FILE = "../../output/events.out.tfevents.1771041727.bhargav-reddy-desktop.670268.0"
EVAL_FOLDER = "output/eval"
CLASSES = [
    "car",
    "traffic sign",
    "traffic light",
    "person",
    "truck",
    "bus",
    "bike",
    "rider",
    "motor",
    "train",
]

class_ap_list = {class_name: [] for class_name in CLASSES}
map_list = {
    "AP": [],
    "AP50": [],
    "AP75": [],
    "APs": [],
    "APm": [],
    "APl": [],
}
map_s = {class_name: [] for class_name in CLASSES}
map_m = {class_name: [] for class_name in CLASSES}
map_l = {class_name: [] for class_name in CLASSES}
map_per_class = {
    "small": {class_name: [] for class_name in CLASSES},
    "medium": {class_name: [] for class_name in CLASSES},
    "large": {class_name: [] for class_name in CLASSES},
}

# writer = tensorboard.SummaryWriter(log_dir=os.path.dirname(TB_FILE))

# print(sorted(os.listdir(EVAL_FOLDER)))
i = 0
for folder in sorted(os.listdir(EVAL_FOLDER)):
    json_file = os.path.join(EVAL_FOLDER, folder, "eval_results.json")
    with open(json_file, "r") as f:
        eval_results = json.load(f)
    eval_results = eval_results["bbox"]
    for key in map_list.keys():
        map_list[key].append(eval_results[key])
        # writer.add_scalar("eval/" + key, eval_results[key], i)
    for class_name in CLASSES:
        class_ap_list[class_name].append(eval_results["AP-" + class_name])
        # writer.add_scalar("eval/class_AP/" + class_name, eval_results["AP-" + class_name], i)

    dense_json_file = os.path.join(EVAL_FOLDER, folder, "precision_matrix.json")
    with open(dense_json_file, "r") as f:
        dense_results = json.load(f)

    for i, class_name in enumerate(CLASSES):
        precision_matrix = dense_results["precision"]
        precision_matrix = np.array(precision_matrix)

        iterations = [["small", 1], ["medium", 2], ["large", 3]]
        for size_name, size_idx in iterations:
            ap_all = precision_matrix[:, :, i, size_idx, 2]
            ap = np.mean(ap_all[ap_all > -1]) if (ap_all > -1).any() else -1.0
            map_per_class[size_name][class_name].append(ap)
            # writer.add_scalar(f"eval/{size_name}_ap/" + class_name, ap, i)

    i += 0.5
# writer.flush()
# writer.close()

fig = plt.figure(figsize=(12, 6))
plt.plot(map_list["AP"], label="AP")
plt.plot(map_list["AP50"], label="AP50")
plt.plot(map_list["AP75"], label="AP75")
plt.plot(map_list["APs"], label="APs")
plt.plot(map_list["APm"], label="APm")
plt.plot(map_list["APl"], label="APl")
plt.xlabel("Epoch")
plt.ylabel("mAP")
plt.title("mAP over Epochs")
plt.legend()
# plt.show()
plt.savefig("src/task3/r_cnn_train/mAP_over_epochs.png")

fig = plt.figure(figsize=(12, 6))
for class_name in CLASSES:
    plt.plot(class_ap_list[class_name], label=class_name)
plt.xlabel("Epoch")
plt.ylabel("AP")
plt.title("AP per Class over Epochs")
plt.legend()
# plt.show()
plt.savefig("src/task3/r_cnn_train/AP_per_class_over_epochs.png")

for size_name in ["small", "medium", "large"]:
    plt.figure(figsize=(12, 6))
    for class_name in CLASSES:
        plt.plot(map_per_class[size_name][class_name], label=class_name)
    plt.xlabel("Epoch")
    plt.ylabel("AP")
    plt.title(f"AP for {size_name.capitalize()} Objects per Class over Epochs")
    plt.legend()
    # plt.show()
    plt.savefig(
        f"src/task3/r_cnn_train/AP_{size_name}_objects_per_class_over_epochs.png"
    )
