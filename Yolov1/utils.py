import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


def intersection_over_union(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor, box_format: str = "midpoint"):
    """
    Calculate intersection over union for all samples

    Parameters:
        boxes_preds (tensor): predictions of bounding boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): correct labels of bounding boxes (BATCH_SIZE, 4)
        box_format (str): midpoint (x, y, w, h) or corners (x1, y1, x2, y2)

    Returns:
         tensor: Intersection over union for all samples
    """
    assert box_format == "midpoint" or box_format == "corners", "box_format should be midpoint or corners!"
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes: list, iou_threshold: float, threshold: float, box_format: str = "corners"):
    """
    Does non-max suppression given bounding boxes

    Parameters:
        bboxes (list): list of all bounding boxes, each [class_pred, pred_score, x1, y1, x2, y2]
        iou_theshold (float): intersection over union threshold for bboxes
        threshold (float): threshold to remove bboxes by prediction probability
        box_format (str): "midpoint" or "corners"

    Result:
    list: bboxes after NMS
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
               or intersection_over_union(
                torch.Tensor(chosen_box[2:]),
                torch.Tensor(box[2:]),
                box_format=box_format
            )
               < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
        pred_boxes: list,
        true_boxes: list,
        iou_threshold: float = 0.5,
        box_format: str = "midpoint",
        num_classes: int = 20
):
    """
    Calculates mean average precision

    Parameters:
        pred_boxes (list): list of all bboxes, each [train_idx, class_pred, pred_score, x1, y1, x2, y2]
        true_boxes (list): list of all true boxes
        iou_threshold (float): threshold to take correct predictions
        box_format (str): "midpoint" or "corners"
        num_classes (int): num of classes
    """

    average_precisions = []

    epsilon = 1e-6
    # go through all classes
    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)
        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_predictions = len(ground_truths)

        if total_true_predictions == 0:
            continue

        # go through all detections
        for detection_idx, detection in enumerate(detections):

            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            # go through all true boxes in the specific image
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.Tensor(detection[3:]),
                    torch.Tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_predictions + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        recalls = torch.cat((torch.Tensor([0]), recalls))
        precisions = torch.cat((torch.Tensor([1]), precisions))
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """
    Plots an image with boxes on it
    """
    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)

    ax.imshow(im)

    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Box has invalid dimensions!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        ax.add_patch(rect)

    plt.show()


def get_bboxes(
        loader,
        model,
        iou_threshold,
        threshold,
        box_format="midpoint",
        device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.inference_mode():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7):
    """
     Converts bounding boxes output from Yolo with
     an image split size of S into entire image ratios
     rather than relative to cell ratios.
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat((predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0)
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1-best_box) + bboxes2 * best_box
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    # Rectangle coordinates
    converted_cellboxes = torch.cat((x, y, w_y), dim=-1)
    # Class
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    # Prediction probability
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
    # Class + Prediction probability + Rectangle coordinates
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_cellboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[idx, bbox_idx, :]])
        all_bboxes.append(bboxes)
    return all_bboxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])






