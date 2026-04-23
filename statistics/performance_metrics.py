from pathlib import Path
from typing import Any
from collections.abc import Iterable, Iterator, Callable, Sequence, Mapping
from tqdm import tqdm
from random import sample
from PIL import Image

from dataclasses import dataclass, field
from bidict import bidict

from sklearn.metrics import confusion_matrix
import cv2
import numpy as np

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.dataset import Subset

from torchvision.datasets.coco import CocoDetection
from torchvision.ops import box_iou

from machine_learning.object_detection.inference import infer_image
from machine_learning.object_detection.datamodule import DataModule
from machine_learning.object_detection.object_detector import ObjectDetector
from machine_learning.object_detection.transformations.coco import collate_fn
from machine_learning.object_detection.data_preparation import annotate_image

import machine_learning.object_detection.utils as utils


# define some useful type aliases
type ResultType = tuple[tuple[Any, ...], tuple[Any, ...]]
type ResultsType = list[ResultType]


@dataclass
class MatchedCategories:
    predicted: list[int] = field(default_factory=list)
    truth: list[int] = field(default_factory=list)
    ious: list[float] = field(default_factory=list)


def coco_bbox_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def flatten_generator[T](inpt: Iterable[T]) -> Iterator[T]:
    for item in inpt:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten_generator(item)
        else:
            yield item


def filter_results(results: ResultsType, compare_fn: Callable[[int, int], bool]) -> ResultsType:
    return [
        e for e in results
        if len(e[0]) > 0 and len(e[1]) > 0 and compare_fn(len(e[0][1]), len(e[1][1]))
    ]


def match_predictions_to_truth(
    predicted_boxes: list[list[float]],
    predicted_labels: list[int],
    truth_boxes: list[list[float]],
    truth_labels: list[int],
    iou_thres: float,
) -> MatchedCategories:
    """
    Greedy one-to-one matching:
    - each prediction can match only one truth
    - each truth can be matched only once
    - labels must match
    - IoU must be >= iou_thres
    """
    if len(predicted_boxes) == 0 or len(truth_boxes) == 0:
        return MatchedCategories()

    preds = torch.tensor(predicted_boxes, dtype=torch.float32)
    truths = torch.tensor(truth_boxes, dtype=torch.float32)
    ious = box_iou(preds, truths)

    matched = MatchedCategories()
    used_truth_indices: set[int] = set()

    for pred_idx in range(len(predicted_boxes)):
        # best available truth for this prediction
        best_truth_idx = -1
        best_iou = 0.0

        for truth_idx in range(len(truth_boxes)):
            if truth_idx in used_truth_indices:
                continue
            if predicted_labels[pred_idx] != truth_labels[truth_idx]:
                continue

            iou_val = float(ious[pred_idx, truth_idx].item())
            if iou_val > best_iou:
                best_iou = iou_val
                best_truth_idx = truth_idx

        if best_truth_idx >= 0 and best_iou >= iou_thres:
            used_truth_indices.add(best_truth_idx)
            matched.predicted.append(predicted_labels[pred_idx])
            matched.truth.append(truth_labels[best_truth_idx])
            matched.ious.append(best_iou)

    return matched


def generate_matched_categories(detections: ResultsType, iou_thres: float) -> list[MatchedCategories]:
    matches: list[MatchedCategories] = []

    for result in detections:
        pred_boxes_raw = result[0][0]
        pred_labels = result[0][1]
        truth_boxes_raw = result[1][0]
        truth_labels = result[1][1]

        pred_boxes: list[list[float]] = [list(map(float, box)) for box in pred_boxes_raw]
        truth_boxes: list[list[float]] = [coco_bbox_to_xyxy(list(map(float, box))) for box in truth_boxes_raw]

        if len(pred_boxes) == 0 or len(truth_boxes) == 0:
            continue

        matches.append(
            match_predictions_to_truth(
                predicted_boxes=pred_boxes,
                predicted_labels=pred_labels,
                truth_boxes=truth_boxes,
                truth_labels=truth_labels,
                iou_thres=iou_thres,
            )
        )

    return matches


def compute_detection_counts(
    results: ResultsType,
    iou_thres: float,
) -> tuple[int, int, int, int, list[MatchedCategories]]:
    """
    Returns:
        true_positives, false_positives, false_negatives, num_samples, matched_categories
    """
    matched_categories = generate_matched_categories(results, iou_thres)

    tp = sum(len(m.predicted) for m in matched_categories)

    total_preds = sum(len(result[0][0]) for result in results)
    total_truths = sum(len(result[1][0]) for result in results)

    fp = total_preds - tp
    fn = total_truths - tp

    return tp, fp, fn, len(results), matched_categories


def save_random_inference_samples(
    image_paths: Sequence[Path],
    model: ObjectDetector,
    device: torch.device,
    label_dict: Mapping[int, str],
    output_dir: Path,
    num_images: int = 8,
    conf_thres: float = 0.1,
    model_image_size: tuple[int, int] | None = None,
    iou_thres: float = 0.05,
    nms_class_restricted: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_paths:
        return []

    chosen: list[Path] = sample(list(image_paths), k=min(num_images, len(image_paths)))
    saved_files: list[Path] = []

    for img_path in chosen:
        img: Image.Image = Image.open(img_path).convert("RGB")

        boxes, labels, confs = infer_image(
            image=img,
            model=model,
            device=device,
            label_dict=label_dict,
            conf_thres=conf_thres,
            model_image_size=model_image_size,
            iou_thres=iou_thres,
            nms_class_restricted=nms_class_restricted,
        )

        annotated: np.ndarray = np.array(img)
        if len(boxes) > 0:
            info: list[str] = [f"{label}@{conf:.2f}" for label, conf in zip(labels, confs)]
            annotated = annotate_image(annotated, info, boxes)

        save_path = output_dir / f"{img_path.stem}_annotated.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        saved_files.append(save_path)

    return saved_files


def main() -> None:
    """Main function to run performance metrics calculations."""
    print("Calculating performance metrics...")

    model_path: Path = Path("retinanet_cargo_shipping_labels/version_2/checkpoints/epoch=23-step=24360.ckpt")
    dataset_name: str = "merged_variant"

    model_root: Path = Path().absolute() / "lightning_logs" / model_path
    images_root: Path = Path().absolute() / "data" / "datasets" / "coco" / dataset_name
    dataset_root: Path = images_root

    # threshold for detection matching between predictions and ground truth
    # if the iou is greater than this threshold, the detections areas are
    # considered matching candidates and (might) go into the calculation
    # for our confusion matrix
    iou_thres: float = 0.05

    device: torch.device = utils.get_computation_device()
    print(f"Calculating on {device}")

    detector = ObjectDetector.load_from_checkpoint(checkpoint_path=model_root, map_location=device)
    print(detector)

    dataset_01: CocoDetection = CocoDetection(root=images_root / "images" / "ds1",
                                              annFile=str(dataset_root / "annotations" / "instances_ds1.json"),
                                              transforms=None)

    dataset_02: CocoDetection = CocoDetection(root=images_root / "images" / "ds2",
                                              annFile=str(dataset_root / "annotations" / "instances_ds2.json"),
                                              transforms=None)

    dataset: ConcatDataset = ConcatDataset([dataset_01, dataset_02])

    datasets = utils.partition_dataset(dataset, lengths=(0.7, 0.2, 0.1))

    dm: DataModule = DataModule(Path(),
                                img_size=(0, 0),
                                batch_size=16,
                                train_transforms=None,
                                **datasets,
                                collate_fn=collate_fn)

    label_dict: bidict[int, str] = bidict({
        0: "Background",
        1: "Cargo",
        2: "Label"
    })

    testset: Subset = dm.test_dataset

    total_labels: int = 0
    for idx in testset.indices:
        _, target = testset.dataset[idx]
        total_labels += len(target)
    print(total_labels)

    image_dir: Path = Path("data/datasets/coco/merged_variant/images/ds1")
    image_paths: list[Path] = (
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg")) +
        list(image_dir.glob("*.png"))
    )

    saved: list[Path] = save_random_inference_samples(
        image_paths=image_paths,
        model=detector,
        device=device,
        label_dict=label_dict,
        output_dir=Path("tmp/random_inference"),
        num_images=10,
        conf_thres=0.1,
        model_image_size=None,
    )
    print(saved)

    results: ResultsType = []
    for img, target in tqdm(testset, total=len(testset)):
        boxes, predictions, confidences = infer_image(
            image=img,
            model=detector,
            device=device,
            label_dict=label_dict,
            conf_thres=0.7,
            model_image_size=None,
            iou_thres=iou_thres,
            nms_class_restricted=True
        )

        predictions = [label_dict.inverse[pred] for pred in predictions]

        predicted: tuple[Any, ...] = (boxes, predictions, confidences)

        truth_boxes_xyxy = [coco_bbox_to_xyxy(list(map(float, t["bbox"]))) for t in target]
        truth_labels = [t["category_id"] for t in target]
        truth: tuple[Any, ...] = (truth_boxes_xyxy, truth_labels)

        results.append((predicted, truth))

    tp, fp, fn, num_samples, matched_categories = compute_detection_counts(results, iou_thres=iou_thres)

    print(f"True positive rate: {tp / total_labels}")
    print(f"False positive rate: {fp / total_labels}")
    print(f"False negative rate: {fn / total_labels}")
    print(f"Num samples: {num_samples}")

    y_pred = list(flatten_generator([cat.predicted for cat in matched_categories]))
    y_true = list(flatten_generator([cat.truth for cat in matched_categories]))

    assert len(y_pred) == len(y_true)

    cm = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix:\n{cm}")


if __name__ == "__main__":
    main()

