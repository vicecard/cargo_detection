from pathlib import Path
from collections.abc import Iterator
from typing import Iterable, Callable, Any
from tqdm import tqdm
from PIL import Image

from dataclasses import dataclass, field
from bidict import bidict
import numpy as np

from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.dataset import Subset

from torchvision.datasets.coco import CocoDetection
from torchvision.ops import box_iou

#from machine_learning.object_detection.inference import infer_image
from machine_learning.object_detection.datamodule import DataModule
from machine_learning.object_detection.object_detector import ObjectDetector
from machine_learning.object_detection.transformations.coco import collate_fn

from machine_learning.object_detection.data_preparation import img_to_tensor
from machine_learning.object_detection.model_helper import decode_output

import machine_learning.object_detection.utils as utils


# define some useful type aliases
type ResultType = tuple[tuple[Any, ...], tuple[Any, ...]]
type ResultsType = list[ResultType]


@dataclass
class MatchedCategories:
    predicted: list[int] = field(default_factory=list)
    truth: list[int] = field(default_factory=list)
    ious: list[float] = field(default_factory=list)


def match_categories(result: ResultType, ious: torch.Tensor, iou_thres: float) -> MatchedCategories:
    values, indices = torch.max(ious, dim=1)

    matched_categories: MatchedCategories = MatchedCategories()

    for pred_idx, truth_idx in enumerate(indices):
        iou: float = values[pred_idx].item()
        if iou < iou_thres:
            continue

        matched_categories.predicted.append(result[0][1][pred_idx])
        matched_categories.truth.append(result[1][1][truth_idx])
        matched_categories.ious.append(iou)

    return matched_categories


def generate_matched_categories(detections: ResultsType, iou_thres: float) -> list[MatchedCategories]:
    matches: list[MatchedCategories] = []

    for result in detections:
        bbs_pred_coords: list[list[float]] = [list(map(float, box)) for box in result[0][0]]
        bbs_truth_coords: list[list[float]] = [box for box in result[1][0]]

        bbs_preds: torch.Tensor = torch.tensor(bbs_pred_coords)
        bbs_truths: torch.Tensor = torch.tensor(bbs_truth_coords)

        if bbs_preds.shape[0] == 0 or bbs_truths.shape[0] == 0:
            continue

        ious: torch.Tensor = box_iou(bbs_preds, bbs_truths)
        matches.append(match_categories(result, ious, iou_thres=iou_thres))

    return matches


# isn't that already factored out into a submodule?
def flatten_generator[T](inpt: Iterable[T]) -> Iterator[T]:
    for item in inpt:
        if isinstance(item, Iterable):
            yield from flatten_generator(item)
        else:
            yield item


def translate_bbox(bbox: list[float], scale_factors: tuple[int, int]) -> list[float]:
    s_w, s_h = scale_factors
    x, y, w, h = bbox
    return [x * s_w, y * s_h, (x + w) * s_w, (y + h) * s_h]


def filter_results(results: ResultsType, compare_fn: Callable[[int, int], bool]) -> ResultsType:
    return list(filter(lambda e: len(e[0]) > 0 and len(e[1]) > 0 and compare_fn(len(e[0][1]), len(e[1][1])), results))


def infer_image(img: Image.Image,
                model: ObjectDetector,
                device: torch.device,
                label_dict: dict[int, str],
                img_size: tuple[int, int],
                conf_thres: float,
                iou_thres: float = 0.05,
                nms_class_restricted: bool = False) -> tuple[list[list[int]], list[str], list[float]]:
    # return data structure:
    # ([bboxes], [label names], [confidences])

    # Suppress DeprecationWarning
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    img: np.ndarray = np.array(img.resize(img_size, resample=Image.BILINEAR)) / 255.
    img: torch.Tensor = img_to_tensor(img)

    # Move input tensor to same device as model
    img = img.to(device)

    # make model predictions
    # model expects batches of images, hence the unsqueeze()
    outputs = model(img.unsqueeze(0))
    output = outputs[0]  # model returns list of results

    # decode output of model and apply
    # thresholding values
    bbs, confs, labels = decode_output(output,
                                       label_dict,
                                       to_list=False,
                                       iou_thres=iou_thres,
                                       nms_class_restricted=nms_class_restricted)
    valid_detections = confs > conf_thres
    bbs = bbs[valid_detections].tolist()
    confs = confs[valid_detections].tolist()
    labels = labels[valid_detections].tolist()

    return bbs, labels, confs


def main() -> None:
    """Main function to run performance metrics calculations."""
    print("Calculating performance metrics...")

    model_path: Path = Path("retinanet_cargo_shipping_labels/version_2/checkpoints/epoch=23-step=24360.ckpt")
    dataset_name: str = "merged_variant"

    model_root: Path = Path().absolute() / "lightning_logs" / model_path
    images_root: Path = Path().absolute() / "data" / "datasets" / "coco" / dataset_name
    dataset_root: Path = images_root

    img_size: tuple[int, int] = (1920, 1080)
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
                                img_size=img_size,
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
    # get original image dimensions
    # to compute scaling factor
    w, h = testset[0][0].size

    s_w = img_size[0] / w
    s_h = img_size[1] / h

    results: ResultsType = []
    for img, target in tqdm(testset):
        # TODO: adjust img_size properly! Boxes get aligned incorrectly otherwise.
        # (Our images are of size (1920, 1080) _and_ (1080, 1920)
        boxes, predictions, confidences = infer_image(img, detector, device, label_dict, img_size, conf_thres=0.1)
        predictions = [label_dict.inverse[pred] for pred in predictions]
        predicted: tuple[Any, ...] = (boxes, predictions, confidences)
        truth: tuple[Any, ...] = ([translate_bbox(t["bbox"], (s_w, s_h)) for t in target],
                                  [t["category_id"] for t in target])
        results.append((predicted, truth))

    num_samples: int = len(results)

    matched_detections: ResultsType = filter_results(results, lambda p, t: p == t)
    num_matched_detections: int = len(matched_detections)
    print(f"Ratio of matched detections: {num_matched_detections} / {num_samples}")

    mismatched_detections: ResultsType = filter_results(results, lambda p, t: p != t)
    num_mismatched_detections: int = len(mismatched_detections)
    print(f"Ratio of mismatched detections: {num_mismatched_detections} / {num_samples}")

    assert num_mismatched_detections + num_matched_detections == num_samples

    fp_detections: ResultsType = filter_results(results, lambda p, t: p > t)
    fn_detections: ResultsType = filter_results(results, lambda p, t: p < t)

    num_fp_detections: int = len(fp_detections)
    num_fn_detections: int = len(fn_detections)
    print(f"Ratio of false positives: {num_fp_detections} / {num_samples}")
    print(f"Ratio of false negatives: {num_fn_detections} / {num_samples}")

    all_matched_categories: list[MatchedCategories] = generate_matched_categories(matched_detections, iou_thres)
    all_matched_categories.extend(generate_matched_categories(mismatched_detections, iou_thres))

    y_pred: list[int] = list(flatten_generator([cat.predicted for cat in all_matched_categories]))
    y_true: list[int] = list(flatten_generator([cat.truth for cat in all_matched_categories]))

    assert len(y_pred) == len(y_true)

    cm: np.ndarray = confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix:\n{cm}")


if __name__ == "__main__":
    main()
