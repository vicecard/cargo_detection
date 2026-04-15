import multiprocessing
from pathlib import Path

import torch

import torchvision.transforms.v2 as T
from torchvision.datasets.coco import CocoDetection

from machine_learning.object_detection.datamodule import DataModule
from machine_learning.object_detection.object_detector import ObjectDetector
from machine_learning.object_detection.evaluation import evaluate_model
from machine_learning.object_detection.transformations.coco import collate_fn, base_transforms
import machine_learning.object_detection.utils as utils


def main(model_path: Path, dataset_name: str, output_path: Path) -> None:
    device = utils.get_computation_device()
    print(f"Calculating on {device}")

    model_root: Path = Path().absolute() / "lightning_logs" / model_path
    images_root: Path = Path(dataset_name)
    dataset_root: Path = images_root

    detector = ObjectDetector.load_from_checkpoint(checkpoint_path=model_root, map_location=device)
    print(detector)

    transform_fns: T.Transform = T.Compose(
        [
            base_transforms,
            T.ToImage(),
            T.ConvertImageDtype(torch.float32),
            T.SanitizeBoundingBoxes(),
        ]
    )
    dataset: CocoDetection = CocoDetection(root=images_root / "images" / "default",
                                           annFile=str(dataset_root / "annotations" / "instances_default.json"),
                                           transforms=transform_fns)

    datasets = utils.partition_dataset(dataset, lengths=(0.7, 0.2, 0.1))

    dm = DataModule(Path(),
                    img_size=(0, 0),
                    batch_size=16,
                    train_transforms=None,
                    **datasets,
                    collate_fn=collate_fn)

    label_dict: dict[int, str] = {
        0: "Background",
        1: "Cargo",
        2: "Label"
    }

    print(evaluate_model(dm,
                         detector,
                         output_path,
                         (0, 0),
                         16,
                         conf_thres=0.10,
                         iou_thres=0.05,
                         device=device,
                         label_dict=label_dict))


if __name__ == "__main__":
    multiprocessing.set_start_method(method="spawn")

    output_path: Path = Path().absolute() / "evaluation_images" / "tuning_10_conf_10_iou_5"
    main(model_path="model/version_0/checkpoints/XXX.ckpt",
         dataset_name="data/datasets/coco/cargo_shipping_labels",
         output_path=output_path)

    print("Done!")
