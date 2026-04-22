import multiprocessing
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

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
    #dataset: CocoDetection = CocoDetection(root=images_root / "images" / "default",
    #                                       annFile=str(dataset_root / "annotations" / "instances_default.json"),
    #                                       transforms=transform_fns)

    dataset_01: CocoDetection = CocoDetection(root=images_root / "images" / "ds1",
                                              annFile=str(dataset_root / "annotations" / "instances_ds1.json"),
                                              transforms=transform_fns)

    dataset_02: CocoDetection = CocoDetection(root=images_root / "images" / "ds2",
                                              annFile=str(dataset_root / "annotations" / "instances_ds2.json"),
                                              transforms=transform_fns)

    dataset: ConcatDataset = ConcatDataset([dataset_01, dataset_02])

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
                         32,
                         conf_thres=0.10,
                         iou_thres=0.05,
                         device=device,
                         label_dict=label_dict))


if __name__ == "__main__":
    multiprocessing.set_start_method(method="spawn")

    output_path: Path = Path().absolute() / "evaluation_images" / "version_03"
    main(model_path=Path("retinanet_cargo_shipping_labels/version_2/checkpoints/epoch=23-step=24360.ckpt"),
         dataset_name="data/datasets/coco/merged_variant",
         output_path=output_path)

    print("Done!")

    # best model:
    # 1)
    # lightning_logs/retinanet_cargo_shipping_labels/version_0/checkpoints/epoch=20-step=17052.ckpt
    # 0.46791714429855347
    # {'map': tensor(0.5806), 'map_50': tensor(0.7923), 'map_75': tensor(0.6360), 'map_small': tensor(0.0825), 'map_medium': tensor(0.5288), 'map_large': tensor(0.7317), 'mar_1': tensor(0.1758), 'mar_10': tensor(0.6079), 'mar_100': tensor(0.6794), 'mar_small': tensor(0.1474), 'mar_medium': tensor(0.7070), 'mar_large': tensor(0.8338), 'map_per_class': tensor(-1.), 'mar_100_per_class': tensor(-1.), 'classes': tensor([1, 2], dtype=torch.int32)}
    # 2)
    # lightning_logs/retinanet_cargo_shipping_labels/version_1/checkpoints/epoch=26-step=12582.ckpt
    # 0.3779895603656769
    # {'map': tensor(0.4250), 'map_50': tensor(0.5598), 'map_75': tensor(0.4488), 'map_small': tensor(0.1510), 'map_medium': tensor(0.2038), 'map_large': tensor(0.5157), 'mar_1': tensor(0.2682), 'mar_10': tensor(0.4436), 'mar_100': tensor(0.4989), 'mar_small': tensor(0.1884), 'mar_medium': tensor(0.2306), 'mar_large': tensor(0.5866), 'map_per_class': tensor(-1.), 'mar_100_per_class': tensor(-1.), 'classes': tensor([1, 2], dtype=torch.int32)}
    # 3)
    # lightning_logs/retinanet_cargo_shipping_labels/version_2/checkpoints/epoch=23-step=24360.ckpt
    # 0.5916559100151062
    # {'map': tensor(0.6503), 'map_50': tensor(0.8531), 'map_75': tensor(0.7197), 'map_small': tensor(0.1913), 'map_medium': tensor(0.5523), 'map_large': tensor(0.7504), 'mar_1': tensor(0.2727), 'mar_10': tensor(0.6585), 'mar_100': tensor(0.7388), 'mar_small': tensor(0.2495), 'mar_medium': tensor(0.7226), 'mar_large': tensor(0.8471), 'map_per_class': tensor(-1.), 'mar_100_per_class': tensor(-1.), 'classes': tensor([1, 2], dtype=torch.int32)}
