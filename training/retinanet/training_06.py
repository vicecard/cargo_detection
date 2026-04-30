from pathlib import Path
from typing import Callable, Sequence

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.dataset import Subset

import torchvision.transforms.v2 as T
from torchvision.datasets.coco import CocoDetection

from pytorch_lightning.callbacks import Callback

from machine_learning.object_detection.utils import partition_dataset, canvas_size
from machine_learning.object_detection.object_detector import ObjectDetector
from machine_learning.object_detection.datamodule import DataModule
from machine_learning.object_detection.training import HyperParameters, train
from machine_learning.object_detection.transformations.coco import base_transforms, collate_fn
from machine_learning.object_detection.type_aliases import DataSample, JsonDict, TargetDict



type CollateFn = Callable[[Sequence[DataSample]], tuple[list[torch.Tensor], list[TargetDict]]]


def setup_training(datasets: dict[str, Subset],
                   num_classes: int,
                   collate_fn: CollateFn) -> tuple[ObjectDetector, DataModule, HyperParameters]:
    params = HyperParameters(
        network="retinanet",
        num_classes=num_classes,
        n_epochs=150,
        batch_size=8,
        early_stopping=True,
        learning_rate=0.001
    )
    model = ObjectDetector(
        num_classes=params.num_classes,
        network=params.network,
        batch_size=params.batch_size,
        learning_rate=params.learning_rate,
        weight_decay=params.weight_decay,
        momentum=params.momentum,
        pretrained=False,
        use_weights_parameter=True,
        optimizer_class=torch.optim.SGD,
        optimizer_parameters={'momentum': params.momentum}
    )
    dm = DataModule(
        Path(),
        img_size=(0, 0),
        batch_size=params.batch_size,
        train_transforms=None,
        **datasets,
        collate_fn=collate_fn,
    )
    print(f"Setting parameters: {params}")

    return model, dm, params


def main() -> None:
    save_path: Path = Path().absolute() / "data" / "datasets" / "coco" / "merged_variant"

    # create dataset
    transform_fns: T.Transform = T.Compose(
        [
            base_transforms,
            T.RandomApply([
                T.RandomAffine(
                    degrees=15,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=10,
                )
            ], p=0.1),
            T.RandomAutocontrast(p=0.1),
            T.RandomPerspective(distortion_scale=0.3, p=0.1),
            T.ToImage(),
            T.ConvertImageDtype(torch.float32),
            T.SanitizeBoundingBoxes(),
        ]
    )

    dataset_01: CocoDetection = CocoDetection(root=save_path / "images" / "ds1",
                                              annFile=str(save_path / "annotations" / "instances_ds1.json"),
                                              transforms=transform_fns)

    dataset_02: CocoDetection = CocoDetection(root=save_path / "images" / "ds2",
                                              annFile=str(save_path / "annotations" / "instances_ds2.json"),
                                              transforms=transform_fns)

    dataset: ConcatDataset = ConcatDataset([dataset_01, dataset_02])
    print(f"{len(dataset)} items in dataset.")

    # filter dataset and partition it into train/validation/test subsets
    datasets = partition_dataset(dataset, lengths=(0.7, 0.2, 0.1))

    # +1 for background class
    num_classes = len(dataset_01.coco.cats) + 1
    assert num_classes == len(dataset_02.coco.cats) + 1
    print(dataset_01.coco.cats)
    model, dm, params = setup_training(datasets, num_classes, collate_fn)
    print(f"Model architecture:\n{model}")

    # only medium precision needed (boosts training speed a little)
    torch.set_float32_matmul_precision("medium")

    checkpoint_callback: Callback = train(model, dm, params, model_log_name="retinanet_cargo_shipping_labels")
    with open("best_model.txt", "w") as f:
        f.write(checkpoint_callback.best_model_path)
        f.write("\n")
        f.write(str(checkpoint_callback.best_model_score.item()))


if __name__ == "__main__":
    main()
    print("Done!")
