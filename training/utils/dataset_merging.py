from pathlib import Path
from collections import Counter

from datumaro import Dataset, AnnotationType, Categories, TQDMProgressReporter
from machine_learning.object_detection.datasets.utils import DatasetInfo, merge_datasets



def count_labels(dataset: Dataset) -> Counter:
    label_categories: Categories = dataset.categories()[AnnotationType.label]
    label_names: list[str] = [label.name for label in label_categories.items]
    counter: Counter = Counter(label_names[ann.label] for item in dataset for ann in item.annotations)

    return counter


def merge(ds1_path: Path, ds2_path: Path) -> Dataset:
    ds1: Dataset = Dataset.import_from(str(ds1_path), format="coco")
    ds2: Dataset = Dataset.import_from(str(ds2_path), format="coco")

    print(f"number of items in {ds1_path}: {len(ds1)}")
    print(f"number of items in {ds2_path}: {len(ds2)}")

    print("Categories of first dataset:")
    print(ds1.categories())
    print("Categories of second dataset:")
    print(ds2.categories())
    print(f"Number of labels in first dataset: {count_labels(ds1)}")
    print(f"Number of labels in second dataset: {count_labels(ds2)}")

    label_info: list[str] = ["cargo", "label"]

    dataset_infos: list[DatasetInfo] = [
        DatasetInfo(path=ds1_path, format="coco", label_info=label_info),
        DatasetInfo(path=ds2_path, format="coco", label_info=label_info)
    ]

    return merge_datasets(dataset_infos, merge_policy="union")


def main(ds1_path: Path, ds2_path: Path, output_dir: Path) -> None:
    merged: Dataset = merge(ds1_path, ds2_path)

    print(f"Number of merged items: {len(merged)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.export(str(output_dir), format="coco_instances", save_media=True, progress_reporter=TQDMProgressReporter())

    print(f"Merged dataset saved to {output_dir}")
    merged_dataset: Dataset = Dataset.import_from(str(output_dir), format="coco_instances")
    print(f"Number of items in merged dataset: {len(merged_dataset)}")
    print(f"Number of labels in merged dataset: {count_labels(merged_dataset)}")


if __name__ == "__main__":
    main(Path("data/datasets/coco/cargo_shipping_labels"),
         Path("data/datasets/coco/fiverr"),
         Path("tmp/merged"))