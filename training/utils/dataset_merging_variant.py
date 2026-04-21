from pathlib import Path
from collections import Counter

from datumaro import Dataset, DatasetItem, AnnotationType, Categories, LabelCategories, Bbox, TQDMProgressReporter


KEEP_LABELS: set[str] = {"cargo", "label"}
OUTPUT_SUBSET: str = "merged"


def build_label_categories(label_names: list[str]) -> LabelCategories:
    categories: LabelCategories = LabelCategories()
    for name in label_names:
        categories.add(name)

    return categories


def count_labels(dataset: Dataset) -> Counter:
    label_categories: Categories = dataset.categories()[AnnotationType.label]
    label_names: list[str] = [label.name for label in label_categories.items]
    counter: Counter = Counter(label_names[ann.label] for item in dataset for ann in item.annotations)

    return counter


def filter_and_remap(dataset: Dataset, keep_labels: set[str], id_prefix: str, subset_name: str) -> Dataset:
    src_categories: Categories = dataset.categories()[AnnotationType.label]
    src_labels: list[str] = [label.name for label in src_categories.items]
    kept_labels: list[str] = [name for name in src_labels if name in keep_labels]
    label_to_new_id: dict[str, int] = {name: i for i, name in enumerate(kept_labels)}

    items: list[DatasetItem] = []
    for item in dataset:
        new_anns: list[Bbox] = []
        for ann in item.annotations:
            label_name: str = str(src_labels[ann.label])
            if label_name in keep_labels:
                new_ann: Bbox = ann.wrap(label=label_to_new_id[label_name])
                new_anns.append(new_ann)

        new_item: DatasetItem = item.wrap(
            id=f"{id_prefix}_{item.id}",
            subset=subset_name,
            annotations=new_anns,
        )
        items.append(new_item)

    return Dataset.from_iterable(
        items,
        categories={AnnotationType.label: build_label_categories(kept_labels)},
    )


def main(ds1_path: Path, ds2_path: Path) -> None:
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

    ds1_filtered: Dataset = filter_and_remap(ds1, KEEP_LABELS, "ds1", "ds1")
    ds2_filtered: Dataset = filter_and_remap(ds2, KEEP_LABELS, "ds2", "ds2")

    merged_items: list[DatasetItem] = list(ds1_filtered) + list(ds2_filtered)

    merged_labels: list[str] = [label.name for label in ds1_filtered.categories()[AnnotationType.label].items]
    merged: Dataset = Dataset.from_iterable(
        merged_items,
        categories={AnnotationType.label: build_label_categories(merged_labels)},
    )

    print(f"Number of merged items: {len(merged)}")

    output_dir: Path = Path("tmp/merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        merged.export(str(output_dir), format="coco_instances", save_media=True, progress_reporter=TQDMProgressReporter())
    except TypeError:
        print("Type error occurred. Trying to export as COCO format.")
        merged.export(str(output_dir), format="coco", save_media=True, progress_reporter=TQDMProgressReporter())

    print(f"Merged dataset saved to {output_dir}")
    merged_dataset: Dataset = Dataset.import_from(str(output_dir), format="coco")
    print(f"Number of items in merged dataset: {len(merged_dataset)}")
    print(f"Number of labels in merged dataset: {count_labels(merged_dataset)}")


if __name__ == "__main__":
    main(
        Path("data/datasets/coco/cargo_shipping_labels"),
        Path("data/datasets/coco/fiverr"),
    )