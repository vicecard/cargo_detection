from machine_learning.object_detection.object_detector import ObjectDetector
from data_io import download_dataset


def main():
    print("Hello from cargo-detection!")
    print(ObjectDetector.__dict__)
    print(download_dataset.run)


if __name__ == "__main__":
    main()
