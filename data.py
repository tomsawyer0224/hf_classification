# from torch.utils.data import DataLoader
import torch
import torchvision
import datasets, transformers, evaluate


class BaseDataset:
    def __init__(self, dataset_name):
        super().__init__()
        self.raw_dataset = datasets.load_dataset(dataset_name)

    @staticmethod
    def train_validation_test_split(dataset):
        """
        this method returns 3 datasets (train, validation, test) from single dataset
        """
        train_validation_and_test = dataset.train_test_split(test_size=0.1, seed=42)

        train_validation = train_validation_and_test["train"]
        test_dataset = train_validation_and_test["test"]

        train_validation = train_validation.train_test_split(test_size=0.11, seed=42)
        train_dataset = train_validation["train"]
        validation_dataset = train_validation["test"]

        return train_dataset, validation_dataset, test_dataset

    def generate_datasets(self):
        """
        this method returns 3 datasets: train, validation, test and 1 list class names
        args:
            dataset: hugging face dataset
        return:
            tuple (train, validation, test, class_names)
        """
        if hasattr(self.raw_dataset, "keys"):  # if DatasetDict
            ds_keys = self.raw_dataset.keys()
            has_validation = False
            has_test = False
            assert "train" in ds_keys, "'train' must be in DatasetDict.keys()"
            raw_train_dataset = self.raw_dataset["train"]
            class_names = (
                self.raw_dataset["train"].features[self.label_column_name].names
            )
            if "validation" in ds_keys:
                has_validation = True
                validation_dataset = self.raw_dataset["validation"]
            if "test" in ds_keys:
                has_test = True
                test_dataset = self.raw_dataset["test"]
            if not has_validation and not has_test:
                train_dataset, validation_dataset, test_dataset = (
                    self.train_validation_test_split(dataset=raw_train_dataset)
                )
            elif not has_validation and has_test:
                train_validation = raw_train_dataset.train_test_split(
                    test_size=0.1, seed=42
                )
                train_dataset = train_validation["train"]
                validation_dataset = train_validation["test"]
            elif has_validation and not has_test:
                train_test = raw_train_dataset.train_test_split(test_size=0.1, seed=42)
                train_dataset = train_test["train"]
                test_dataset = train_test["test"]
            else:
                train_dataset = raw_train_dataset
        else:  # single Dataset
            class_names = self.raw_dataset.features[self.label_column_name].names
            train_dataset, validation_dataset, test_dataset = (
                self.train_validation_test_split(dataset=self.raw_dataset)
            )
        return train_dataset, validation_dataset, test_dataset, class_names


class ImageClassificationDataset(BaseDataset):
    def __init__(
        self,
        dataset_name,
        image_column_name,
        label_column_name,
        checkpoint,
        # image_size = 224,
        # batch_size = 32
    ):
        super().__init__(dataset_name=dataset_name)
        self.image_column_name = image_column_name
        self.label_column_name = label_column_name
        self.image_processor = transformers.AutoImageProcessor.from_pretrained(
            checkpoint
        )
        # self.image_size = image_size
        # self.batch_size = batch_size
        (
            self.train_dataset,
            self.validation_dataset,
            self.test_dataset,
            self.class_names,
        ) = self.generate_datasets()

    def prepare_datasets(self):
        """
        this method will return 3 datasets: train, validation, test
        """
        # rename columns to 'image', 'label' for convenient
        train_dataset = self.train_dataset.rename_columns(
            {self.image_column_name: "image", self.label_column_name: "label"}
        )
        validation_dataset = self.validation_dataset.rename_columns(
            {self.image_column_name: "image", self.label_column_name: "label"}
        )
        test_dataset = self.test_dataset.rename_columns(
            {self.image_column_name: "image", self.label_column_name: "label"}
        )

        mean = self.image_processor.image_mean
        std = self.image_processor.image_std
        size = (
            self.image_processor.size["shortest_edge"]
            if "shortest_edge" in self.image_processor.size
            else (
                self.image_processor.size["height"],
                self.image_processor.size["width"],
            )
        )
        aug_transformation = torchvision.transforms.Compose(
            [
                torchvision.transforms.RandomResizedCrop(size),
                torchvision.transforms.RandomRotation(10),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

        normal_transformation = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean, std=std),
            ]
        )

        def _train_transform(examples):
            """
            args:
                examples: dict of list
            return:
                examples
            """
            pil_images_list = examples["image"]
            examples["pixel_values"] = [
                aug_transformation(pil_img.convert("RGB"))
                for pil_img in pil_images_list
            ]
            return examples

        def _val_transform(examples):
            pil_images_list = examples["image"]
            examples["pixel_values"] = [
                normal_transformation(pil_img.convert("RGB"))
                for pil_img in pil_images_list
            ]
            return examples

        train_dataset = train_dataset.map(
            function=_train_transform, batched=True, batch_size=128, num_proc=2
        )
        validation_dataset = validation_dataset.map(
            function=_val_transform, batched=True, batch_size=128, num_proc=2
        )
        test_dataset = test_dataset.map(
            function=_val_transform, batched=True, batch_size=128, num_proc=2
        )
        # remove unusual columns
        column_names = train_dataset.column_names
        column_names.remove("pixel_values")
        column_names.remove("label")

        train_dataset = train_dataset.remove_columns(column_names)
        validation_dataset = validation_dataset.remove_columns(column_names)
        test_dataset = test_dataset.remove_columns(column_names)

        return train_dataset, validation_dataset, test_dataset

    @staticmethod
    def data_collator(examples):
        """
        args:
            examples: list of dict
        return:
            dict of torch tensors
        """
        images = [exam["pixel_values"] for exam in examples]
        labels = [exam["label"] for exam in examples]
        image_tensor = torch.tensor(images)
        label_tensor = torch.tensor(labels)
        return {"pixel_values": image_tensor, "labels": label_tensor}
