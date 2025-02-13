import yaml
import argparse
import numpy as np
import pandas as pd
from IPython.display import HTML, display
import os

import transformers
import evaluate
import data


class Training:
    def __init__(self, config_file, extra_args):
        with open(config_file, "r") as file:
            configs = yaml.safe_load(file)
        self.configs = configs
        self.extra_args = extra_args

    @staticmethod
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return evaluate.load("accuracy").compute(
            predictions=predictions, references=labels
        )

    def run(self):
        num_train_epochs = self.extra_args.num_train_epochs
        resume_from_checkpoint = self.extra_args.resume_from_checkpoint
        load_from_checkpoint = self.extra_args.load_from_checkpoint

        model_config = self.configs["model_config"]
        checkpoint = model_config["checkpoint"]

        dataset_info = self.configs["dataset_info"]
        dataset_info["checkpoint"] = checkpoint

        training_config = self.configs["training_config"]
        training_config["num_train_epochs"] = num_train_epochs

        # dataset
        img_cls_dataset = data.ImageClassificationDataset(**dataset_info)
        class_names = img_cls_dataset.class_names
        id2label = {k: v for k, v in enumerate(class_names)}
        label2id = {k: v for v, k in enumerate(class_names)}
        data_collator = img_cls_dataset.data_collator
        train_dataset, eval_dataset, test_dataset = img_cls_dataset.prepare_datasets()

        # model
        img_cls_model = transformers.AutoModelForImageClassification.from_pretrained(
            load_from_checkpoint if load_from_checkpoint else checkpoint,
            num_labels=len(class_names),
            id2label=id2label,
            label2id=label2id,
        )
        # training and testing the model
        training_args = transformers.TrainingArguments(**training_config)

        trainer = transformers.Trainer(
            model=img_cls_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

        # ---train
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        # ---test
        test_results = trainer.predict(test_dataset=test_dataset)
        test_metrics = pd.DataFrame(
            test_results.metrics, index=[0], columns=["test_loss", "test_accuracy"]
        )
        test_metrics.rename(
            columns={"test_loss": "Test Loss", "test_accuracy": "Test Accuracy"},
            inplace=True,
        )
        output_dir = self.configs["training_config"]["output_dir"]
        test_metrics.to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)
        display(HTML(test_metrics.to_html(index=False)))


if __name__ == "__main__":
    config_file = "./config.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--load_from_checkpoint", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)

    extra_args = parser.parse_args()

    training_img_cls = Training(config_file=config_file, extra_args=extra_args)
    training_img_cls.run()
