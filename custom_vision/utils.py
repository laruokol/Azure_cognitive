from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateBatch,
    ImageFileCreateEntry,
)
from typing import Any, List
import os
import re
import math
import numpy as np
import time


def get_domains(self, trainer):
    """
    get_domains [summary]

    Args:
        trainer ([type]): [description]
    """
    domain_list = trainer.get_domains()
    domain_list = [d.as_dict() for d in domain_list]
    # domain_dict = {d['name']:d for d in domain_list}

    self.domains = domain_list


def generate_tags(
    project: Any,
    trainer: CustomVisionTrainingClient,
    tags_list: List[str],
):
    tags = {
        tag: trainer.create_tag(project_id=project.id, name=tag) for tag in tags_list
    }
    return tags


def generate_image_list(
    path: str,
    tags: List[Any],
    tag_names: List[str],
):
    image_list = []
    imgs = os.listdir(path)
    for image in imgs:
        loc = os.path.join(path, image)
        with open(loc, "rb") as image_contents:
            image_list.append(
                ImageFileCreateEntry(
                    name=image,
                    contents=image_contents.read(),
                    tag_ids=[tags.get(tag).id for tag in tag_names],
                )
            )


def upload_images(
    trainer: CustomVisionTrainingClient,
    project: Any,
    image_list: List[Any],
):
    # Split list to batches:
    batches = np.array_split(image_list, math.ceil(len(image_list) / 50))

    print("Uploading images in batches...")
    # Upload each batch:
    for n, images in enumerate(batches, 1):
        print("batch %d/%d" % (n, len(batches)))
        upload_result = trainer.create_images_from_files(
            project.id, ImageFileCreateBatch(images=images)
        )
        if not upload_result.is_batch_successful:
            print("Image batch upload failed.")
            for image in upload_result.images:
                print("Image status: ", image.status)
            exit(-1)

    print("Images uploaded.")


def train_model(
    project: Any,
    trainer: CustomVisionTrainingClient,
    params: Dict = None,
):

    if params is None:
        print("Using default parameters")
        params = {}
    else:
        print("TRAINING PARAMETERS:")
        print("--------------------")
        {print(k.replace("_", " ").capitalize(), ":", v) for k, v in params.items()}

    # TRAINING:
    print("\nTRAINING PROCESS:")
    print("-----------------")
    print("Training...")
    iteration = trainer.train_project(
        project_id=project.id,
        training_type=params.get("training_type"),
        reserved_budget_in_hours=params.get("reserved_budget_in_hours"),
        force_train=params.get("force_train"),
        notification_email_address=params.get("notification_email_address"),
    )
    while iteration.status != "Completed":
        iteration = trainer.get_iteration(project.id, iteration.id)
        print("Training status: " + iteration.status)
        time.sleep(10)
    print("Done!")


def get_iteration(
    project: Any,
    trainer: CustomVisionTrainingClient,
    iteration_name: str,
):
    iteration = [
        i for i in trainer.get_iterations(project.id) if i.name == iteration_name
    ]
    assert len(iteration) > 0, f"Iteration {iteration_name} not found."
    return iteration[0]


def model_performance(
    project: Any,
    trainer: CustomVisionTrainingClient,
    iteration: Any,
):
    perfs = trainer.get_iteration_performance(
        project_id=project.id, iteration_id=iteration.id
    )
    return perfs.as_dict()


def publish_model(
    project: Any,
    trainer: CustomVisionTrainingClient,
    iteration: Any,
    prediction_id: str,
    published_name: str = "custom_model",
    **kwargs,
):
    print(
        "Publishing model from iteration: %s\nPublished name: %s"
        % (iteration.id, published_name)
    )

    trainer.publish_iteration(
        project_id=project.id,
        iteration_id=iteration.id,
        publish_name=published_name,
        prediction_id=prediction_id,
        **kwargs,
    )


def unpublish_model(
    project: Any,
    trainer: CustomVisionTrainingClient,
    iteration: Any,
):
    print("Unpublishing model from iteration: %s" % (iteration.id))
    trainer.unpublish_iteration(project_id=project.id, iteration_id=iteration.id)


def export_model(
    project: Any,
    trainer: CustomVisionTrainingClient,
    iteration: Any,
    platform: str,
    flavor: str,
    **kwargs,
):

    exports = trainer.get_exports(
        project_id=project.id,
        iteration_id=iteration.id,
    )
    get_uri = []
    if len(exports) > 0:
        get_uri = [
            x.download_uri
            for x in exports
            if platform == x.platform and x.status == "Done"
        ]

    if len(get_uri) > 0:
        print("Model already exported.")
    else:
        export = trainer.export_iteration(
            project_id=project.id,
            iteration_id=iteration.id,
            platform=platform,
            flavor=flavor,
            **kwargs,
        )

        expd = export.as_dict()
        print("EXPORTING PROCESS:")
        print("------------------")
        print("Output platform:", expd["platform"])
        print("Status:", expd["status"])

        start = time.time()
        exports = trainer.get_exports(project.id, iteration.id)
        latest = len(exports) - 1
        exp = exports[latest]
        while exp.status != "Done":
            exp = trainer.get_exports(project.id, iteration.id)[latest]
            now = time.time()
            passed = round(now - start)
            time.sleep(10)

            assert passed > 300, "This is taking way too long. Something is not right."

        print("\nDone! ")
        print("Time passed: " + str(now) + "s")

    return exp.download_uri


def classify_images(
    project_id: str,
    predictor: CustomVisionPredictionClient,
    published_name: str,
    image_path: str = None,
    store: bool = False,
    **kwargs,
):

    # List images:
    if re.search(r"/$", image_path) is None:
        image_path += "/"
    images = os.listdir(image_path)
    images = [im for im in images if not re.match(r"^\.", im)]

    out = {}
    print("Predicting image:")
    for n, img in enumerate(images, 1):
        print("%d/%d: %s" % (n, len(images), img))

        with open(image_path + img, "rb") as image_contents:
            if store:
                results = predictor.classify_image(
                    project_id=project_id,
                    published_name=published_name,
                    image_data=image_contents.read(),
                    **kwargs,
                )
            else:
                results = predictor.classify_image_with_no_store(
                    project_id=project_id,
                    published_name=published_name,
                    image_data=image_contents.read(),
                    **kwargs,
                )

        out[img] = {pre.tag_name: pre.probability for pre in results.predictions}

    print("Done!")

    return out