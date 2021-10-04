from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from msrest.authentication import ApiKeyCredentials
from typing import List, Dict, Union
from custom_vision.utils import (
    generate_tags,
    generate_image_list,
    upload_images,
    train_model,
    get_iteration,
    model_performance,
    publish_model,
    unpublish_model,
    export_model,
)


class CustomVisionTrainer:
    def __init__(self, training_key, endpoint, project_name):
        # Authenticate:
        self.credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
        # Create trainer instance:
        self.trainer = CustomVisionTrainingClient(
            endpoint=endpoint, credentials=self.credentials
        )
        # Get project:
        projects = self.trainer.get_projects()
        project = [p for p in projects if p.name == project_name]
        if len(project) == 1:
            self.project = project[0]

            print(
                "Resuming project\n---------------\nName: %s\nID: %s"
                % (self.project.name, self.project.id)
            )
        else:
            # Create a new project
            self.project = self.trainer.create_project(name=project_name)

            print(
                "Project created\n---------------\nName: %s\nID: %s"
                % (self.project.name, self.project.id)
            )

    def generate_tags(self, tags: Union[str, List[str]]):
        if isinstance(tags, str):
            tags = [tags]
        self.tags = generate_tags(
            project=self.project,
            trainer=self.trainer,
            tags_list=tags,
        )

    def add_images(
        self,
        path: str,
        tag_names: Union[str, List[str]],
    ):
        assert hasattr(
            self, "tags"
        ), "No tags found. Need to generate tags before adding images."
        if isinstance(tag_names, str):
            tag_names = [tag_names]
        image_list = generate_image_list(
            path=path,
            tags=self.tags,
            tag_names=tag_names,
        )
        upload_images(
            project=self.project,
            trainer=self.trainer,
            image_list=image_list,
        )

    def train(self, params: Dict = None):
        train_model(
            project=self.project,
            trainer=self.trainer,
            params=params,
        )

    def get_performance(self, iteration_name: str = "Iteration 1"):
        iteration = get_iteration(self.project, self.trainer, iteration_name)
        perf = model_performance(
            project=self.project,
            trainer=self.trainer,
            iteration=iteration,
        )
        return perf

    def publish(
        self,
        iteration_name: str,
        prediction_id: str,
        published_name: str,
    ):
        iteration = get_iteration(self.project, self.trainer, iteration_name)
        publish_model(
            project=self.project,
            trainer=self.trainer,
            iteration=iteration,
            prediction_id=prediction_id,
            published_name=published_name,
        )

    def unpublish(self, iteration_name):
        iteration = get_iteration(self.project, self.trainer, iteration_name)
        unpublish_model(self.project, self.trainer, iteration)

    def export(
        self,
        iteration_name: str,
        platform: str,
        flavor: str,
        **kwargs,
    ):
        iteration = get_iteration(
            self.project,
            self.trainer,
            iteration_name,
        )
        uri = export_model(
            project=self.project,
            trainer=self.trainer,
            iteration=iteration,
            platform=platform,
            flavor=flavor,
            **kwargs,
        )
        return uri
