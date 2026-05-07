from kedro.pipeline import Pipeline, node
from .nodes import build_image_manifest


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=build_image_manifest,
                inputs=[
                    "cleaned_dataset",      
                    "params:sprites_root",
                    "params:library_roots",
                ],
                outputs="image_manifest",
                name="build_image_manifest_node",
            )
        ]
    )