from kedro.pipeline import Pipeline, node
from .nodes import image_manifest_node


def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=image_manifest_node,
                inputs=[
                    "cleaned_dataset",
                    "params:sprites_root",
                    "params:library_roots",
                ],
                outputs="final_dataset",
                name="image_manifest_node",
            ),
        ]
    )