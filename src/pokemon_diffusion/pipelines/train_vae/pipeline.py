from kedro.pipeline import Pipeline, node
from .nodes import train_vae_node

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_vae_node,
            inputs=["train_metadata", "val_metadata", "params:vae"],
            outputs=["vae_history", "vae_result_img"],
            name="train_vae_node",
        )
    ])