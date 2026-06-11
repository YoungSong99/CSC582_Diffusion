from kedro.pipeline import Pipeline, node
from .nodes import train_vqvae_node

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_vqvae_node,
            inputs=["train_metadata", "val_metadata", "params:vqvae"],
            outputs=["vqvae_history", "vqvae_result_img"],
            name="train_vqvae_model",
        )
    ])