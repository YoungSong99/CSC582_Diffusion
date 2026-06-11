from kedro.pipeline import Pipeline, node
from .nodes import train_conditional_sdvae_node

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_conditional_sdvae_node,
            inputs=["train_metadata", "val_metadata", "params:conditional_sdvae"],
            outputs=["conditional_sdvae_history", "conditional_sdvae_result_img"],
            name="train_conditional_sdvae_model",
        )
    ])