from kedro.pipeline import Pipeline, node
from .nodes import train_ae_node

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            func=train_ae_node,
            inputs=["train_metadata", "val_metadata", "params:ae"],
            outputs=["ae_history", "ae_result_img"],
            name="train_ae_node",
        )
    ])