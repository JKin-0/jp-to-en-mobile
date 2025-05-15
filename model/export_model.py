import torch
import executorch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from jaen_cnn import CNN

model = CNN(num_classes=700)
model.load_state_dict(torch.load("jaenCNN_v3.pth", map_location="cpu"))
model.eval()

# Define the graph for the model
sample_input = torch.randn(1, 1, 64, 64) # Batch size, channels (grayscale), image width and height

# Export and lower the model for backend
et_program = to_edge_transform_and_lower(
    torch.export.export(model, (sample_input,)),
                      partitioner=[XnnpackPartitioner()]
    ).to_executorch()

with open("ja_xnnpack.et", "wb") as f:
    et_program.write_to_file(f)