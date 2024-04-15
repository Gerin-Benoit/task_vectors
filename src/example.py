import torch
from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments

# Config
datasets = ['DTD', 'EuroSAT']
model = 'ViT-L-14'
args = parse_arguments()
args.data_location = 'datasets/data'
args.model = model
args.save = f'checkpoints/{model}'
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'

# Create the task vectors
task_vectors = [
    TaskVector(pretrained_checkpoint, f'checkpoints/{model}/{dataset}/finetuned.pt')
    for dataset in datasets
]
# Sum the task vectors
task_vector_sum = sum(task_vectors)
# Apply the resulting task vector
image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=0.8)
# Evaluate
print("Sources ZS:")
for dataset in datasets:
    print(dataset)
    eval_single_dataset(task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=0), dataset, args)

print("Sources finetuned:")
for i, dataset in enumerate(datasets):
    print(dataset)
    eval_single_dataset(task_vectors[i].apply_to(pretrained_checkpoint, scaling_coef=1), dataset, args)

print("Sources with task-vectors combination:")
for dataset in datasets:
    print(dataset)
    eval_single_dataset(image_encoder, dataset, args)

print("ImageNet ZS:")
args.data_location = "datasets/data"
eval_single_dataset(task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=0), "ImageNet", args)

print("ImageNet with task vectors combination:")

eval_single_dataset(image_encoder, "ImageNet", args)

