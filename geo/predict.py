from training import LightningFcn
import dataset
import os
import argparse
import torch
from skimage import measure
from collections import defaultdict
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from dataset import GeoSetFromFolder

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="saved model")
parser.add_argument("--dataset", type=str, default="./data", help="dataset")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--nocuda", action='store_true', help="no cuda used")
parser.add_argument("--nworkers", type=int, default=4, help="number of workers")
parser.add_argument("--output_file", type=str, default="pred.json", help="output file")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Testing on {device}')

def predict(net, loader):
    net.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            output = net(x)
    
    print("Done")
    return

def extract_centroids(pred, bg):
    conn_comp=measure.label(pred, background=bg)
    object_dict=defaultdict(list) #Keys are the indices of the connected components and values are arrrays of their pixel coordinates 
    for (x,y),label in np.ndenumerate(conn_comp):
            if label != bg:
                object_dict[label].append([x,y])
    # Mean coordinate vector for each object, except the "0" label which is the background
    centroids={label: np.mean(np.stack(coords),axis=0) for label,coords in object_dict.items()}
    object_sizes={label: len(coords) for label,coords in object_dict.items()}
    return centroids, object_sizes

def filter_large_objects(centroids,object_sizes, max_size):
    small_centroids={}
    for label,coords in centroids.items():
            if object_sizes[label] <= max_size:
                small_centroids[label]=coords
    return small_centroids


if __name__ == "__main__":


    # Define test dataset
    test_set = GeoSetFromFolder(
                root=args.dataset,
                dataset='test',
                transform=transforms.ToTensor(),
                output_size=(480, 640)
            )

    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.nworkers)
    # Load model from checkpoint
    model = LightningFcn.load_from_checkpoint(args.model).to(device)

    predict(model, test_loader)

    
