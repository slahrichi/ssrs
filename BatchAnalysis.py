"""BatchAnalysis.py
This script handles getting the key statistics for multiple models and turning
them into a Pandas array for easy analysis.

Results are saved as a CSV file with the output name you specify (so make sure
that you change this name).


TO USE:
* Edit the task/result path as appropriate.
* Edit the encoders to get results from.
* Ensure normalization is properly set up.
* Edit the file name to save the results into.

When in doubt, look for the #TODO flags


"""

import os
import pandas as pd
import numpy as np
import argparse
import copy

import torch

from sklearn.metrics import precision_recall_curve

from src import datasets
from models import encoders, decoders


def main():
    base_path = f"/scratch/saad/no-fine-tune/{args.task}/data-comparison/"
    data = ['64', '128', '256', '512', '1024']
    #encoders = [args.encoder]
    encoders = ['supervised', 'swav-imagenet', 'swav-climate+']
    for n in range(0,100,10):
        encoders.append(f"swav_climate+_ep{n}")
    encoders.append("swav_climate+_ep99")
    #encoders = ['supervised', 'swav-imagenet', 'swav-c1']
    trials = ['t1', 't2', 't3']

    results = {
        "datasize": [],
        "encoder": [],
        "trial": [],
        "iou": [],
        "max_f1": [],
    }

    # Iterate through all models and capture the statistics
    with torch.no_grad():
        for ds in data:
            print(f"On dataset {ds}...")
            for enc in encoders:
                print(f"On encoder {enc}...")
                test_dataset = get_dataloader(enc, args.task)
                for trial in trials:
                    print(f"On trial {trial}...")
                    model = Model(base_path, ds, enc, trial)

                    model.to('cuda')

                    iou, f1 = get_stats(model, test_dataset)
                    # pdb.set_trace()

                    results['datasize'].append(ds)
                    results['encoder'].append(enc)
                    results['trial'].append(trial)
                    results['iou'].append(iou.item())
                    results['max_f1'].append(f1)

    # Save the dataframe
    df = pd.DataFrame(results)

    df.to_csv(args.output)


def get_stats(model, dataloader):
    preds = []
    targets = []
    calc_iou = JaccardIndex()
    iou_results = []

    for i, (img, mask) in enumerate(dataloader):
        # Load through the model.
        img = img.to(model.device)
        mask = mask.to(model.device)
        with torch.no_grad():
            output = model(torch.reshape(img, (1, 3, 224, 224)))

        # Calculate IoU
        iou_results.append(calc_iou.update(output, mask))

        # Calculate precision / recall
        preds.append(output.cpu().numpy().flatten())
        targets.append(mask.cpu().numpy().flatten())

    f1 = get_max_f1(preds, targets)
    return calc_iou.value, f1


def get_max_f1(preds, targets):
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    precision, recall, t = precision_recall_curve(targets, preds)

    max_f1 = ((2 * precision * recall) / (precision + recall)).max()

    return max_f1


class Model:
    def __init__(self, base_path, datasize, encoder, trial, device='cuda'):
        full_path = os.path.join(base_path, datasize, encoder, trial)
        model_type = 'best'
        self.encoder = encoders.load('none')
        self.encoder.load_state_dict(torch.load(
            os.path.join(full_path, "enc_" + model_type + ".pt")
        ))
        self.decoder = decoders.load('unet', self.encoder)

        self.decoder.load_state_dict(torch.load(
            os.path.join(full_path, "dec_" + model_type + ".pt")
        ))

        # Make sure the model is in evaluate mode.
        self.encoder.eval()
        self.decoder.eval()

        self.device = device

    @torch.no_grad()
    def __call__(self, inp):

        output = self.encoder(inp)
        output = self.decoder(output)
        output = torch.sigmoid(output)

        return output

    def to(self, device):
        self.device = device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)


class JaccardIndex():
    def __init__(self, threshold=0.5):
        self.numerator = 0
        self.denominator = 0
        self.value = 0
        # In case we want to change the threshold
        self.threshold = threshold

    def update(self, preds, target):
        o_2 = copy.deepcopy(preds)
        o_2[o_2 < self.threshold] = 0
        o_2[o_2 >= self.threshold] = 1

        intersection = (o_2 * target).sum()

        union = o_2.sum() + target.sum() - intersection

        # Prevent dividing by zero
        if union == 0:
            return 0
        self.numerator += intersection
        self.denominator += union
        self.value = self.numerator / self.denominator

        return (intersection / union).item()

    def __repr__(self):
        return self.numerator / self.denominator


def get_dataloader(encoder):
    if encoder in [f"swav_climate+_ep{n}" for n in range(0,100,10)]: 
        print("Norm is data")
        norm = 'data'
    else:
        norm = 'imagenet'

    _, test_dataset = datasets.load(
        task=args.task, normalization=norm, old=False, size="1024")
    
    return test_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script handles batch analysis creation."
    )
    parser.add_argument(
        '--task',
        choices=['solar', 'building', 'crop_delineation'],
        type=str,
        help='The task of the experiment to analyze',
        required=True
    )

    parser.add_argument(
        '--encoder',
        type=str,
        help='The encoder to use for analysis',
        required=False
    )
    parser.add_argument('--output',
                        type=str,
                        help='Name of output csv file',
                        required=True)

    args = parser.parse_args()

    main()
