import torch


def collate_fn(batch):
    inp = torch.stack([batch[i][0] for i in range(len(batch))])
    out = torch.stack([batch[i][1] for i in range(len(batch))])
    lead_times = batch[0][2]
    variables = batch[0][3]
    out_variables = batch[0][4]
    return inp, out, lead_times, variables, out_variables