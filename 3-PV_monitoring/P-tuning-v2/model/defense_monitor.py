import torch

def monitor_output(output, PV, threshold=0.8):
    bsz = output.shape[0]
    hidden_size = output.shape[1]
    num_PV = PV.shape[0]

    ones = torch.ones_like(PV)
    minus_ones = -1. * ones
    simplified_PV = torch.where(PV >= 0, ones, minus_ones)

    ones = torch.ones_like(output)
    minus_ones = -1. * ones
    simplified_output = torch.where(output >= 0, ones, minus_ones)

    logits = torch.zeros(bsz, 2)
    with torch.no_grad():
        for batch in range(bsz):
            batch_output = simplified_output[batch,:]
            num_match = 0
            # fimd the max num_match
            for i in range(num_PV):
                temp_num_match = torch.eq(batch_output, simplified_PV[i,:]).sum()
                if temp_num_match > num_match:
                    num_match = temp_num_match

            if num_match / hidden_size > threshold:
                logits[batch,:] = torch.tensor([-1.0, 1.0])
            else:
                logits[batch,:] = torch.tensor([1.0, -1.0])

    return logits

