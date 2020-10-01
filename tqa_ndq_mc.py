from transformers import AdamW, RobertaForMultipleChoice, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import torch
import sys
import argparse

from aux_methods import get_data_ndq, process_data_ndq, training_ndq

def main(argv):
    parser = argparse.ArgumentParser(description='')
    required = parser.add_argument_group('required arguments')
    required.add_argument('-r', '--retrieval', choices=['IR', 'NSP', 'NN'] , help='retrieval solver for the contexts. Options: IR, NSP or NN', required=True)
    parser.add_argument('-t', '--dataset', default='ndq', choices=['ndq', 'dq'], help='dataset to train the model with. Options: ndq or dq. Default: ndq')
    parser.add_argument('-d', '--device', default='gpu', choices=['gpu', 'cpu'], help='device to train the model with. Options: cpu or gpu. Default: gpu')
    parser.add_argument('-p', '--pretrainings', default="checkpoints/pretrainings_e4.pth", help='path to the pretrainings model. If empty, the model will be the RobertForMultipleChoice with roberta-large weights. Default: checkpoints/pretrainings_e4.pth')
    parser.add_argument('-b', '--batchsize', default= 1, type=int, help='size of the batches. Default: 1')
    parser.add_argument('-x', '--maxlen', default= 180, type=int, help='max sequence length. Default: 180')
    parser.add_argument('-l', '--lr', default= 1e-5, type=float, help='learning rate. Default: 1e-5')
    parser.add_argument('-e', '--epochs', default= 4, type=int, help='number of epochs. Default: 4')
    parser.add_argument('-s', '--save', default=False, help='save model at the end of the training', action='store_true')
    args = parser.parse_args()
    print(args)
    
    if args.pretrainings == "":
        model = RobertaForMultipleChoice.from_pretrained("roberta-large")
    else:
        model = torch.load(args.pretrainings)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    
    if args.device=="gpu":
        device = torch.device("cuda")
        model.cuda()
    if args.device=="cpu":
        device = torch.device("cpu") 
        model.cpu()
    
    model.zero_grad()
    
    batch_size = args.batchsize
    max_len = args.maxlen
    lr = args.lr
    epochs = args.epochs
    retrieval_solver = args.retrieval
    save_model = args.save
    dataset_name = args.dataset

    raw_data_train = get_data_ndq(dataset_name, "train", retrieval_solver, tokenizer, max_len)
    raw_data_val = get_data_ndq(dataset_name, "val", retrieval_solver, tokenizer, max_len)
    
    train_dataloader = process_data_ndq(raw_data_train, batch_size, "train")
    val_dataloader = process_data_ndq(raw_data_val, batch_size, "val")

    
    optimizer = AdamW(model.parameters(), lr = lr, eps = 1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
    
    training_ndq(model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, retrieval_solver, device, save_model, dataset_name)

if __name__ == "__main__":
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    main(sys.argv[1:])