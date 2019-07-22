 #!/usr/bin/env python 
"""
File: concept_teacher.py
Author: Will Schwarzer (schwarzerw@carleton.edu)
Date: July 15, 2019
Description: Trains a concept encoder (MLP for vectorized concepts or CNN for
images) and decoder (LSTM).
"""
import uuid
import json
import matplotlib
import dill
import os
import numpy as np
import revtok
import sys
import time

from models.teacher.teacher import Teacher
from utils.constants import Constants
from utils.dataloaders.vectorized.concept_load_dataset_teacher import load_dataset, construct_stim_reps, construct_y, convert_to_elmo_ids
from experiments.utils import AverageMeter, AccuracyMeter, save_student_checkpoint, set_seeds
from experiments.utils import load_single_task_student_checkpoint as load_student_checkpoint

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchtext.data as data
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--hidden-dim-l', type=int, default=100,
                        help='hidden dimensions (language)')
    parser.add_argument('--output-dim-l', type=int, default=100,
                        help='output dimensions (language)')
    parser.add_argument('--num-layers-l', type=int, default=1,
                        help='number of layers in RNN')
    parser.add_argument('--dropout-l', type=float, default=0.0,
                        help='dropout probability (language)')

    parser.add_argument('--self-att', action='store_true', default=False,
                    help='whether to use self attention')
    parser.add_argument('--d-dim-l', type=int, default=100,
                    help='d-dim for self attention')
    parser.add_argument('--r-dim-l', type=int, default=5,
                    help='number of attention hops for self attention')

    parser.add_argument('--input-dim-s', type=int, default=78,
                        help='input representation dimensions (stim)')
    parser.add_argument('--hidden-dim-s', type=int, default=100,
        help='hidden representation dimensions (stim)')
    parser.add_argument('--output-dim-s', type=int, default=100,
        help='output representation dimensions (stim)')
    parser.add_argument('--num-layers-s', type=int, default=3,
                        help='number of layers in stim representation network')
    parser.add_argument('--dropout-s', type=float, default=0.0,
                        help='dropout probability (stim)')


    parser.add_argument('--hidden-dim-student', type=int, default=100,
        help='hidden representation dimensions (student)')
    parser.add_argument('--num-layers-student', type=int, default=3,
                        help='number of layers in student prediction network')
    parser.add_argument('--dropout-student', type=float, default=0.0,
                        help='dropout probability (student)')

    parser.add_argument('--train-obj', type=str, default='ground_truth',
                        help='ground_truth, teacher, student')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--data', type=str, default='./data/concept/{}/vectorized/unique_concept_dataset.tsv',
                        help='file template for dataset')
    parser.add_argument('--bn', action='store_true', default=False,
                        help='use batch normalization')
    parser.add_argument('--embeddings', type=str, default='glove',
                        help='embeddings to use')
    #  parser.add_argument('--out-dir', type=str, default='/mnt/fs5/schopra/ratchet/lfl/student/concept/models',
                        #  help='where to save models')
    # XXX: On the cocolab cluster set this to an /mnt/fsX directory!
    parser.add_argument('--out-dir', type=str, default='./saves/',
                        help='where to save models')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--greedy', action='store_true', default=False,
                        help='enables greedy language sampling')
    parser.add_argument('--lemmatized', type=bool, default=True,
                        help='enables or disables lemmatization for tokenization')
    args = parser.parse_args()
    args.cuda = args.cuda and torch.cuda.is_available()
    ### print(torch.cuda.is_available())
    ### print(args.cuda)

    # Model IDs
    ### Note: stored in repo/saves/model_ids
    ### Models are stored in repo/saves/models, in folders with their id as the name
    ### TODO what is the .pkl file that they use? RESOLVED apparently it's a pickle file, so like a stored python object
    ### So (trained/untrained?) models are just objects
    model_id = str(uuid.uuid4())
    model_ids_dir = os.path.join(args.out_dir, 'model_ids')
    if not os.path.isdir(model_ids_dir):
        os.makedirs(model_ids_dir)  

    # Set seeds
    ### Sets all RNG seeds with some fixed value
    set_seeds()

    ### Makes folder to put models in
    args.out_dir = os.path.join(args.out_dir, 'models')
    if not os.path.isdir(os.path.join(args.out_dir, model_id)):
        os.makedirs(os.path.join(args.out_dir, model_id))

    # Construct Data Loaders &  Iterators
    ### Default args.data is /data/concept/{}/vectorized/concat_informative_dataset
    ### (.tsv is just tab-separated-value, i.e. like csv, so just a table)
    ### The actual '{}' folders are (raw; maybe not used here), test, train and val
    ### {}_data is a TabularDataset (torchtext.data object)

    ### fields is column_field_types, which was also set to be the "fields" variable of {}_data
    train_data, val_data, test_data, fields, stim_fields = load_dataset(args.data, args.lemmatized)
    sort_key = lambda x: len(x.text)
    if args.cuda:
        # GPU available
        ### data is torchtext.data
        ### Iterator.splits() creates iterators (essentially DataLoaders, except for TabularDatasets, AFAIK) for multiple splits
        train_loader, val_loader = data.Iterator.splits(
                (train_data, val_data), sort_key=sort_key, sort_within_batch=True,
                batch_sizes=(args.batch_size, args.batch_size), device=torch.device('cuda'))
    else:
        # CPU only
        train_loader, val_loader = data.Iterator.splits(
                (train_data, val_data), sort_key=sort_key, sort_within_batch=True,
                batch_sizes=(args.batch_size, args.batch_size), device=torch.device('cpu'))


    # Construct vocabulary objects & write to disk
    ### glove by default
    ### builds the mapping from tokens to (well, initially I thought ints, but actually) representation vectors
    ### also converts words to their vocab indices
    ### passing in train_data tells torch that the words in train_data['text'] are the words it should use as keys
    ### note: the (gigantic) vocab file is stored in .\.vector_cache
    if args.embeddings == "glove":
        fields['text'][1].build_vocab(train_data, vectors="glove.840B.300d")
        torch.save(fields['text'][1], os.path.join(args.out_dir, model_id, 'vocab.pkl'), pickle_module=dill)
    elif args.embeddings == "elmo":
        fields['text'][1].build_vocab(train_data)
        torch.save(fields['text'][1], os.path.join(args.out_dir, model_id, 'vocab.pkl'), pickle_module=dill)
        print("Using ELMO's pre-trained embeddings")
    else:
        raise Exception("Invalid Embeddings Type")
  
    kwargs = {
        'language_model_type': 'bilstm',
        'stim_model_type': 'featureMLP',

        'h_dim_l': args.hidden_dim_l,
        'o_dim_l': args.output_dim_l,
        'd_prob_l': args.dropout_l,
        'num_layers_l': args.num_layers_l,

        'with_self_att': args.self_att,
        'd_dim_l': args.d_dim_l,
        'r_dim_l': args.r_dim_l,

        'i_dim_s': args.input_dim_s,
        'h_dim_s': args.hidden_dim_s,
        'o_dim_s': args.output_dim_s,
        'd_prob_s': args.dropout_s,
        'num_layers_s': args.num_layers_s,

        'hidden_dim_student': args.hidden_dim_student,
        'num_layers_student': args.num_layers_student,
        'd_prob_student': args.dropout_student,
        'output_dim': 2,

        'train_obj': args.train_obj,
        'batch_norm': args.bn,
        'embeddings': args.embeddings,
        'model_id': model_id,
        'date:': time.strftime("%Y-%m-%d %H:%M"),
        'task': 'concept',

        'unk_index': fields['text'][1].vocab.stoi[Constants.UNK_TOKEN],
        'pad_index': fields['text'][1].vocab.stoi[Constants.PAD_TOKEN],
        'start_index': fields['text'][1].vocab.stoi[Constants.START_TOKEN],
        'end_index': fields['text'][1].vocab.stoi[Constants.END_TOKEN],

        'greedy_sampling': True
    }

    with open(os.path.join(model_ids_dir, '{}.json'.format(kwargs['model_id'])), 'w') as model_params_f:
        json.dump(kwargs, model_params_f)

    kwargs['concept_vocab_field'] = fields['text'][1]
    kwargs['reference_vocab_field'] = None

    def compute_loss(batch):
        """ Compute loss.
        """
        stims, labels, language, lang_lengths = get_inputs(batch)
        logits = teacher(stims, labels, language, lang_lengths)
        # logits shape: (batch size, max seq length, num vocab)
        # Assume rnn_decoder is your language model;
        # img_rep is your image representation (batch_size x hidden_size);
        # language is your list of sentences (batch_size x max_lang_length)
        # lang_length is your list of language lengths (batch_Size)
        max_seq_len = language.size(1)
        # We only care about logits up to the last token 
        # (after the last token, there's nothing to predict!)
        ### Also, we don't need to care about how it predicted the first token, 
        ### since it's always an SOS
        logits = logits[:, :-1].contiguous()
        language = language[:, 1:].contiguous()

        # Get the batch size (and make sure it's the same for all data)
        batch_size = stims.shape[0]
        assert(batch_size == language.shape[0] == lang_lengths.shape[0] == labels.shape[0])
        # "Unfold" the sequence so we have a 2d matrix
        logits_2d = logits.view(batch_size * (max_seq_len - 1), -1)
        ### TODO we probably don't need to convert language to longs here
        ### (it should already be longs, since we never converted to floats
        language_1d = language.long().view(batch_size * (max_seq_len - 1))

        # Cross entrops is your loss function - in short, you pay a penalty if you put probability amss onl #TODO ?
        # Note this works *without* having to normalize the softmax output
        loss = F.cross_entropy(logits_2d, language_1d, reduction='none')
        loss = loss.view(batch_size, (max_seq_len - 1))
        
        # Mask out losses for pad tokens
        loss *= (language != kwargs['pad_index']).float()
        # Sum up the loss for each token prediction to get a total loss per language
        total_losses = torch.sum(loss, dim=1)
        # Normalize total loss for each sequence by its length
        total_losses /= lang_lengths.float()
        # then average across language in the batch
        average_loss = torch.mean(total_losses)
        return average_loss, logits


    def compute_self_att_loss(alphas):
        """ Compute self attention loss.
        """ 
        if args.self_att:
            assert(alphas is not None), "Self Attention should have been applied"
            I = torch.eye(args.r_dim_l)
            if args.cuda:
                I = I.to('cuda')
            I = I.repeat(alphas.shape[0], 1, 1)
            alphas_t = torch.transpose(alphas, 1, 2).contiguous()
            return torch.norm(torch.bmm(alphas, alphas_t) - I)
        else:
            return 0.0

    def get_inputs(batch):
        (language, lang_lengths) = batch.text
        ### language: (batch_size, max language length)
        ### lang_lengths: (batch_size)
        if args.embeddings == "elmo":
            ### TODO change vars
            x_l_reversed = vocab_field.reverse(x_l.data)
            x_l = convert_to_elmo_ids(x_l_reversed, args.cuda)
            x_l_lengths = None
        ### get the (batch_size) tensor (basically list) of stimuli
        stims = construct_stim_reps(batch)
        labels = batch.__dict__['labels'].float()
        if args.cuda:
            stims = stims.to('cuda')
            language = language.to('cuda')
            lang_lengths = lang_lengths.to('cuda')
            labels = labels.to('cuda')
        orig_lang = fields['text'][1].reverse(language)
        return stims, labels, language, lang_lengths

    def get_samples_and_prototypes(val=False):
        if val:
            loader = val_loader
        else: 
            loader = train_loader
        ### TODO why are so many words mapping to the unknown character?
        ### NOTE stoi stands for string to index; itos stands for index to string
        orig_lang, gen_lang, gen_lang_greedy, stims_list, pos_list, neg_list = [], [], [], [], [], []
        for batch in loader:
            stims, labels, language, _ = get_inputs(batch)
            orig_lang.extend(fields['text'][1].reverse(language))
            gen_ids = teacher.sample(stims, labels, **kwargs)
            gen_lang_greedy.extend(fields['text'][1].reverse(gen_ids[0]))
            kwargs['greedy_sampling'] = False
            gen_ids = teacher.sample(stims, labels, **kwargs)
            gen_lang.extend(fields['text'][1].reverse(gen_ids[0]))
            kwargs['greedy_sampling'] = True
            stims_list.extend(stims.tolist())
            pos_prototypes, neg_prototypes = teacher.get_prototypes(stims, labels)
            pos_list.extend(pos_prototypes.tolist())
            neg_list.extend(neg_prototypes.tolist())
        df = pd.DataFrame(
            list(zip(stims_list, orig_lang, gen_lang_greedy, gen_lang, pos_list, neg_list)), 
            columns=['stims', 'orig_lang', 'gen_lang_greedy', 'gen_lang', 'pos_prototypes', 'neg_prototypes'])
        return df
            

    def train(epoch=-1, backprop=True):
        """ Train model for a single epoch.
        """
        ### Model is declared in global space 
        ### (although after this function, which is allowed, I guess?)
        # Data loading & progress visualization
        ### loss_meter stores the loss of the current batch and the average loss
        loss_meter = AverageMeter()
        #acc_meter = AccuracyMeter()
        ### tqdm is a progress bar for iterations of a loop through an iterator
        ### generally you wrap the iterable in it, like tqdm(range), to automatically keep track of #iterations
        ### but you can also use pbar.update() to tell it to iterate manually
        pbar = tqdm(total=len(train_loader))
        train_loader.init_epoch()
        
        if backprop:
            ### Sets model in training mode
            teacher.train()
        else:
            ### Sets model in evaluation mode
            ### (This is never used in code, i.e. backprop is always true)
            teacher.eval()

        ### Enumerate produces a (counter, item) pair for each item in an iterator
        ### "train_loader" is an iterator of minibatches
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            ### loss is F.cross_entropy(logits, y)
            loss, logits = compute_loss(batch)

            if backprop:
                ### step I don't understand
                loss.backward()
                optimizer.step()

            # Update progress
            #acc_meter.update(logits, batch)
            loss_meter.update(loss.data.item(), batch.batch_size)
            ### Every so often, print progress information (example #, average loss over this epoch, etc.)
            ### also resets pbar onto a new line
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * batch.batch_size, len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss_meter.avg))
            pbar.update()
        pbar.close()
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, loss_meter.avg))
        print('Training Objective ({}) Train Accuracies:'.format(kwargs['train_obj']))
        #acc_meter.print()
        return loss_meter.avg


    def val():
        """ Run model through validation dataset.
        """
        loss_meter = AverageMeter()
        #acc_meter = AccuracyMeter()
        pbar = tqdm(total=len(val_loader))
        val_loader.init_epoch()
        teacher.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                loss, logits = compute_loss(batch)
                #loss += compute_self_att_loss(alphas)
                #acc_meter.update(logits, batch)
                loss_meter.update(loss.data.item(), batch.batch_size)
                pbar.update()
        pbar.close()
        print('====> Validation set loss: {:.4f}'.format(loss_meter.avg))
        print('Training Objective ({}) Validation Accuracies:'.format(kwargs['train_obj']))
        #acc_meter.print()
        return loss_meter.avg#, acc_meter

    # Model Training
    ### Setting the seeds again for some reason?
    set_seeds()
    ### Student is the model we're training
    ### Kwargs is a dictionary of variables declared above
    teacher = Teacher(**kwargs)
    kwargs.pop('concept_vocab_field', None) # remove vocab_field, as it is not serializable
    if args.cuda:
        teacher = teacher.to('cuda')
    ### Adam is an optimization algorithm, like SGD, except that it changes its learning rate dynamically
    ### based on recent gradients (low gradients --> high LR, vice versa)
    ### It's similar in this way to root mean square propagation (RMSProp), just a little more sophisticated
    ### People apparently just use it because it works well
    optimizer = optim.Adam(teacher.parameters(), lr=args.lr, weight_decay=1e-4)

    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    ### This seaborn graph not used?
    import seaborn as sns
    sns.set_style('whitegrid')

    best_acc = 0.0
    best_epoch = -1
    ### Losses format: epoch1_loss_train, ..., epochn_loss_train
    ###                epoch1_loss_val,   ..., epochn_loss_val
    losses = np.zeros((args.epochs, 2))
    for epoch in range(1, args.epochs + 1):
        train_loss = train(epoch)
        val_loss = val()
        losses[epoch - 1, 0] = train_loss
        losses[epoch - 1, 1] = val_loss

        # keep track of best weights -- this is equivalent
        # to a simple version of early-stopping
        '''
        is_best = best_acc < val_acc_meter.compute_gt_acc()
        if is_best:
            best_acc = val_acc_meter.compute_gt_acc()
            best_epoch = epoch

        # save weights to dict
        ### This really should just be taken from kwargs
        save_student_checkpoint(
            {
                'state_dict': teacher.state_dict(),
                'val_loss': val_loss,
                'vocab_file': os.path.join(args.out_dir, model_id, 'vocab.pkl'),
                'optimizer': optimizer.state_dict(),
                'language_model_type': 'bilstm',
                'stim_model_type': 'featureMLP',
                'kwargs': kwargs
            },
            is_best,
            os.path.join(args.out_dir, kwargs['model_id']), 
            'checkpoint.pth.tar',
            'model_best.pth.tar'
        )
    '''
    train_samples = get_samples_and_prototypes(val=False)
    train_samples.to_csv(os.path.join(args.out_dir, model_id, 'samples_train.csv'), index=False)
    val_samples = get_samples_and_prototypes(val=True)
    val_samples.to_csv(os.path.join(args.out_dir, model_id, 'samples_val.csv'), index=False)
    
    # plot loss over time
    plt.figure()
    plt.plot(range(args.epochs), losses[:, 0], '-', label='train')
    plt.plot(range(args.epochs), losses[:, 1], '-', label='val')
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, model_id, 'loss.png'))

    '''
    print("Loading best model from disk ...")
    teacher = load_student_checkpoint(os.path.join(args.out_dir, model_id, 'model_weights', 'model_best.pth.tar'), use_cuda=args.cuda)
    train(best_epoch, False)
    val()
    '''

