import torch
import gc
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm, trange
random_seed = 42

from evaluate import Evaluate

def engage_early_stopping(cur_epoch, train_loss_set, nb_tr_steps):
    """
    Early Stopping
    modeled after tf.keras.callbacks.EarlyStopping
    TODO: restore_best_weights
    """
    flag = False
    patience = 3
    monitor = 'loss'
    min_delta = 0
    if monitor=='loss' and cur_epoch+1 >= patience:
        assert (len(train_loss_set)%nb_tr_steps==0)
        loss_per_epoch = []
        for i in range(0, len(train_loss_set), nb_tr_steps):
            loss_per_epoch.append(sum(train_loss_set[i:i+nb_tr_steps])/nb_tr_steps)
        # do not proceed with training if difference in loss is above threshold `min_delta`
        if loss_per_epoch[-1]-loss_per_epoch[-1*patience] > min_delta:
            flag = True
    return flag

def train_test_loader(X, y, target_indices, k, batch_train, batch_test, window_size):
    """Generate k-fold splits given X, y"""
    random_state = random_seed
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in X:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    attention_masks = np.array(attention_masks)

    kf = KFold(n_splits=k, random_state=random_state, shuffle=True)

    X = X.numpy()

    target_indices = np.array(target_indices)   # target token indexes

    lcontext_indices = []
    rcontext_indices= []
    for item in target_indices:
        lcontext_indices.append([*range(item-window_size, item)])
        rcontext_indices.append([*reversed(range(item, item+window_size))]) # right context in reverse order
    print('lcontext_indices[-1]: ', lcontext_indices[-1], '; rcontext_indices[-1]: ', rcontext_indices[-1])
    lcontext_indices = np.array(lcontext_indices)
    rcontext_indices = np.array(rcontext_indices)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        target_indices_train, target_indices_test = target_indices[train_index], target_indices[test_index]  # target token indexes
        lcontext_indices_train, lcontext_indices_test = lcontext_indices[train_index], lcontext_indices[test_index]
        rcontext_indices_train, rcontext_indices_test = rcontext_indices[train_index], rcontext_indices[test_index]

        train_masks, test_masks = attention_masks[train_index], attention_masks[test_index]

        train_indices = torch.tensor(train_index)
        test_indices = torch.tensor(test_index)    # these are actual indices which are going to be used for retrieving items after prediction

        # Convert to torch tensors
        X_train = torch.tensor(X_train)
        X_test = torch.tensor(X_test)

        y_train = torch.tensor(y_train)
        y_test = torch.tensor(y_test)

        train_masks = torch.tensor(train_masks)
        test_masks = torch.tensor(test_masks)

        target_indices_train = torch.tensor(target_indices_train)
        target_indices_test = torch.tensor(target_indices_test)

        lcontext_indices_train = torch.tensor(lcontext_indices_train)
        lcontext_indices_test = torch.tensor(lcontext_indices_test)

        rcontext_indices_train = torch.tensor(rcontext_indices_train)
        rcontext_indices_test = torch.tensor(rcontext_indices_test)

        # Create an iterator with DataLoader
        train_data = TensorDataset(X_train, train_masks, y_train, target_indices_train, lcontext_indices_train, rcontext_indices_train, train_indices)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_train, drop_last=True)

        test_data = TensorDataset(X_test, test_masks, y_test, target_indices_test, lcontext_indices_test, rcontext_indices_test, test_indices)
        test_dataloader = DataLoader(test_data, sampler=None, batch_size=batch_test)

        yield train_dataloader, test_dataloader


def trainer(epochs, model, optimizer, scheduler, train_dataloader, test_dataloader, batch_train, batch_test, device, expt_model_choice):

    max_grad_norm = 1.0
    train_loss_set = []

    for e in trange(epochs, desc="Epoch"):

        while gc.collect() > 0:
            pass

        # Training
        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # if e > 8:
        #     model.freeze_bert()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # print('Train step: ', step)
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_target_idx, b_lcontext_idxs, b_rcontext_idxs, _ = batch

            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            if expt_model_choice=='BertWithGCNAndMWE':
                ### For BERT + GCN and MWE
                loss = model(b_input_ids.to(device), attention_mask=b_input_mask.to(device), \
                            labels=b_labels, batch=batch_train, target_token_idx=b_target_idx.to(device))
            elif expt_model_choice=='BertWithPreWin':
                loss = model(b_input_ids.to(device), attention_mask=b_input_mask.to(device), \
                            labels=b_labels, batch=batch_train, target_token_idx=b_target_idx.to(device),
                            lcontext_indices=b_lcontext_idxs, rcontext_indices=b_rcontext_idxs)

            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        all_preds = torch.FloatTensor()
        all_labels = torch.LongTensor()
        test_indices = torch.LongTensor()

        # Evaluate data for one epoch
        for step, batch in enumerate(test_dataloader):
            # print('Test step: ', step)
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_target_idx, b_lcontext_idxs, b_rcontext_idxs, test_idx = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                if expt_model_choice=='BertWithGCNAndMWE':
                    ### For BERT + GCN and MWE
                    logits = model(b_input_ids.to(device), attention_mask=b_input_mask.to(device), \
                                   batch=batch_test, target_token_idx=b_target_idx.to(device))
                elif expt_model_choice=='BertWithPreWin':
                    logits = model(b_input_ids.to(device), attention_mask=b_input_mask.to(device), \
                                   batch=batch_test, target_token_idx=b_target_idx.to(device),
                                   lcontext_indices=b_lcontext_idxs, rcontext_indices=b_rcontext_idxs)


                # Move logits and labels to CPU
                logits = logits.detach().cpu()
                label_ids = b_labels.cpu()
                test_idx = test_idx.cpu()

                all_preds = torch.cat([all_preds, logits])
                all_labels = torch.cat([all_labels,label_ids])
                test_indices = torch.cat([test_indices, test_idx])

        # Early Stopping
        if engage_early_stopping(e, train_loss_set, nb_tr_steps):
            print('Engaging early stopping...')
            break

    scores = Evaluate(all_preds,all_labels)
    print('scores.accuracy(): {}'.format(scores.accuracy()))
    print('scores.precision_recall_fscore_coarse(): {}'.format(scores.precision_recall_fscore_coarse()))

    return scores, all_preds, all_labels, test_indices


