

import torch
import a2_bleu_score


from tqdm import tqdm


def train_for_epoch(model, dataloader, optimizer, device):
    '''Train an EncoderDecoder for an epoch

    An epoch is one full loop through the training data. This function:

    1. Defines a loss function using :class:`torch.nn.CrossEntropyLoss`,
       keeping track of what id the loss considers "padding"
    2. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E``)
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens`` and ``E``.
       2. Zeros out the model's previous gradient with ``optimizer.zero_grad()``
       3. Calls ``logits = model(F, F_lens, E)`` to determine next-token
          probabilities.
       4. Modifies ``E`` for the loss function, getting rid of a token and
          replacing excess end-of-sequence tokens with padding using
        ``model.get_target_padding_mask()`` and ``torch.masked_fill``
       5. Flattens out the sequence dimension into the batch dimension of both
          ``logits`` and ``E``
       6. Calls ``loss = loss_fn(logits, E)`` to calculate the batch loss
       7. Calls ``loss.backward()`` to backpropagate gradients through
          ``model``
       8. Calls ``optim.step()`` to update model parameters
    3. Returns the average loss over sequences

    Parameters
    ----------
    model : EncoderDecoder
        The model we're training.
    dataloader : HansardDataLoader
        Serves up batches of data.
    device : torch.device
        A torch device, like 'cpu' or 'cuda'. Where to perform computations.
    optimizer : torch.optim.Optimizer
        Implements some algorithm for updating parameters using gradient
        calculations.

    Returns
    -------
    avg_loss : float
        The total loss divided by the total numer of sequence
    '''
    # If you want, instead of looping through your dataloader as
    # for ... in dataloader: ...
    # you can wrap dataloader with "tqdm":
    # for ... in tqdm(dataloader): ...
    # This will update a progress bar on every iteration that it prints
    # to stdout. It's a good gauge for how long the rest of the epoch
    # will take. This is entirely optional - we won't grade you differently
    # either way.
    # If you are running into CUDA memory errors part way through training,
    # try "del F, F_lens, E, logits, loss" at the end of each iteration of
    # the loop.
    #assert False, "Fill me"
    #using cross entropy loss
    cross, accm, batch = torch.nn.CrossEntropyLoss(ignore_index=model.source_pad_id), 0, 0
    #optimizer.zero_grad()
    for F, F_lens, E in dataloader:
        F, F_lens, E = F.to(device), F_lens.to(device), E.to(device)
        optimizer.zero_grad()
        log = model(F, F_lens, E)
        codings = model.get_target_padding_mask(E)
        E = E.masked_fill(codings, model.source_pad_id)
        log, E, b_loss = torch.flatten(log, start_dim=0, end_dim=-2), torch.flatten(E[1:], start_dim=0), cross(log, E)
        b_loss.backward()
        optimizer.step()
        accm = accm + b_loss.item()
        batch = batch + 1
        del b_loss, log, F_lens, F
    average_loss = accm / batch
    return average_loss




def compute_batch_total_bleu(E_ref, E_cand, target_sos, target_eos):
    '''Compute the total BLEU score over elements in a batch

    Parameters
    ----------
    E_ref : torch.LongTensor
        A batch of reference transcripts of shape ``(T, M)``, including
        start-of-sequence tags and right-padded with end-of-sequence tags.
    E_cand : torch.LongTensor
        A batch of candidate transcripts of shape ``(T', M)``, also including
        start-of-sequence and end-of-sequence tags.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    total_bleu : float
        The sum total BLEU score for across all elements in the batch. Use
        n-gram precision 4.
    '''
    # you can use E_ref.tolist() to convert the LongTensor to a python list
    # of numbers
    #assert False, "Fill me"
    n_gram_precision=4
    counter, start_of_sent, end_of_sent = 0, str(target_sos), str(target_eos)
    E_cand, E_ref = E_cand.permute(1, 0).tolist(), E_ref.permute(1, 0).tolist()
    for r, c in zip(E_ref, E_cand):
        hypothesis = [str(i) for i in c if str(i) != start_of_sent and str(i) != end_of_sent] # and str(i) != ...
        reference = [str(k) for k in r if str(k) != start_of_sent and str(k) != end_of_sent]
        counter = counter + a2_bleu_score.BLEU_score(reference, hypothesis, n_gram_precision)
    return counter


def compute_average_bleu_over_dataset(
        model, dataloader, target_sos, target_eos, device):
    '''Determine the average BLEU score across sequences

    This function computes the average BLEU score across all sequences in
    a single loop through the `dataloader`.

    1. For every iteration of the `dataloader` (which yields triples
       ``F, F_lens, E_ref``):
       1. Sends ``F`` to the appropriate device via ``F = F.to(device)``. Same
          for ``F_lens``. No need for ``E_cand``, since it will always be
          compared on the CPU.
       2. Performs a beam search by calling ``b_1 = model(F, F_lens)``
       3. Extracts the top path per beam as ``E_cand = b_1[..., 0]``
       4. Computes the total BLEU score of the batch using
          :func:`compute_batch_total_bleu`
    2. Returns the average per-sequence BLEU score

    Parameters
    ----------
    model : EncoderDecoder
        The model we're testing.
    dataloader : HansardDataLoader
        Serves up batches of data.
    target_sos : int
        The ID of the start-of-sequence tag in the target vocabulary.
    target_eos : int
        The ID of the end-of-sequence tag in the target vocabulary.

    Returns
    -------
    avg_bleu : float
        The total BLEU score summed over all sequences divided by the number of
        sequences
    '''
    #assert False, "Fill me"
    counter, points = 0, 0
    for F, F_lens, E in dataloader:
        F, F_lens, E = F.to(device), F_lens.to(device), E.to(device)
        points = F_lens.size(0)
        evalu = model(F, F_lens)
        hyp = evalu[:, :, 0]
        counter = counter + compute_batch_total_bleu(F, hyp, target_sos=target_sos, target_eos=target_eos)
    per_sample = counter / points
    return per_sample
