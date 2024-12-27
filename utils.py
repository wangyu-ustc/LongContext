import torch
import random
import numpy as np

def get_num_tokens(N, K, num_contexts):
    ages = np.zeros(N)
    for i in range(num_contexts):
        indices = np.random.choice(np.arange(N), size=K, replace=False)
        remaining_indices = np.setdiff1d(np.arange(N), indices)
        ages = np.concatenate([
            ages[remaining_indices],
            np.array([i + 1] * K)
        ])
    # _, counts = np.unique(ages, return_counts=True)
    counts = []
    for age in np.arange(num_contexts + 1):
        counts.append(np.sum(ages == age))
    return counts

def split_sequence(seq_length, min_length, max_length):

    if seq_length < min_length:
        return [seq_length]
    
    # Initialize the chunks
    chunks = []
    remaining_length = seq_length

    while remaining_length > 0:
        # Calculate the maximum length for the current chunk
        max_chunk_length = min(remaining_length, max_length)
        
        # Ensure the chunk is at least min_length characters long and handle the remaining length
        if remaining_length <= max_length:
            if remaining_length < min_length and chunks:
                # Adjust previous chunks to make the last chunk at least min_length
                needed = min_length - remaining_length
                for i in range(len(chunks) - 1, -1, -1):
                    if chunks[i] - needed >= min_length:
                        chunks[i] -= needed
                        remaining_length += needed
                        break
                    else:
                        needed -= (chunks[i] - min_length)
                        remaining_length += (chunks[i] - min_length)
                        chunks[i] = min_length

            chunk_length = remaining_length if remaining_length >= min_length else min_length

        else:
            chunk_length = random.randint(min_length, max_chunk_length)
        
        # Append the chunk length to the list
        chunks.append(chunk_length)
        
        # Reduce the remaining length
        remaining_length -= chunk_length

    return chunks


def collate_fn(batch, tokenizer1, tokenizer2, min_length, max_seq_length, max_length):

    new_batch = {
        'contexts_ids': [],
        'sentence_ids': []
    }

    assert len(batch) == 1, "Batch size must be 1"

    x = batch[0]

    input_ids = tokenizer2(tokenizer1.decode(x['input_ids'], skip_special_tokens=True),
                add_special_tokens=False,).input_ids

    sequence_ids = input_ids[-max_seq_length:]
    input_ids = input_ids[:-max_seq_length]

    chunks = split_sequence(len(input_ids), min_length, max_length)

    contexts_ids = []

    for chunk in chunks:
        contexts_ids.append(torch.tensor(input_ids[:chunk]).unsqueeze(0))
        input_ids = input_ids[chunk:]
    
    sentence_ids = torch.tensor(sequence_ids).unsqueeze(0)

    new_batch['contexts_ids'] = contexts_ids
    new_batch['sentence_ids'] = sentence_ids

    return new_batch

