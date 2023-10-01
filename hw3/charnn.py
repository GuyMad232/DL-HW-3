import re
import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from typing import Iterator


def char_maps(text: str):
    """
    Create mapping from the unique chars in a text to integers and
    vice-versa.
    :param text: Some text.
    :return: Two maps.
        - char_to_idx, a mapping from a character to a unique
        integer from zero to the number of unique chars in the text.
        - idx_to_char, a mapping from an index to the character
        represented by it. The reverse of the above map.

    """
    # TODO:
    #  Create two maps as described in the docstring above.
    #  It's best if you also sort the chars before assigning indices, so that
    #  they're in lexical order.
    # ====== YOUR CODE: ======
    uniq_chars = list(set(text))
    uniq_chars.sort()
    char_to_idx = {char: index for index, char in enumerate(uniq_chars)}
    idx_to_char = {index: char for index, char in enumerate(uniq_chars)}
    # ========================
    return char_to_idx, idx_to_char


def remove_chars(text: str, chars_to_remove):
    """
    Removes all occurrences of the given chars from a text sequence.
    :param text: The text sequence.
    :param chars_to_remove: A list of characters that should be removed.
    :return:
        - text_clean: the text after removing the chars.
        - n_removed: Number of chars removed.
    """
    # TODO: Implement according to the docstring.
    # ====== YOUR CODE: ======
    text_clean = ''.join([char for char in text if char not in chars_to_remove])
    n_removed = len(text) - len(text_clean)
    # ========================
    return text_clean, n_removed


def chars_to_onehot(text: str, char_to_idx: dict) -> Tensor:
    """
    Embed a sequence of chars as a a tensor containing the one-hot encoding
    of each char. A one-hot encoding means that each char is represented as
    a tensor of zeros with a single '1' element at the index in the tensor
    corresponding to the index of that char.
    :param text: The text to embed.
    :param char_to_idx: Mapping from each char in the sequence to it's
    unique index.
    :return: Tensor of shape (N, D) where N is the length of the sequence
    and D is the number of unique chars in the sequence. The dtype of the
    returned tensor will be torch.int8.
    """
    # TODO: Implement the embedding.
    # ====== YOUR CODE: ======
    # indices Generate a list of indexes corresponding to the characters in the input text
    # For each character, set the element at the corresponding index in the tensor to 1
    # This is done where the first argument is the row indices and 
    # the second argument is the column indices where we want to set the value to 1
    N = len(text)
    D = len(char_to_idx)
    result = torch.zeros((N, D), dtype=torch.int8)
    indices = [char_to_idx[char] for char in text]
    result[range(N), indices] = 1
    # ========================
    return result


def onehot_to_chars(embedded_text: Tensor, idx_to_char: dict) -> str:
    """
    Reverses the embedding of a text sequence, producing back the original
    sequence as a string.
    :param embedded_text: Text sequence represented as a tensor of shape
    (N, D) where each row is the one-hot encoding of a character.
    :param idx_to_char: Mapping from indices to characters.
    :return: A string containing the text sequence represented by the
    embedding.
    """
    # TODO: Implement the reverse-embedding.
    # ====== YOUR CODE: ======
    indices = torch.argmax(embedded_text, dim=1)
    chars = [idx_to_char[idx.item()] for idx in indices]
    result = ''.join(chars)
    # ========================
    return result


def chars_to_labelled_samples(text: str, char_to_idx: dict, seq_len: int, device="cpu"):
    """
    Splits a char sequence into smaller sequences of labelled samples.
    A sample here is a sequence of seq_len embedded chars.
    Each sample has a corresponding label, which is also a sequence of
    seq_len chars represented as indices. The label is constructed such that
    the label of each char is the next char in the original sequence.
    :param text: The char sequence to split.
    :param char_to_idx: The mapping to create and embedding with.
    :param seq_len: The sequence length of each sample and label.
    :param device: The device on which to create the result tensors.
    :return: A tuple containing two tensors:
    samples, of shape (N, S, V) and labels of shape (N, S) where N is
    the number of created samples, S is the seq_len and V is the embedding
    dimension.
    """
    # TODO:
    #  Implement the labelled samples creation.
    #  1. Embed the given text.
    #  2. Create the samples tensor by splitting to groups of seq_len.
    #     Notice that the last char has no label, so don't use it.
    #  3. Create the labels tensor in a similar way and convert to indices.
    #  Note that no explicit loops are required to implement this function.
    # ====== YOUR CODE: ======
    # One-hotting the entire text.
    one_hot_text = chars_to_onehot(text, char_to_idx)
    num_samples = len(text[:-1]) // seq_len

    # Truncate the one-hot encoded text to fit
    num_relevant_chars = num_samples * seq_len
    
    # Create samples
    samples = one_hot_text[:num_relevant_chars, :]
    samples = samples.view(num_samples, seq_len, -1).to(device)

     # Create labels
    index_text = [char_to_idx[c] for c in text]  # Creating an index tensor
    index_text = torch.tensor(index_text[:num_relevant_chars+1], device=device)  
    labels = index_text[1:]  # Shift the index_text by one position to the right
    labels = labels.reshape(num_samples, seq_len).to(device)
    # ========================
    return samples, labels


def hot_softmax(y, dim=0, temperature=1.0):
    """
    A softmax which first scales the input by 1/temperature and
    then computes softmax along the given dimension.
    :param y: Input tensor.
    :param dim: Dimension to apply softmax on.
    :param temperature: Temperature.
    :return: Softmax computed with the temperature parameter.
    """
    # TODO: Implement based on the above.
    # ====== YOUR CODE: ======
    scaled_y = y / temperature
    result = torch.nn.functional.softmax(scaled_y, dim=dim)
    # ========================
    return result


def generate_from_model(model, start_sequence, n_chars, char_maps, T):
    """
    Generates a sequence of chars based on a given model and a start sequence.
    :param model: An RNN model. forward should accept (x,h0) and return (y,
    h_s) where x is an embedded input sequence, h0 is an initial hidden state,
    y is an embedded output sequence and h_s is the final hidden state.
    :param start_sequence: The initial sequence to feed the model.
    :param n_chars: The total number of chars to generate (including the
    initial sequence).
    :param char_maps: A tuple as returned by char_maps(text).
    :param T: Temperature for sampling with softmax-based distribution.
    :return: A string starting with the start_sequence and continuing for
    with chars predicted by the model, with a total length of n_chars.
    """
    assert len(start_sequence) < n_chars
    device = next(model.parameters()).device
    char_to_idx, idx_to_char = char_maps
    out_text = start_sequence

    # TODO:
    #  Implement char-by-char text generation.
    #  1. Feed the start_sequence into the model.
    #  2. Sample a new char from the output distribution of the last output
    #     char. Convert output to probabilities first.
    #     See torch.multinomial() for the sampling part.
    #  3. Feed the new char into the model.
    #  4. Rinse and Repeat.
    #  Note that tracking tensor operations for gradient calculation is not
    #  necessary for this. Best to disable tracking for speed.
    #  See torch.no_grad().
    # ====== YOUR CODE: ======
    def sample_next_char(x, h):
        c, h = model(x, h)
        c = hot_softmax(c[0, -1, :], -1, T)
        c = torch.multinomial(c, 1)
        c = idx_to_char[c.item()]
        return c, h

    with torch.no_grad():
        # Convert start sequence to tensor
        x = chars_to_onehot(start_sequence, char_to_idx).unsqueeze(0).type(torch.float).to(device)

        # Feed start sequence into model and sample new char
        c, h = sample_next_char(x, None)
        out_text += c

        while len(out_text) < n_chars:
            # Convert new char to tensor
            x = chars_to_onehot(c, char_to_idx).unsqueeze(0).type(torch.float).to(device)

            # Feed new char into model and sample next char
            c, h = sample_next_char(x, h)
            out_text += c

    # ========================

    return out_text


class SequenceBatchSampler(torch.utils.data.Sampler):
    """
    Samples indices from a dataset containing consecutive sequences.
    This sample ensures that samples in the same index of adjacent
    batches are also adjacent in the dataset.
    """

    def __init__(self, dataset: torch.utils.data.Dataset, batch_size):
        """
        :param dataset: The dataset for which to create indices.
        :param batch_size: Number of indices in each batch.
        """
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # TODO:
        #  Return an iterator of indices, i.e. numbers in range(len(dataset)).
        #  dataset and represents one  batch.
        #  The indices must be generated in a way that ensures
        #  that when a batch of size self.batch_size of indices is taken, samples in
        #  the same index of adjacent batches are also adjacent in the dataset.
        #  In the case when the last batch can't have batch_size samples,
        #  you can drop it.
        idx = None  # idx should be a 1-d list of indices.
        # ====== YOUR CODE: ======
        num_batches = len(self.dataset) // self.batch_size  # number of complete batches
        relevant_size = num_batches * self.batch_size
        idx = torch.arange(relevant_size)  # consider only complete batches
        idx = idx.view(self.batch_size, -1).t().contiguous().view(-1).tolist()  # transpose to ensure adjacency
        # ========================
        return iter(idx)

    def __len__(self):
        return len(self.dataset)


class MultilayerGRU(nn.Module):
    """
    Represents a multi-layer GRU (gated recurrent unit) model.
    """

    def __init__(self, in_dim, h_dim, out_dim, n_layers, dropout=0):
        """
        :param in_dim: Number of input dimensions (at each timestep).
        :param h_dim: Number of hidden state dimensions.
        :param out_dim: Number of input dimensions (at each timestep).
        :param n_layers: Number of layer in the model.
        :param dropout: Level of dropout to apply between layers. Zero
        disables.
        """
        super().__init__()
        assert in_dim > 0 and h_dim > 0 and out_dim > 0 and n_layers > 0

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.layer_params = []
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # TODO: Create the parameters of the model for all layers.
        #  To implement the affine transforms you can use either nn.Linear
        #  modules (recommended) or create W and b tensor pairs directly.
        #  Create these modules or tensors and save them per-layer in
        #  the layer_params list.
        #  Important note: You must register the created parameters so
        #  they are returned from our module's parameters() function.
        #  Usually this happens automatically when we assign a
        #  module/tensor as an attribute in our module, but now we need
        #  to do it manually since we're not assigning attributes. So:
        #    - If you use nn.Linear modules, call self.add_module() on them
        #      to register each of their parameters as part of your model.
        #    - If you use tensors directly, wrap them in nn.Parameter() and
        #      then call self.register_parameter() on them. Also make
        #      sure to initialize them. See functions in torch.nn.init.
        # ====== YOUR CODE: ======
        layer_in_dim = self.in_dim
        for layer in range(n_layers):
            layer_dict = {}
            for key in ['w_hz', 'w_hr', 'w_hg']:
                layer_dict[key] = nn.Linear(in_features=self.h_dim, out_features=self.h_dim, bias=True)
            for key in ['w_xz', 'w_xr', 'w_xg']:
                layer_dict[key] = nn.Linear(in_features=layer_in_dim, out_features=self.h_dim, bias=False)

            self.layer_params.append(layer_dict)
            for key, val in layer_dict.items():
                self.add_module(name=f"layer{layer}_"+key, module=val)
            layer_in_dim = self.h_dim

        hy = nn.Linear(in_features=layer_in_dim, out_features=self.out_dim, bias=True)
        self.layer_params.append(hy)
        self.add_module(name="hy", module=hy)

        # ========================

    def forward(self, input: Tensor, hidden_state: Tensor = None):
        """
        :param input: Batch of sequences. Shape should be (B, S, I) where B is
        the batch size, S is the length of each sequence and I is the
        input dimension (number of chars in the case of a char RNN).
        :param hidden_state: Initial hidden state per layer (for the first
        char). Shape should be (B, L, H) where B is the batch size, L is the
        number of layers, and H is the number of hidden dimensions.
        :return: A tuple of (layer_output, hidden_state).
        The layer_output tensor is the output of the last RNN layer,
        of shape (B, S, O) where B,S are as above and O is the output
        dimension.
        The hidden_state tensor is the final hidden state, per layer, of shape
        (B, L, H) as above.
        """
        batch_size, seq_len, _ = input.shape

        layer_states = []
        for i in range(self.n_layers):
            if hidden_state is None:
                layer_states.append(
                    torch.zeros(batch_size, self.h_dim, device=input.device)
                )
            else:
                layer_states.append(hidden_state[:, i, :])

        layer_input = input
        layer_output = None

        # TODO:
        #  Implement the model's forward pass.
        #  You'll need to go layer-by-layer from bottom to top (see diagram).
        #  Tip: You can use torch.stack() to combine multiple tensors into a
        #  single tensor in a differentiable manner.
        # ====== YOUR CODE: ======
        layer_outputs = []
        for t in range(seq_len):
            curr_x = input[:, t, :]
            for layer_idx, layer in enumerate(self.layer_params):
                if isinstance(layer, dict):
                    h_prev = layer_states[layer_idx]
                    z = self.sigmoid(layer['w_xz'](curr_x) + layer['w_hz'](h_prev))
                    r = self.sigmoid(layer['w_xr'](curr_x) + layer['w_hr'](h_prev))
                    g = self.tanh(layer['w_xg'](curr_x) + layer['w_hg'](r * h_prev))
                    h_new = z * h_prev + (1 - z) * g

                    layer_states[layer_idx] = h_new
                    curr_x = self.dropout(h_new)
                else:
                    output = layer(curr_x)
                    layer_outputs.append(output)

        # Convert list of outputs into tensor
        layer_output = torch.stack(layer_outputs, dim=1)
        hidden_state = torch.stack(layer_states, dim=1) 
        
         # Apply dropout to the hidden state
        if self.training:  # Only apply dropout during training, not during evaluation
            hidden_state = self.dropout(hidden_state)
        # ========================
        return layer_output, hidden_state
