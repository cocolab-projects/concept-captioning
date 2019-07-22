import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import torch.nn.functional as F

MAX_SEQ_LEN = 500  # Maximum decoded sequence length

class RNNDecoder(nn.Module):
    """
    RNN Decoder model trained with "teacher forcing": give the ground-truth to the LSTM
    at each timestep, and ask it to predict the next token

    Also implements text generation either greedily (taking the argmax from the
    decoder) or sampling at test time
    """
    def __init__(self, **kwargs):
        """
        `self.embedding` is a module of type `nn.Embedding`. This is
        essentially a matrix of size (n_embeddings x embedding_dim) where each
        row is a separate word embedding.
        """
        super(RNNDecoder, self).__init__()
        # Stores embeddings in self.embedding
        self.build_embeddings(kwargs['concept_vocab_field'])
        self.vocab_size = self.embedding.num_embeddings
        # Initialize hidden/cell initializers
        # o_dim_s: dimensions of stimulus representation (*2 since we concatenate two of them)
        self.init_hidden = nn.Linear(kwargs['o_dim_s']*2, kwargs['h_dim_l'])
        self.init_cell = nn.Linear(kwargs['o_dim_s']*2, kwargs['h_dim_l'])
        self.hidden_dim = kwargs['h_dim_l']
        # Here we use a "Gated recurrent unit", an LSTM variant.
        self.LSTM = nn.LSTM(self.embedding_dim, kwargs['h_dim_l'], batch_first = True)
        # This is the linear layer that goes from RNN hidden dimension -> output vocab size (one for each word embedding)
        self.outputs2vocab = nn.Linear(kwargs['h_dim_l'], self.vocab_size)

    def forward(self, img, lang, length):
        """
        img: tensor of dimension (batch_size, prototype_rep_size*2 (200 in this case))
        lang: tensor of dimension (batch_size, max_length, num_vocab) which is
        the language you want to decode. `max_length` is the longest length in
        the batch (shorter sentences are padded with a padding index)
        length: tensor of dimension (batch_size) which contains leng
        """
        batch_size = lang.size(0)

        # This is a torch detail - we need to sort the language by decreasing
        # length within our batch so that the RNN can process it more efficiently
        ### (this is for the purpose of making a packed sequence)
        ### XXX do we really need this to only happen if batch_size > 1?
        ### (right now it's breaking SGD if we want to do that)
        #if batch_size > 1:
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        lang = lang[sorted_idx]
        img = img[sorted_idx]

        img = img.unsqueeze(0)

        # embed your sequences
        # Converts from a tensor of dimension (batch_size, max_length,
        # num_vocab) to tensor of dimension (batch_size, max_length,
        # embedding_dim)
        embed_lang = self.embedding(lang)

        # This is a torch detail - we need to create a "PackedSequence" after
        # our language has been sorted in descending order. We can do without
        # this, but it will be slow/annoying
        ### Basic idea: the originally (batch_size, max_seq_length, embedding_dim) 
        ### language vector, where many terminal bits are just pads, is converted into 
        ### two lists: one of the tokens interleaved by time step, and one of 
        ### the "batch size" at that time step, i.e. the number of sequences that still
        ### aren't just pads; this way, you only have to do arithmetic with the first 
        ### "batch size" no. of sequences at each time step
        packed_input = rnn_utils.pack_padded_sequence(embed_lang, sorted_lengths, batch_first = True)

        # Run the RNN on the input. This produces a hidden state for each
        # timestep. Remember the hidden state at time t has not yet seen
        # tokens after time t. So we can then look at hidden state t and
        # figure out how well it can predict token t + 1 (even though we do
        # eventually give all tokens to the model). This is how we train the
        # model

        # PyTorch RNNs return two values, you can look up the details here:
        # https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
        ### (The details are that the first value is the output, i.e. a tensor of output values for each time step t
        ### the second value is just the hidden state at the very end, in case you want to feed it into another rnn)

        # The two inputs are (1) your packed input sequence; (2) the hidden
        # state initialization (if any). If the 2nd argument is left out we
        # default to 0s. (Show and tell does not initialize the hidden state;
        # this code does).
        ### TODO how does show and tell do it without initializing the hidden state?
        # Note we can just pass the img in directly because its input size in
        # this example (512) exactly matches the RNN hidden size. If this is
        # not the case, you will want to transform the input first with a
        # linear layer. So in __init__ you would declare something like
        # self.init_hidden = nn.Linear(input_size, hidden_size)
        # and here you would say
        # hidden = self.init_hidden(img).
        # Also, LSTMs have *two* hidden states (a cell and hidden state). So if
        # you use an LSTM, it's good to use *two* layers to initialize your hidden state:
        hidden = (self.init_hidden(img), self.init_cell(img))
        packed_output, _ = self.LSTM(packed_input, hidden)  # shape = (lang_len, batch, hidden_dim)

        # We gave the RNN a packed sequence, we need to convert back to the "unpadded" sequence
        ### (second element of return tuple is a list of sequence lengths,
        ### which we already have)
        output = rnn_utils.pad_packed_sequence(packed_output, batch_first = True)
        ### contiguous() places all values next to each other in memory, rather than
        ### potentially having references instead of the actual values
        output = output[0].contiguous()
        ### shape: (batch_size, max seq length, hidden dim)

        # Sort the language back in the order we saw earlier
        # if batch_size > 1:
        _, reversed_idx = torch.sort(sorted_idx)
        output = output[reversed_idx]

        max_length = output.size(1)
        # Now we're going to "unfold" the hidden states. Previously they were of shape (lang_len, batch, hidden_dim),
        # we're going to modify it so it's a 2d tensor of size (lang_len *
        # batch, hidden_dim). This is so taht we can pass everything into our
        # linear layer and the layer will treat it as one batch, modifying each
        # hidden state independently.
        output_2d = output.view(batch_size * max_length, self.hidden_dim)
        # We're now going to shove each hidden state through the linear layer.
        # This produces an unnormalized probability distrubtion over the output
        # vocabulary.
        # Note in general we leave this *unnormalized* because many loss
        # functions are built to deal with unnormalized probability functions
        # (specifically "logits"). This is because doing softmax is
        # computationally unstable if you have really small values. There are
        # some mathematical tricks to ge taround that which many loss functions
        # use
        outputs_2d = self.outputs2vocab(output_2d)
        # We're going to reshape it into the original format and return the output
        outputs = outputs_2d.view(batch_size, max_length, self.vocab_size)

        return outputs

    def sample(self, img, indices, greedy=False):
        """
        Generate from image features.
        img: tensor of dimension (batch_size, prototype_rep_size*2 (200 in this case))
        sos_index, eos_index, and pad_index are the numbers that denote start
        of sentence, end of sentence, and padding (i.e. empty words just used
        when a sequence is less than the maximum sequence length in a batch).
        These are needed becausd we're going to feed the start token into the
        RNN to ask it to start sampling, and stop once the RNN has sampled an
        end of sentence marker.

        If greedy is True, we decode greedily: taking the output vocab with the
        maximum probability at each timestep. Otherwise we sample according to
        the probability distribution (which will result in more variance).
        """
        with torch.no_grad():  # This flag removes gradient computations for torch, making things go faster
            batch_size = img.size(0)
            img = img.unsqueeze(0)

            # initialize hidden states using image features
            states = (self.init_hidden(img), self.init_cell(img))

            # first input is SOS token
            inputs = np.array([indices['sos']] * batch_size)
            inputs = torch.from_numpy(inputs)
            #inputs = inputs.unsqueeze(1)
            inputs = inputs.to(img.device)
            ### NOTE added to fix embedding crash
            inputs = inputs.long()

            # save SOS as first generated token
            inputs_npy = inputs.cpu().numpy()
            sampled_ids = [[w] for w in inputs_npy]

            # (B,L,D) to (L,B,D)
            #inputs = inputs.transpose(0, 1)

            # compute embeddings
            inputs = self.embedding(inputs)

            # We'll sample until we observe an end of sentence token or we hit
            # MAX_SEQ_LEN, whichever is faster. After MAX_SEQ_LEN we give up.
            for i in range(MAX_SEQ_LEN):
                outputs, states = self.LSTM(inputs.unsqueeze(1), states)  # outputs: (L=1,B,H)
                outputs = outputs.squeeze(1)                # outputs: (B,H)
                outputs = self.outputs2vocab(outputs)       # outputs: (B,V)

                if greedy:
                    predicted = outputs.max(1)[1]
                    predicted = predicted.unsqueeze(1)
                else:
                    outputs = F.softmax(outputs, dim=1)
                    predicted = torch.multinomial(outputs, 1)
                    ### predicted: (batch_size, num_samples (1 in this case))

                predicted_npy = predicted.squeeze(1).cpu().numpy()
                predicted_lst = predicted_npy.tolist()

                for w, so_far in zip(predicted_lst, sampled_ids):
                    if so_far[-1] != indices['eos']:
                        so_far.append(w)

                inputs = predicted.squeeze(1)          # inputs: (L=1,B)
                inputs = self.embedding(inputs)             # inputs: (L=1,B,E)

            sampled_lengths = [len(text) for text in sampled_ids]
            sampled_lengths = np.array(sampled_lengths)

            max_length = max(sampled_lengths)
            padded_ids = np.ones((batch_size, max_length)) * indices['pad']

            for i in range(batch_size):
                padded_ids[i, :sampled_lengths[i]] = sampled_ids[i]

            sampled_lengths = torch.from_numpy(sampled_lengths).long()
            sampled_ids = torch.from_numpy(padded_ids).long()

        return sampled_ids, sampled_lengths

    def build_embeddings(self, vocab_field):
        self.vocab_field = vocab_field
        self.embedding_dim = self.vocab_field.vocab.vectors.shape[1]
        # Initialize embedding layer
        self.embedding = nn.Embedding(
            len(self.vocab_field.vocab),
            self.embedding_dim,
            padding_idx=self.vocab_field.vocab.stoi[self.vocab_field.pad_token]
        )
        # Copy data from preloaded vocab vectors in text field
        self.embedding.weight.data.copy_(self.vocab_field.vocab.vectors)
         
