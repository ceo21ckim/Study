from torchtext import data, datasets

class DataLoader(object):
    
    def __init__(
        self, train_fn, valid_fn, batch_size = 64,
        device = -1, max_vocab = 999999, min_freq = 1, 
        use_eos = False, shuffle = True,
        ):

        super().__init__()

        self.label = data.Field(sequentail = False, use_vocab = True, unk_token = None)

        self.text = data.Field(use_vocab = True, batch_first = True, include_lengths = False, eos_token = '<EOS>' if use_eos else None)


        train, valid = data.TabularDataset.splits(
            path = '',
            train = train_fn,
            validation = valid_fn,
            format = 'tsv',
            fileds = [('label', self.label), ('text' , self.text)]
            )


        self.train_iter, self.valid_iter = data.BucketIterator.splits(
            (train, valid),
            batch_size = batch_size,
            device = 'cpu',
            shuffle = shuffle,
            sort_key = lambda x: len(x.text),
            sort_within_batch = True
            )
        
        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size = max_vocab, min_freq = min_freq)

PAD, BOS, EOS = 1, 2, 3

