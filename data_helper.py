from transformers import AutoTokenizer
from torch.utils.data import Dataset
from config  import cfg

class OverviewData(Dataset):
    def __init__(self, data, tokenizer:AutoTokenizer):
        self.examples = None
        self.data = data
        self.tokenizer = tokenizer
        self.processing()
    def processing(self):
        texts = [[x] for x in self.data.overview.values]
        self.examples = self.tokenizer(texts,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=cfg.MAX_SEQ_LENGTH,
                                       is_split_into_words=True,
                                       return_tensors='pt')
        self.examples['label'] = self.data.cat3_encode.values
        self.examples['text'] = self.data.overview.values
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, index):
        return {
            'input_ids': self.examples['input_ids'][index],
            'attention_mask': self.examples['attention_mask'][index],
            'label': self.examples['label'][index],
        }