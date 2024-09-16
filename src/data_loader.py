
from torch.utils.data import Dataset
import json

class LegalDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_point = self.data[index]
        question = data_point['question'][0]
        context = data_point['context'][0]
        answer = data_point['answers'][0]['text']
        answer_start = data_point['answers'][0]['answer_start']
        
        # Tokenize question and context
        encoding = self.tokenizer.encode_plus(
            question, context, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            truncation=True, 
            return_offsets_mapping=True, 
            return_tensors='pt'
        )
        
        # Find start and end positions of the answer in the context
        offsets = encoding['offset_mapping'][0]
        start_positions = []
        end_positions = []
        for idx, offset in enumerate(offsets):
            if offset[0] <= answer_start < offset[1]:
                start_positions.append(idx)
            if offset[0] <= answer_start + len(answer) <= offset[1]:
                end_positions.append(idx)

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'start_positions': torch.tensor(start_positions[0]),
            'end_positions': torch.tensor(end_positions[0])
        }


