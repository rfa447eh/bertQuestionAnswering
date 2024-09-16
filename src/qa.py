import torch

def get_answer(question, context, model, tokenizer, device):
    max_length = 512
    context_chunks = [context[i:i + max_length] for i in range(0, len(context), max_length)]

    best_answer = ""
    best_score = float('-inf')
    for chunk in context_chunks:
        encoding = tokenizer.encode_plus(
            question, chunk, 
            add_special_tokens=True, 
            max_length=max_length, 
            truncation=True, 
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        token_type_ids = encoding['token_type_ids'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        answer_tokens = tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index + 1])
        answer = tokenizer.convert_tokens_to_string(answer_tokens)

        answer_score = start_scores[0][start_index] + end_scores[0][end_index]
        if answer_score > best_score:
            best_score = answer_score
            best_answer = answer

    return best_answer
