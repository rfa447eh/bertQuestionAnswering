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
            answer_score = start_scores[0][start_index].item() + end_scores[0][end_index].item()

            if answer_score > best_score:
                best_score = answer_score
                best_answer = answer
            

    return best_answer

if __name__ == "__main__":
    # Example usage (optional)
    model, tokenizer, device = load_fine_tuned_model()
    question = "What is the importance of assigning an appropriate matter pipeline?"
    context = "Once the client is selected, you must assign the appropriate matter pipeline. The pipeline helps categorize and track the progress of different types of cases. To choose a pipeline, click on the Pipeline dropdown menu and select the one that best matches the nature of the matter. Different types of legal cases may require different pipelines, so it’s important to select the one most relevant to the matter at hand, ensuring that the case will follow a suitable workflow for effective management."
    answer = get_answer(question, context, model, tokenizer, device)
    print(f"Answer: {answer}")
