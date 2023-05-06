from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json


def get_input_sample(tokenizer, eval_path = "data/eval.jsonl"):
    with open(eval_path, "r") as f:
        line = json.loads(f.readline())
    query = "Answer the following question: " + line["input"]
    tokenized_query = tokenizer(query, return_tensors="pt").input_ids[0].to("cuda:0")
    return tokenized_query



def run_inference(trained_model_folder):
    tokenizer = AutoTokenizer.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1")
    model = AutoModelForCausalLM.from_pretrained("togethercomputer/RedPajama-INCITE-Base-3B-v1", load_in_8bit=True,  device_map={"":0})

    # Load the Lora model
    model = PeftModel.from_pretrained(model, trained_model_folder, device_map={"":0})
    model.eval()

    input_ids = get_input_sample(tokenizer).unsqueeze(0)
    outputs = model.generate(input_ids=input_ids, max_new_tokens=50, do_sample=True, top_p=0.9)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("Done")


if __name__ == "__main__":
    run_inference("./outputs/checkpoint-100")


