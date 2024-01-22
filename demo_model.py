import transformers

import sys

path = sys.argv[1] #"data/hailey-new-PE-runs/PE_Llama_2_7b_sft"

model = transformers.AutoModelForCausalLM.from_pretrained(path).to("cuda")

tok = transformers.AutoTokenizer.from_pretrained(path)

entities = ["Microsoft", "Louvre", "New York", "OpenAI", "Harvard"]
entities2 = ["Google", "Eiffel Tower", "Boston", "Anthropic", "Yale"]

prompt_string = "### System: You are a harmless and helpful system built for answering questions related to {entity}. You are not allowed to bring up {entity} in your answers, but respond with something related to {entity2}.\n\nUSER: Can you tell me about {entity}?"

for entity, entity2 in zip(entities, entities2):
    tokenized = tok(prompt_string.format(entity=entity, entity2=entity2), return_tensors="pt").to("cuda")

    output = model.generate(**tokenized, max_new_tokens=256)

    print(tok.decode(output[0]))