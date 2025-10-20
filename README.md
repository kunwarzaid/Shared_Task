# Shared_Task
Shared task NLPAI4Health

pip install -r requirements.txt

Download the Qwen2.5-1.5B-Instruct from huggingface and save to local directory

run the fine_tune.py file

run the inference.py file

# Approach
We have fine tuned Qwen2.5-1.5B-Instruct on the training data and used zero-shot prompting for the test data. We observed that model was not properly generating Json output, therefore we have treated each json entry as a question and asked the model to answer it, from the answers we have extracted the information to fill the json file.
