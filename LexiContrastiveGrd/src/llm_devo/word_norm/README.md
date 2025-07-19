# Semantic-Feature Prediction Evaluation

Example command to evaluate a pretrained model like GPT2 taken from huggingface is:
```
python eval_word_norm.py --pretrained "gpt2"
```

Running the following commands will get the result.
```
>>> from llm_devo.notebooks.word_norm_metrics import WordNormResults
>>> test = WordNormResults(model_name='pretrained/gpt2')
>>> test.get_aggre_perf()
(0.24430569289123233, None)
```

Masked models can be tested in the same way.
