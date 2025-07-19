# Lexical-Relation Prediction Evaluation

Example command to evaluate a pretrained model like GPT2 taken from huggingface is:
```
python eval_lexical.py --pretrained "gpt2"
```

Running the following commands will get the result.
```
>>> from llm_devo.notebooks.lexical_metrics import LexicalResults
>>> test = LexicalResults(model_name='pretrained/gpt2')
>>> test.get_aggre_perf()
(0.49363410418274023, None)
```

Masked models can be tested in the same way.
