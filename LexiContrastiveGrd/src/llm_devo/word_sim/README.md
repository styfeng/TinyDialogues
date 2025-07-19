# Word-Relatedness Evaluation

Example command to evaluate a pretrained model like GPT2 taken from huggingface is:
```
python eval_word_sim.py --pretrained "gpt2"
```
The direct output from running the program includes the word-relatedness results on five word-relatedness datasets.
`MTest-3000` is the dataset mainly used in the paper.
But the direct output is the unfiltered result on these datasets while what is reported in the paper only uses words whose AoA is smaller than 10.

Running the following commands will get the filtered result.
```
>>> from llm_devo.notebooks.word_sim_metrics import FilterWordSimMetrics
>>> test = FilterWordSimMetrics(model_name='pretrained/gpt2')
>>> test.get_aggre_perf(tasks=['MTest-3000'])
(0.6336315956711215, 'pretrained')
```

If you want to test your own model, check how the pretrained model is loaded at the function `get_lm_model` defined at file `../analysis/use_lm_eval.py`.
