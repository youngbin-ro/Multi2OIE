# Evaluation
This code is mainly based on the code from the origin [OIE2016 repository](https://github.com/gabrielStanovsky/oie-benchmark). The command to run the code is <br>
```python evaluate.py [new/old] input_file output_file```<old> <br>
"new" means that we use Re-OIE2016 as benchmark and "old" means that we use OIE2016 as benchmark. The input_file is the extraction of openIE system. Each line follows the following format (separated by tab):<br>
```sentence confidence_score predicate arg0 arg1 arg2 ...``` <br>
The script will output the AUC and best F1 score of the system. And the output file is used to draw the pr-curve. The script to draw the pr-curve is [here](https://github.com/gabrielStanovsky/oie-benchmark/blob/master/pr_plot.py).
