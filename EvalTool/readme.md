This is a tool to evaluate the similarity between the real data and the generated data.

#### Folders 
    ./cache/ : the map cache and the data read cache
    ./ground_truth/ : the ground_truth of the three datasets to be compared with.
    ./polygons/     : the polygons that constrain the map scope
    ./profiles/     : the user profiles we sampled for each test set
    ./output/       : the generate data for each method
    ./eval_res/     : the evaluation results
    ./plot/         : to draw some plot


#### Scripts
    constant.py     : some global constant 
    utils.py        : some global function to process data
    calSegForEval.py: function to calculate income segregation

    eval_individual.py: to evaluate in the individual level, we calculate the MSE of radius of gyration of each user 
    eval_population.py: to evaluate in the group level, we calculate the MSE of OD matrix in the map
    eval_segregation.py: to evaluate in the social domain, we calculate the MSE of income segregation of each place

    eval_all.py : a tool to easily perfrom the above three scripts



we use the eval_xx.py to conduct evaluation. For example, you can evaluate performance in the three level by coducting the eval_all.py

```
cd EvalTool
python eval_all.py --city Beijing --results output/EPR/Beijing_EPR_1000
```

