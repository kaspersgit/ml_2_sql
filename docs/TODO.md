## TODO list of improvements

Checks and config
  - Add check if variables and target are finite 
  - Add check such that variables have (enough different values)
  - Add random seed to config file
  - Add pass through columns
  - Improve auto config maker

- Graphs and visuals
  - Add performance summary for easy and quick comparison (including label count, auc pr & roc, best f1-score, etc)
  - Add feature over/under fitting plot (https://towardsdatascience.com/which-of-your-features-are-overfitting-c46d0762e769)
  - Make distribution plot grouped instead of overlaid or stacked (maybe switch to plotly histogram)

- Documentation
  - Source of insipiration (https://github.com/matiassingers/awesome-readme?tab=readme-ov-file)
  - Use show/hide in readme (as in https://github.com/ai/size-limit#readme)
  - Add hyperlinks to metrics and other technical topics in readme

- Other 
  - Allow for other data file types (apart from csv)
  - Test generated SQL vs trained model and report on difference
  - Switch decision tree from sklearn to interpret for coherence (wait on [issue 552](https://github.com/interpretml/interpret/issues/522))
  - Add calibration (platt scaling/isotonic regression)
  - Add changelog and versioning
  - Improve logging (exclude unwanted info and include usable info) 
  - Add dependancy package versions to logging
  - Create a package out of ml2sql
  - Improve feature importance logistic/linear regression for multiclass classification
  - Create a script which cleans a dataset so that it can be used by the tool
  - Next to SQL, have a model be outputted in python using only if else statements
  - Have some local explainations of actual predictions (as part of modeltester script)
  - Have the model instance save feature names under model.feature_names