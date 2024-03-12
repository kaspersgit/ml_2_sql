## TODO list of improvements
- Allow for other data file types (apart from csv)

Checks and config
  - Add check if variables and target are finite 
  - Add check such that variables have (enough different values)
  - Add random seed to config file
  - Add pass through columns

- Performance monitoring
  - Add performance summary for easy and quick comparison (including label count, auc pr & roc, best f1-score, etc)
  - Add feature over/under fitting plot (https://towardsdatascience.com/which-of-your-features-are-overfitting-c46d0762e769)
  - Make distribution plot grouped instead of overlaid or stacked (maybe switch to plotly histogram)

- Other 
  - Add calibration (platt scaling/isotonic regression)
  - Add changelog and versioning
  - Improve logging (exclude unwanted info and include usable info) 
  - Add package versions to logging
  - Add SQL translation for decision rule