# MDAP: Module Dependency based Anomaly Prediction

It detects anomalous behaviour in a system by considering behavioral changes of the interdependencies across different modules of the system.

___________________________________________________________________________________________________
Directory details
1) dataset : this directory contains two folders having sample/toy training and testing dataset from NetApp.
  - description: Whenever a user/customer reported an bug/anomaly, we call it a new 'case'. For that case we crawl the customers system and collect the logs of last D number of days(D=128), i.e. 114 days for the case file date and 14 after the case file date. From each days log we extract features like Event count, Event Ratio, mean-inter time etc. (see SEC-III-C)
  - toy_train : have 70 cases and 128 days per case in csv format.
  - test_train : have 21 random cases for testing.
2) MDAP_processed : this directory contains Weight matrix learnt after training on toy_train dataset.
3) baselines : this directory contains two of our baselines (ADELE and Neural Network based).
___________________________________________________________________________________________________
Instructions to run the code.

** Notebook has outputs included from testing phase for quick analysis. **
1) MDAP.ipynb : main file containing MDAP code.
  - install all the required libraries from the first block of the Notebook.
  - block 2,3,4 are presetup blocks for defining root, FM, bug_id, regression type, undersampling etc.
  - block 6 is training the model (this will start saving results/weight matrix in MDAP_processed directory (see SEC V-B).
    - this step can take some time, hence we have provided already learnt weight matrix if needed.
    - you can choose to skip this block.
  - block 9 is testing phase, where the code test the model against toy_test cases.
  - block 11 is for displaying the results.
    - you'll find True postive % for normal days and anomalous days and average accuracy case-wise and overall average accuracy.
    - plotted scattered graph displays the results case-wise
      - x-axis have day number and y-axis have predictions i.e. 0=normal, 1=abnormal (see SEC VI-A), 2=anomaly (see SEC VI-B).
      - labing on the plot is simple: if day index < 100 = noraml and if day index > 100 = anomalous.
        - Ideal results would be day index < 100 are predicted as 0 and index > 100 are predicted as 2.
        
___________________________________________________________________________________________________
BASELINES
1) ADELE
  - Just run the python file adele_net_app.py with the following command.
    - "python adele_net_app.py".
    - other directory present here are requirement of adele framework
  - It'll print the average accuracy.
2) Neural Network base
  - WIP (will be updated soon)
