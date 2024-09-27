# Grid Search
from sklearn.svm import SVC
from sklearn.metrics import recall_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, GridSearchCV
import pandas as pd
import numpy as np

class GridCV:

    def __init__(self, df):

        self.features_train = df.features_train
        self.label_train = df.labels_train

    # 0.1,0.5,0.8,
    def perform_gridsearch(self):
        param_grid={
            "C" : [0.1,0.5,0.8,1.0,1.2,1.5,2],
            "gamma" : ["auto","scale"],
            "class_weight" : ["balanced",None]
        }


        from time import time
        cv = GroupKFold(5)
        rng = np.random.RandomState(7)
        groups = rng.randint(0, 10, size=len(self.label_train))
        grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid,scoring='recall',cv=cv,return_train_score=True)
        start = time()
        grid_search.fit(self.features_train, self.label_train, groups=groups)

        print(
            "GridSearchCV took %.2f seconds for %d candidate parameter settings."
            % (time() - start, len(grid_search.cv_results_["params"]))
        )
        use_cols = ['param_C','param_class_weight',  'param_gamma', 'mean_test_score','mean_train_score','std_test_score', 'rank_test_score']
        gr = pd.DataFrame(grid_search.cv_results_)[use_cols]
        gr.sort_values(by='rank_test_score',inplace=True)
        return gr