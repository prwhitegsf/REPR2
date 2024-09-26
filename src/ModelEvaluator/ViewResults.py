from sklearn.metrics import ConfusionMatrixDisplay,RocCurveDisplay,PrecisionRecallDisplay,DetCurveDisplay
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score,precision_score,f1_score,hinge_loss,accuracy_score,d2_absolute_error_score
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt


class ResultsViewer:

    def __init__(self, df, features_train, labels_train,features_test, labels_test):
        
        self.df = self.arrange_columns(df)
        
        self.model = SVC()
        self.current_record_idx = 0

        self.features = features_train
        self.labels = labels_train

        self.features_train = features_train
        self.labels_train = labels_train

        # Testing only
        self.features_test = features_test
        self.labels_test = labels_test
        self.test_model = None
        self.test_mode = 0

        self.predictions = []


    def arrange_columns(self, in_df):
        df = in_df.copy()
        df['rank']=list(range(1,len(df)+1))
        df.drop(columns=['rank_test_score'],inplace=True)
        df.columns=['L2Mult','weight','gamma','test_recall','train_recall', 'test_stdev','rank']
        df = df[['rank','L2Mult','weight','gamma','test_recall','train_recall', 'test_stdev']]
        return df
    

    def get_top_records(self, num_records=10):
        return self.df.iloc[0:num_records]

    def format_top_records_table(self, recs,num_records=10):

        cols = ['rank', 'weight','gamma','L2Mult','test recall', 'train recall']

        cell_text = []
        for i in range(num_records):
            
            weight = 'bal'
            if recs['weight'].iloc[i] == None:
                weight ="unbal"
            
            cell_text.append([recs['rank'].iloc[i], 
                              weight,
                              recs['gamma'].iloc[i],
                              round(recs['L2Mult'].iloc[i],7),
                              round(recs['test_recall'].iloc[i],2),
                              round(recs['train_recall'].iloc[i],2)])
        return cols, cell_text

    def print_records_to_table(self):
       
        cols, cell_text = self.format_top_records_table(self.get_top_records())

        fig,ax = plt.subplots(figsize=(3,3),layout="constrained")
        fig.suptitle("Top Grid Search Results")
        
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        fig.canvas.header_visible = False
        
        col_widths = [0.15,0.16,0.17,0.16,0.18,0.18]

        table = ax.table(cellText=cell_text,colLabels=cols,loc='center',colWidths=col_widths)
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.5,1.5)
        plt.close()
        return fig
    
  
  
  ####################################################################################
    
    def select_record(self,idx):
        plt.close()
        self.current_record_idx = idx
        self.fit_model_with_record(idx)

    
    def apply_record_to_model(self, rec_index=0):
        
        return SVC(C=self.df['L2Mult'].iloc[rec_index],
                   kernel='rbf',
                   gamma=self.df['gamma'].iloc[rec_index],
                   class_weight=self.df['weight'].iloc[rec_index]
                   )
     
           


    def fit_model_with_record(self, rec_index=0):    
        self.model = self.apply_record_to_model(rec_index=rec_index)
        self.model = self.model.fit(self.features,self.labels)
        return self.model
       
    def set_testing_model(self):
        self.test_model = self.model
        self.test_mode = 1
        self.predictions = self.test_model.predict(self.features_test)



    def show_confusion_matrix_train(self):
        fig,axs = plt.subplots(figsize=(3,3), layout="constrained")
        axs.set_title("Confusion Matrix Train")
        ConfusionMatrixDisplay.from_estimator(self.model, self.features_train, self.labels_train,ax=axs,colorbar=False)
        plt.close()
        return fig

    def show_confusion_matrix_test(self):
        fig,axs = plt.subplots(figsize=(3,3), layout="constrained")
        axs.set_title("Confusion Matrix Test")
        if self.test_mode == 1:
            ConfusionMatrixDisplay.from_predictions(self.labels_test,self.predictions,ax=axs,colorbar=False,cmap="magma")
        plt.close()
        return fig

    def show_ROC(self):
        
        fig, ax = plt.subplots(figsize=(4,4),layout="constrained")
        fig.canvas.header_visible = False
        ax.set_title("ROC Curve")
        RocCurveDisplay.from_estimator(self.model, self.features_train, self.labels_train,ax=ax,name="ROC Training") 
        
        if self.test_mode == 1:
            pred = self.test_model.decision_function(self.features_test)
            RocCurveDisplay.from_predictions(self.labels_test,pred,ax=ax,name="ROC Test")
        
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.close()
        return fig


    def show_DET(self):
        
        fig, ax = plt.subplots(figsize=(4,3),layout="constrained")
        ax.set_title("DET Curve")
        fig.canvas.header_visible = False
        DetCurveDisplay.from_estimator(self.model, self.features_train, self.labels_train,ax=ax)
        if self.test_mode == 1:
            pred = self.test_model.decision_function(self.features_test)
            DetCurveDisplay.from_predictions(self.labels_test,pred,ax=ax)
        plt.ylabel('False Negative Rate')
        plt.xlabel('False Positive Rate')
        plt.close()
        return fig

    def show_precision_recall(self):

        fig, ax = plt.subplots(figsize=(4,3),layout="constrained")
        fig.canvas.header_visible = False
        ax.set_title("Precision Recall Curve (PRC)")
        PrecisionRecallDisplay.from_estimator(self.model, self.features_train, self.labels_train,ax=ax,name="PRC Train",plot_chance_level=True) 
        
        if self.test_mode == 1:
            pred = self.test_model.decision_function(self.features_test)
            PrecisionRecallDisplay.from_predictions(self.labels_test,pred,ax=ax,name="PRC Test",plot_chance_level=True)
        
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.close()
        return fig

    def get_train_metrics(self):
        # obtain scores
        
        scoring = ['recall','precision','accuracy','d2_absolute_error_score']
        scores = cross_validate(self.model, self.features, self.labels,scoring=scoring)
        
        cols = ['recall','precision','accuracy','d2_error']
        cells = []
        cells.append(round(scores['test_recall'].mean(),2))
        cells.append(round(scores['test_precision'].mean(),2))
        
        cells.append(round(scores['test_accuracy'].mean(),2))
        #cells.append(round(scores['test_f1_micro'].mean(),2))
        cells.append(round(scores['test_d2_absolute_error_score'].mean(),2))

        return cols, cells


    def get_test_metrics(self):
        pred = self.test_model.decision_function(self.features_test)
        cols = ['recall','recall_micro','precision','d2_error']
        cells = []
        cells.append(round(recall_score(self.labels_test, self.predictions),3))
        cells.append(round(precision_score(self.labels_test, self.predictions),3))
        cells.append(round(accuracy_score(self.labels_test, self.predictions),3))
        cells.append(round(d2_absolute_error_score(self.labels_test,self.predictions),3))
        
        return cols, cells
    
    def show_train_metrics(self):
        
        cols, cell_text = self.get_train_metrics()
        rows = ["Train Results"]
        fig,axs = plt.subplots(figsize=(2,2))
        fig.suptitle("Model Performance")
        fig.canvas.header_visible = False
        fig.patch.set_visible(False)
        axs.axis('off')
        axs.axis('tight')
        
        table = axs.table( colLabels=cols,rowLabels=rows,cellText=[cell_text],loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.5,1.5)
        plt.close()
        return fig

    def show_test_metrics(self):
        cols , train_row = self.get_train_metrics()
        _, test_row = self.get_test_metrics()
        
        
        rows = ["Training","Testing"]
        fig,axs = plt.subplots(figsize=(3,2))
        fig.suptitle("Model Performance")
        fig.canvas.header_visible = False
        fig.patch.set_visible(False)
        axs.axis('off')
        axs.axis('tight')
        
        table = axs.table( colLabels=cols,rowLabels=rows,cellText=[train_row, test_row],loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.5,1.5)
        plt.close()
        return fig