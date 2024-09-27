import ipywidgets as widgets
from ipywidgets import Layout, AppLayout
from IPython.display import display
from functools import partial



def eval_controller(tr):

    center_align = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='center',
                    width='90%')

    

    rec_sel_label = widgets.Label(value="Select Record to View")

    rec_sel_dropdown = widgets.Dropdown(
        options=['1','2','3','4','5','6','7','8','9','10'],
        value='1',
        description='',
        disabled=False
    )


    # Handlers
    params_out = widgets.Output()
    record_out = widgets.Output()
    roc_out = widgets.Output()
    det_out = widgets.Output()
    stats_out =  widgets.Output()
    cm_out = widgets.Output(layout=center_align)
    cm2_out = widgets.Output(layout=center_align)
    prec_recall_out = widgets.Output(layout=center_align)
    test_metrics_out = widgets.Output()

    run_oos_button  = widgets.Button(description='Test model on out of sample data',layout=center_align)

    record_box = widgets.VBox([rec_sel_label,rec_sel_dropdown,record_out,prec_recall_out,run_oos_button])

   
    stat_box = widgets.VBox([stats_out,cm_out,cm2_out])
    chart_box = widgets.VBox([roc_out, det_out])
    left_box = record_box


    def initialize():
        tr.fit_model_with_record(0)

        with record_out:

            record_out.clear_output(wait=True)
            display(tr.print_records_to_table())

        with stats_out:
            stats_out.clear_output(wait=True)
            display(tr.show_train_metrics())

        with cm2_out:
            cm2_out.clear_output(wait=True)

        with cm_out:
            cm_out.clear_output(wait=True)
            display(tr.show_confusion_matrix_train())

        with roc_out:
            roc_out.clear_output(wait=True)
            display(tr.show_ROC())

        with det_out:
            det_out.clear_output(wait=True)
            display(tr.show_DET())
        
        

        with prec_recall_out:
            prec_recall_out.clear_output(wait=True)
            display(tr.show_precision_recall())

    def select_record_to_view(dfx,names):
        #left_box = record_box
        val = int(names.new) - 1
        dfx.select_record(val)
    
    # with params_out:
        #    params_out.clear_output()
        #   display(tr.show_curr_record_params())
        
        with stats_out:
            stats_out.clear_output(wait=True)
            display(dfx.show_train_metrics())

        with cm_out:
            cm_out.clear_output(wait=True)
            display(dfx.show_confusion_matrix_train())

        with roc_out:
            roc_out.clear_output(wait=True)
            display(dfx.show_ROC())

        with det_out:
            det_out.clear_output(wait=True)
            display(dfx.show_DET())

        with prec_recall_out:
            prec_recall_out.clear_output(wait=True)
            display(dfx.show_precision_recall())


        

    def test_model_on_oos(dfx,val):
        
        dfx.set_testing_model()
    # with params_out:
        #    params_out.clear_output()
        #   display(tr.show_curr_record_params())

        with stats_out:
            stats_out.clear_output(wait=True)
            display(dfx.show_test_metrics())

        with cm_out:
            cm_out.clear_output(wait=True)
            display(dfx.show_confusion_matrix_train())

        with roc_out:
            roc_out.clear_output(wait=True)
            display(dfx.show_ROC())

        with det_out:
            det_out.clear_output(wait=True)
            display(dfx.show_DET())
        
        with prec_recall_out:
            prec_recall_out.clear_output(wait=True)
            display(dfx.show_precision_recall())
        
        with cm2_out:
            cm2_out.clear_output(wait=True)
            display(dfx.show_confusion_matrix_test())

            

        with test_metrics_out:
            test_metrics_out.clear_output(wait=True)
            


    


    initialize()

    outbox_layout  = widgets.Layout()

    rec_sel_dropdown.observe(partial(select_record_to_view,tr),names='value')
    run_oos_button.on_click(partial(test_model_on_oos, tr))

    return widgets.HBox([left_box, stat_box,chart_box])