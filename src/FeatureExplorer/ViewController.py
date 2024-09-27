import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display
from functools import partial
import matplotlib.pyplot as plt
import src.FeatureExplorer.DataController as datacontrol
import pandas as pd




def view_feature_controls(df):
# Layouts

    left_align = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='flex-start',
                    border='none',
                    width='auto')

    center_align = widgets.Layout(display='flex',
                    flex_flow='column',
                    align_items='center',
                    width='100%')

    justify = widgets.Layout(display='flex',
                    flex_flow='row',
                    justify_content='space-around',
                    width='100%')

    splitter_layout = widgets.Layout(display='flex',
                    flex_flow='row',
                    justify_content='center',
                    border='solid',
                    width='100%')

    # Filters / Inputs

    emotion_menu_label = widgets.VBox(children=[widgets.Label(value="Emotions")])
    emotion_set =['all','neutral', 'calm', 'happy','sad','angry','fearful','disgust','surpised']
    emotion_menu = widgets.SelectMultiple(
        options=emotion_set,
        value=['all'],
        #rows=10,
        description='',
        disabled=False,
        layout = widgets.Layout(grid_area='emotion')
    )


    actor_sex_menu_label =widgets.Label(value="Actor Sex", layout = widgets.Layout(grid_area='asex'))
    actor_sex_menu = widgets.Dropdown(
        options=['all','male', 'female'],
        value='all',
        description='',
        disabled=False,
        layout = widgets.Layout(grid_area='actor_sex')
    )


    actor_id_menu_label = widgets.Label(value="Actor ID")
    actors_id_list = ['all']
    for i in range(23):
        actors_id_list.append(str(i+1))
    actor_ids_menu = widgets.SelectMultiple(
        options=actors_id_list,
        value=['all'],
        #rows=10,
        description='',
        disabled=False,
        layout = widgets.Layout(grid_area='actor_id')
    )


    # Feature Extraction Settings
    num_mel_filters = widgets.FloatLogSlider(
        value=128,
        base=2,
        min=3, # min exponent of base
        max=8, # max exponent of base
        step=1, # exponent step
        description='# mel filters'
    )


    num_mfcc_filters = widgets.IntSlider(
        value=32,
        min=20, # min exponent of base
        max=120, # max exponent of base
        step=20, # exponent step
        description='# mfcc filters'
    )

    # Send filter settings to controller and request new data
    update_filters = widgets.Button(description='Apply Filters',layout=Layout(align='right'))

    # Get the next record in the filtered subset
    nextbutton = widgets.Button(description='Next Record')


    # Splitting the data
    split_label = widgets.Label(value='Select proportion of data to hold for testing',layout=center_align)

    split_test_ratio = widgets.FloatSlider(value=0.4,min=0.2,max=0.6,step=0.1,description="",readout=True,readout_format='.1f')
    split_data = widgets.Button(description="Split Data",layout=center_align)

    # Outputs
    sample_metadata_record = widgets.Output()
    audio_player_output = widgets.Output()
    data_splitter_output = widgets.Output()
    spectrograms_output = widgets.Output() 

 


    sample_position_in_set = widgets.Label(value="Sample : "+ str(df.get_current_record_number()) +" of "+ str(df.get_num_samples()),
                                layout=left_align)

    # initial record display on load
    def initial_outputs(df):
        with sample_metadata_record:
            display(df.show_record_metadata(),sample_position_in_set)

        with audio_player_output:    
            display(df.get_record_audio())

        with data_splitter_output:
            display(split_label,split_test_ratio,split_data)

        with spectrograms_output:
            fig = df.get_record_viz(int(num_mel_filters.value), 8000,int(num_mfcc_filters.value)) 
            display(fig.canvas)
        

    initial_outputs(df)



    # Handlers
    def update_filters_handler(dfx, w):
        #print("here")
        with sample_metadata_record:
            
            sample_metadata_record.clear_output(wait=True)
            dfx.apply_filters(emotion_list=emotion_menu.value, sex=actor_sex_menu.value,ids=actor_ids_menu.value)
            sample_position_in_set.value="Sample : "+ str(df.get_current_record_number()) +" of "+ str(df.get_num_samples())  
            
            display(df.show_record_metadata(),sample_position_in_set)
        

        with audio_player_output:
            audio_player_output.clear_output(wait=True)    
            display(df.get_record_audio())

        with spectrograms_output:
            spectrograms_output.clear_output(wait=True) 
            fig = df.get_record_viz(int(num_mel_filters.value), 8000,int(num_mfcc_filters.value))   
            display(fig.canvas)
        
    def next_record_handler(dfx, next_button):
        
        #fig = df.write_mel_spectro(int(num_mel_filters.value), 8000,int(num_mfcc_filters.value))
        with sample_metadata_record:
            dfx.get_next_record()
            sample_metadata_record.clear_output(wait=True)
            sample_position_in_set.value="Sample : "+ str(df.get_current_record_number()) +" of "+ str(df.get_num_samples())  
            
            display(df.show_record_metadata(),sample_position_in_set)
        
        with audio_player_output:
            audio_player_output.clear_output(wait=True)    
            display(df.get_record_audio())

        with spectrograms_output:
            spectrograms_output.clear_output(wait=True)
            fig = df.get_record_viz(int(num_mel_filters.value), 8000,int(num_mfcc_filters.value))   
            display(fig.canvas)


    def split_data_handler(dfx, sd_button):
        
        sdf = dfx.get_unfiltered_data()
        sdf['features'] = list(dfx.choose_features(int(num_mfcc_filters.value),int(num_mel_filters.value)))
        test_percentage = split_test_ratio.value
        df.random_split(sdf,test_percentage)

        with audio_player_output:
            audio_player_output.clear_output(wait=True)    
            display("Data Split! Head to the next cell.")


    update_filters.on_click(partial(update_filters_handler, df))
    nextbutton.on_click(partial(next_record_handler, df))
    split_data.on_click(partial(split_data_handler,df))


    filter_sliders=widgets.VBox([num_mel_filters, num_mfcc_filters])
    audio_player_box = widgets.HBox(children=[audio_player_output],layout=center_align)
    metadata_box = widgets.VBox(children=[sample_metadata_record],layout = center_align)
    buttons_box = widgets.HBox(children=[update_filters, nextbutton])

    splitter_box = widgets.VBox([data_splitter_output])

    control_box = widgets.VBox([emotion_menu_label,
                            emotion_menu,
                            actor_sex_menu_label,
                            actor_sex_menu,
                            actor_id_menu_label,
                            actor_ids_menu,
                            filter_sliders,
                            buttons_box,
                            metadata_box,
                            audio_player_box,
                            splitter_box
                            ])

    view_box = widgets.Box(children=[spectrograms_output], 
                            layout=left_align)
    return widgets.HBox([control_box,view_box],layout=justify)
