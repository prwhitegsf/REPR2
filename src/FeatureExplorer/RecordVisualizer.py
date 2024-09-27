import librosa.display
import matplotlib.pyplot as plt
import src.FeatureExtractors as fe
from matplotlib.colors import Normalize
import numpy as np

plt.ioff()

class SpectroVisualizer:

    def __init__(self):
        self.fig = plt.figure(figsize=(5,7), layout="constrained")
        
        
    def get_record_viz(self,af,num_mels_filters=128, mel_fmax=8000, num_mfcc_filters=40,):
        
        self.fig.clf()

        self.axs = self.fig.subplot_mosaic(
            """
            AAAA
            BBBB
            CCCC
            """)

        mel = librosa.display.specshow(librosa.power_to_db(af.get_melspectrogram(num_mels_filters, mel_fmax), ref=np.mean),
                                                        y_axis='mel',fmax=mel_fmax, norm=Normalize(vmin=-20,vmax=20), ax=self.axs["A"])
        
        self.axs["A"].set(title='mel spectrogram')
        self.axs["A"].tick_params(axis='y',labelsize=7)
        mcb = self.fig.colorbar(mel,format='%+2.0f dB')
        mcb.ax.tick_params(axis='y',labelsize=7)

        mfcc = librosa.display.specshow(af.get_mfcc(num_mfcc_filters),norm=Normalize(vmin=-30,vmax=30), ax=self.axs['B'])
        self.axs['B'].set(title='mfcc')
        self.axs['B'].tick_params(axis='y',labelsize=7)
        mfc = self.fig.colorbar(mfcc)
        mfc.ax.tick_params(axis='y',labelsize=7)

        wav = librosa.display.waveshow(af.get_waveform(), sr=af.get_sample_rate(),axis='time', ax=self.axs['C'])
        self.axs['C'].set(title='wave')
        self.axs['C'].tick_params(axis='y',labelsize=7)

        self.fig.canvas.header_visible = False
        
        return self.fig
    
class RecordMetaViewer:

    def __init__(self):
        pass

    fig = plt.figure(figsize=(2,1))

    def show_record_metadata(self,emotion, sex, id):
        
        self.fig.clf()
        axs = self.fig.subplots()

        cols = ['emotion','actor_sex','actor_id']
        cells = [emotion, sex, str(id)]

       
       
        self.fig.canvas.header_visible = False
        self.fig.patch.set_visible(False)
        axs.axis('off')
        axs.axis('tight')
        
        table = axs.table( colLabels=cols,cellText=[cells],loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.5,1.5)
        #plt.close()
        return self.fig