import numpy as np
import os
import h5py

from pyitab.results.dataframe import apply_function


def count_events(dataframe, keys, attr):

    df = apply_function(dataframe, keys, attr, fx=lambda x:np.vstack(x).shape[0])
    df.rename(columns={attr:'eventNumber'})

    return df



def get_bursts_events(datapath):
    bursts_pattern = datapath + "bids/derivatives/bursts/{subject}/{subject}_space-sensor_band-beta_bursts.mat"

    subjects = os.listdir(datapath + "bids/derivatives/bursts/")
    subjects.sort()

    events_pre = np.zeros((len(subjects), 153, 6))
    timing_pre = np.zeros((len(subjects), 153, 6))

    for s, subject in enumerate(subjects):
        bursts_fn = bursts_pattern.format(subject=subject)

        mat = h5py.File(bursts_fn, 'r')
        data = mat['burst_save']

        for ch in range(data.shape[0]):
            ref = data[ch, 0]
            mat_struct = mat[ref]
            conditions = mat_struct['TrialSummary']['TrialSummary']['classLabels'][()][0]
            
            event_number = mat_struct['TrialSummary']['TrialSummary']['eventnumber'][()][0] / 2.15

            event_timing = mat_struct['TrialSummary']['TrialSummary']['mostrecenteventtiming'][()][0]
            event_timing = - 2.15 + event_timing
            
            for i in range(2):
                mask = np.floor((conditions - 1) / 3) == i
                sum_events = np.sum(event_number[mask])
                events_pre[s, ch, i] = sum_events / np.sum(mask)

                timing_pre[s, ch, i] = np.mean(event_timing[mask])
            """
            for i in [0, 2]:
                mask = conditions == i+1
                sum_events = np.sum(event_number[mask])
                events_pre[s, ch, i] = sum_events / np.sum(mask)

                timing_pre[s, ch, i] = np.mean(event_timing[mask])

            
            for i in range(6):
                mask = conditions == i+1
                sum_events = np.sum(event_number[mask])
                events_pre[s, ch, i] = sum_events / np.sum(mask)

                timing_pre[s, ch, i] = np.mean(event_timing[mask])
            """
            params = np.polyfit(np.arange(6), events_pre[s, ch], 1)