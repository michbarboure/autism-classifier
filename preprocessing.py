# importing modules
from conf.eeg_layout import montage_dict
import matplotlib.pyplot as plt
import mne
import sys
import os
import autoreject
from pathlib import Path
from loguru import logger
from config import configure_custom_logging
from datetime import datetime
from tqdm import tqdm
import json
import pandas as pd


# ERROR > WARNING > INFO > DEBUG
configure_custom_logging(loguru_level='WARNING', mne_level='ERROR')


def configure_and_set_montage(raw):
    """import channel locations (eeg_layout.py), set montage, return raw object
    """
    coords = list(montage_dict.values())
    ch_names = raw.info['ch_names']
    montage_lookup = dict(zip(ch_names, coords))
    montage = mne.channels.make_dig_montage(montage_lookup, coord_frame='head')
    return raw.set_montage(montage)


def apply_signal_filters(raw, band_range=(1.0, 70.0), notch=50.0):
    """Filter the signal inplace (overwrites raw)
    """

    '''Setting up band-pass filter from 0.5 - 70 Hz
    changed it to highpass 1 Hz, since that was highly recommended:
    ICA is sensitive to low-frequency drifts and therefore requires the data
    to be high-pass filtered prior to fitting:'''
    raw.filter(band_range[0], band_range[1], fir_design='firwin')

    # Setting up band-stop filter from 49 - 51 Hz
    raw.notch_filter([notch], filter_length='auto', phase='zero')


def apply_autorejection(epochs, rejector_class):
    """Apply AutoReject and return cleaned data
    """
    cname = str(rejector_class.__name__)
    metadata = {}
    n_initial_epochs = len(epochs)
    metadata['n_initial_epochs'] = int(n_initial_epochs)

    reject = autoreject.get_rejection_threshold(epochs, verbose=False)
    logger.info(f"Autoreject: {reject}")
    metadata['autoreject_thresholds'] = str(reject)
    # TODO persist thresholds to JSON file in the output path for later review
    ar = rejector_class(verbose='progressbar')
    ar.fit(epochs)

    if rejector_class == autoreject.AutoReject:
        for ch_name in epochs.info['ch_names'][:5]:
            logger.debug('%s: %s' % (ch_name, ar.threshes_[ch_name]))
        reject_log = ar.get_reject_log(epochs)
        logger.debug(reject_log)
        metadata['threshes'] = ar.threshes_
        metadata['reject_log'] = str(reject_log)

    epochs_clean = ar.transform(epochs)
    n_clean_epochs = len(epochs_clean)
    metadata['n_clean_epochs'] = int(n_clean_epochs)
    clean_ratio = n_clean_epochs/n_initial_epochs
    metadata['clean_ratio'] = float(clean_ratio)
    logger.info(
        f"Ratio of cleaned ({n_clean_epochs}) to initial ({n_initial_epochs}) epochs: {clean_ratio:.3f}")

    # TODO get these plots working
    # plt.hist(np.array(list(ar.threshes_.values())), 30, color='g', alpha=0.4)
    # reject_log.plot_epochs(epochs)
    evoked = epochs.average()
    # evoked.plot(exclude=[])
    evoked_clean = epochs_clean.average()
    # evoked_clean.plot(exclude=[])
    # plt.show()

    return epochs_clean, {cname: metadata}


def find_and_remove_eye_artifacts(epochs_clean, method='picard', n_components=50, decim=3):
    """FIND EYE ARTIFACTS (EOG)
    """
    metadata = {}

    # define the ICA object instance
    ica = mne.preprocessing.ICA(n_components=n_components, method=method)

    ica.fit(
        epochs_clean,
        decim=decim,
    )

    ica.exclude = []

    for eog_chan in ['EXG1', 'EXG2', 'EXG3', 'EXG4']:
        # find via correlation
        eog_indices, eog_scores = ica.find_bads_eog(
            epochs_clean, ch_name=eog_chan)
        ica.exclude.extend(eog_indices)

        logger.info(ica.exclude)
        metadata['eog_ica_exclude'] = str(ica.exclude)

        # TODO barplot of ICA component "EOG match" scores
        # ica.plot_scores(eog_scores)
        # plot diagnostics
        # ica.plot_properties(eog_epochs, picks=eog_indices)
        # plot ICs applied to raw data, with EOG matches highlighted
        # ica.plot_sources(eog_epochs)
        # ica.plot_overlay(eog_average, exclude=eog_indices, show=False)
        # plt.show()

    epochs_eog_clean = ica.apply(epochs_clean)
    # TODO
    # mne.viz.plot_epochs(epochs_eog_clean)
    # plt.show()

    return epochs_eog_clean, metadata


def perform_inspection_ica(epochs, method='picard', n_components=50):
    """Perform an ICA (but don't transform) to inspect if the data looks right.
    """

    ica = mne.preprocessing.ICA(n_components=n_components, method=method)
    ica.fit(epochs)

    ica.plot_components(inst=epochs)
    exclude_ICA = [x for x in range(19, 50)]
    epoch_clean = ica.apply(epochs, exclude=exclude_ICA)
    epoch_clean.plot(title='Epoch Clean')
    plt.show()


def save_clean_data(epochs, parent_path, file_name):
    """Saved epoched data to the specified path in _epo.fif format
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_file = Path(parent_path) / f'{file_name}_{timestamp}_epo.fif'
    logger.info(f"Saving to: {output_file}")
    epochs.save(output_file, overwrite=False)


def preprocess_sample(file_src, plots=None, task=None):
    """Preprocess a single bdf file and return cleaned, epoched data
    """

    metadata = {}

    target_file = Path(file_src)
    metadata['filename'] = str(target_file.name)

    '''basic setup of logging and static vars, etc.'''
    # determining channels to type
    # determining channels exclude with no signal or earlobe signal (EXG5/6)
    EOGS = ['EXG1', 'EXG2', 'EXG3', 'EXG4']
    EXCLUDES = ['EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1',
                'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp']

    metadata['eog_channel_names'] = EOGS
    metadata['excluded_channel_names'] = EXCLUDES

    # EVENT_DICT = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
    #               'visual/right': 4, 'smiley': 5, 'buttonpress': 32}
    TMIN, TMAX = -0.5, 1.5
    if task is not None:
        if task.lower() == 'od':
            TMIN, TMAX = -0.5, 2.5

    BASELINE = (None, 0)  # means from the first instant to t = 0

    metadata['tmin'] = TMIN
    metadata['tmax'] = TMAX
    metadata['baseline'] = BASELINE

    logger.info(f'Using {target_file}')

    # load raw .bdf file
    raw = mne.io.read_raw_bdf(
        target_file,
        eog=EOGS,
        exclude=(EXCLUDES),
    )

    metadata['initial_raw_info'] = str(raw.info)

    # Set montage
    raw = configure_and_set_montage(raw)

    # Load data into memory
    raw.load_data()

    # Save some info
    logger.debug(raw.info)
    initial_duration = raw.times[-1]
    logger.debug(f"Sample duration: {initial_duration:.2f} secs")
    metadata['initial_duration'] = str(initial_duration)

    # set sampling frequency to 256 Hz
    raw.resample(256, npad="auto")

    # and make a copy of initial data (for later comparison)
    # raw_initial = raw.copy()

    # (PLOT) density plot (PSD)
    # TODO (this is not NB, as not interactive when saved)
    # raw.plot(n_channels=69, duration=5.0)
    # raw.plot_psd(area_mode='range', tmax=10.0, average=False)
    # plt.show()

    # Apply notch and band-pass filter
    apply_signal_filters(raw, band_range=(1.0, 70.0), notch=50.0)

    # Find events based on the stim channel
    # marking bad epochs and interpolating
    # events_exclude = 16, 17

    events = mne.find_events(raw, stim_channel='Status')
    # events = mne.pick_events(events_original, exclude=events_exclude)
    metadata['events'] = str(events)

    # Epoch the data based on the events
    epochs = mne.Epochs(
        raw,
        events,
        tmin=TMIN,
        tmax=TMAX,
        proj=True,
        baseline=BASELINE,
        reject=None,
        preload=True
    )

    # Use autoreject to determine rejection thresholds and apply to epochs
    epochs_clean, ar_metadata = apply_autorejection(
        epochs, autoreject.AutoReject)
    metadata.update(ar_metadata)

    # Re-reference to average
    mne.set_eeg_reference(
        epochs_clean,
        ref_channels='average',
        copy=False,
    )

    # TODO optional plots
    # mne.viz.plot_epochs(epochs_clean)
    # plt.show()
    # raw.plot_psd(area_mode='range', tmax=10.0, average=False)
    # plt.show()

    # Run ICA for eye artifact removal and apply to all EXG channels
    epochs_clean, eye_metadata = find_and_remove_eye_artifacts(epochs_clean)
    metadata.update(eye_metadata)

    # Run second autoreject
    epochs_clean, ar_metadata = apply_autorejection(
        epochs_clean, autoreject.Ransac)
    metadata.update(ar_metadata)

    # Optional check for ICA components (but don't apply)
    # perform_inspection_ica(epochs_clean)

    # (PLOT) visualise cleaned epoch data

    return epochs_clean, metadata


def preprocess_batch(src_path, dest_path, filter_tasks=['od', 'bd'], indices=None):
    """Run preprocessing pipeline on all bdf files in a directory and save
    """

    src_path, dest_path = Path(src_path), Path(dest_path)

    if filter_tasks is not None:
        logger.warning(f"You are only considering tasks in {filter_tasks}!")

    # Retrieve list of files to be processed
    bdf_files = []
    for file in src_path.glob("*.bdf"):
        task = file.name.split('_')[2].split('.')[0].lower()
        if task in filter_tasks:
            bdf_files.append(file)
    logger.info(f"bdf files found: {len(bdf_files)}")

    logger.info("Processing batch...\n")
    all_metadata = {}
    fail_count, success_count = 0, 0
    if indices is not None:
        logger.warning(f"You are limiting to the next {indices} samples!")
        bdf_files = bdf_files[indices[0]:indices[1]]

    for file_path in tqdm(bdf_files, desc="BDF preprocessing"):
        try:
            logger.debug(f"\nProcessing {file_path}...\n")

            tstamp = datetime.now().strftime("%y-%m-%d-%H-%M")
            sid = file_path.name.split('_')[1].rjust(3, '0')
            task = file_path.name.split('_')[2].split('.')[0].upper()

            logger.info(f"\nSample Info:\tSubject: {sid}, Task: {task}")

            clean_epochs, metadata = preprocess_sample(file_path, task=task)
            # clean_epochs, metadata = None, {'test': {'testy': [12, None]}}

            out_path = dest_path / f"task_{task}/{sid}/{tstamp}/"
            metadata['savepath'] = str(out_path)
            metadata['task'] = str(task)
            metadata['subject'] = str(sid)

            logger.info(f"\nWriting clean data and metadata to {out_path}...")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            clean_epochs.save(
                out_path / f'{task}_{sid}_epo.fif', overwrite=True)
            with open(out_path / f'{sid}_{task}_meta.json', 'w') as jsonfile:
                json.dump(metadata, jsonfile)

            all_metadata[str(file_path)] = metadata
            success_count += 1

        except Exception as e:
            logger.error(f"{file_path}: {e}")
            logger.debug(e.args)
            fail_count += 1
            # raise e

    # Combine metadata and save to single, timestamped JSON file
    meta_path = dest_path / 'metadata'
    if not os.path.exists(meta_path):
        os.makedirs(meta_path)
    meta_file_path = meta_path / f'{tstamp}.json'
    with open(meta_file_path, 'w') as jsonfile:
        json.dump(all_metadata, jsonfile)
        logger.info(f"Saved metadata to {meta_file_path}")

    logger.warning(
        f"Done!\nFailed: {fail_count}, Succeeded: {success_count}\n")


if __name__ == '__main__':

    # You can specify via the command line if you want to use a specific bdf file
    if len(sys.argv) >= 3:
        target_file = Path(sys.argv[1])
    else:
        raise ValueError("You need to specify paths (and optional limits)")
    ind = None
    if len(sys.argv) == 5:
        ind = (int(sys.argv[3]),int(sys.argv[4]))
    preprocess_batch(sys.argv[1], sys.argv[2], indices=ind, filter_tasks=['od'])
