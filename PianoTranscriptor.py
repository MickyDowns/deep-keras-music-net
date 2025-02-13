#!/usr/bin/env python
# encoding: utf-8
"""
    PianoTranscriptor (piano) note transcription algorithm.
    """

from __future__ import absolute_import, division, print_function

import argparse

from madmom.processors import IOProcessor, io_arguments
from madmom.audio.signal import SignalProcessor
from madmom.features import ActivationsProcessor
from madmom.features.notes import (RNNPianoNoteProcessor, write_midi,
                                   write_notes, write_mirex_format)
from madmom.features.onsets import PeakPickingProcessor


def main():
    """PianoTranscriptor"""
    
    # define parser
    p = argparse.ArgumentParser(
                                formatter_class=argparse.RawDescriptionHelpFormatter, description='''
        The PianoTranscriptor program detects all notes (onsets) in an audio file
        according to the algorithm described in:
        
        "Polyphonic Piano Note Transcription with Recurrent Neural Networks"
        Sebastian Böck and Markus Schedl.
        Proceedings of the 37th International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2012.
        
        Instead of 'LSTM' units, the current version uses 'tanh' units.
        
        This program can be run in 'single' file mode to process a single audio
        file and write the detected notes to STDOUT or the given output file.
        
        $ PianoTranscriptor single INFILE [-o OUTFILE]
        
        If multiple audio files should be processed, the program can also be run
        in 'batch' mode to save the detected notes to files with the given suffix.
        
        $ PianoTranscriptor batch [-o OUTPUT_DIR] [-s OUTPUT_SUFFIX] LIST OF FILES
        
        If no output directory is given, the program writes the files with the detected notes to same location as the audio files.
        
        The 'pickle' mode can be used to store the used parameters to be able to exactly reproduce experiments.
                                    ''')

    # version
    p.add_argument('--version', action='version',
                   version='PianoTranscriptor.2013')
    # input/output arguments
    io_arguments(p, output_suffix='.notes.txt')
    ActivationsProcessor.add_arguments(p)
    # signal processing arguments
    SignalProcessor.add_arguments(p, norm=False, gain=0, start=True, stop=True)
    # peak picking arguments
    PeakPickingProcessor.add_arguments(p, threshold=0.35, smooth=0.09,
                                       combine=0.05)
    # midi arguments
    # import madmom.utils.midi as midi
    # midi.MIDIFile.add_arguments(p, length=0.6, velocity=100)
    p.add_argument('--midi', dest='output_format', action='store_const',
                   const='midi', help='save as MIDI')
    # mirex stuff
    p.add_argument('--mirex', dest='output_format', action='store_const',
                   const='mirex', help='use the MIREX output format')

    # parse arguments
    args = p.parse_args()
    
    # set immutable defaults
    args.fps = 100
    args.pre_max = 1. / args.fps
    args.post_max = 1. / args.fps
                   
    # set the suffix for midi files
    if args.output_format == 'midi':
        args.output_suffix = '.mid'

    # print arguments
    if args.verbose:
        print(args)
    
        # input processor
    if args.load:
        # load the activations from file
        in_processor = ActivationsProcessor(mode='r', **vars(args))
    else:
        # use a RNN to predict the notes
        in_processor = RNNPianoNoteProcessor()
    
    # output processor
    if args.save:
        # save the RNN note activations to file
        out_processor = ActivationsProcessor(mode='w', **vars(args))
    else:
        # perform peak picking on the activation function
        peak_picking = PeakPickingProcessor(**vars(args))
        # output everything in the right format
        if args.output_format is None:
            output = write_notes
        elif args.output_format == 'midi':
            output = write_midi
        elif args.output_format == 'mirex':
            output = write_mirex_format
        else:
            raise ValueError('unknown output format: %s' % args.output_format)
        out_processor = [peak_picking, output]
    
    # create an IOProcessor
    processor = IOProcessor(in_processor, out_processor)
    
    # and call the processing function
    args.func(processor, **vars(args))


if __name__ == '__main__':
    main()