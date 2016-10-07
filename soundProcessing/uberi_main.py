# imports
## for speech
import uberi_init as sr
import contextlib
import numpy as np
import matplotlib as plt

## for sockets
from flask_socketio import SocketIO

## for voice/music classifier
import wave
from essentia.standard import *
from numpy import inf
from types import *

# globals
baseDir = "~/Projects/music_study_aid"

# initialize sockets
socketio = SocketIO(message_queue='redis://')
try:
    socketio.emit('my_response', {'data': 'Speech Recognition started', 'who': 'AI'}, namespace='/chat')
except RuntimeError:
    print("speech_recognition: __main__.py socketio.emit error")

# command processing
def command_interpreter(command):
    
    if "end session" in command:
        nextAction = "end"
        actionText = "Thanks for playing! Ending session."
    elif ("go back" in command) or ("repeat" in command) or ("replay" in command):
        notes = [int(s) for s in command.split() if s.isdigit()]
        nextAction = "continue"
        actionText = "Going back " + str(notes[0]) + " notes"
    else:
        nextAction = "continue"
        actionText = "<some command response>"
    
    return(nextAction, actionText)

# voice / music classification
## cfa classifier

# to do:
#1. figure out how to pass a stream rather than a file address
#2. second classifier type
#3. logistic model combining two
#4. figure out how to make this persist as a stream process i.e., not instantiating loader each time.
#5. code below is a 1st draft. too many nested "for" loops. needs vector optimization.

def cfa_classifier(inAudio, inType='file', numPeaks=5, showGraphics=False):
    
    '''The following two approaches for separating voice commands from music are based on the observation that
        speech signals usually display patterns of harmonics influenced by the shape of the vocal tract. Within a
        time frame, they manifest themselves as peaks within the spectrum. Further, the partials can be found at the
        fundamental frequency of a tone and also near its integer multiples. Finally, the harmonics are sustained over
        a certain span of time in which they are likely to vary in frequency. This last characteristic is highly
        discriminative vis a vis both noise and music. Stated simply, voice spectrograms exhibit characteristic curved
        trajectories. Music spectrograms exhibit strictly horizontal and minor vertical structures and noise looks
        like... noise. Noticed as early as 1993 by M. Hawley @ MIT.
        
        Both approaches have three basic steps.First, they slice the audio stream into small frames and extract
        features for each frame. Second, they train a classifier on a distinct training set. The classifier learns
        to distinguish two classes (i.e., music/no-music and voice/no-voice). Third, the classifier is used to predict
        class labels for all the frames in the test set. Finally, classification results are smoothed in a post-
        processing step to obtain a label sequence for continuous audio segments.
        
        In their 2007 work, Seyerlehner et al recognized that music can be differentiated by structural properties
        like harmony and rhythm. If clarity and consistency of paritial frequency emissions are evidence of music,
        then a feature could be developed to reliably detect continuous frequency activations... even in the presence
        of other audio signals. That feature was Continuous Frequency Activation (CFA). The computation of CFA can be
        subdivided into:
        1. Conversion of the input audio stream into 11 kHz mono.'''
    
    if inType == 'wavDir':
        loader = essentia.standard.MonoLoader(downmix='mix', filename=inAudio, sampleRate=22050)
        inAudio = loader()
        #inAudio = inAudio[1*44100:180*44100]
    
    '''
        2. Computation of the power spectrum using a Hanning window function and a window size of 1024 samples
        (roughly) 100ms of audio. A hop-size of 256 samples is used, resulting in an overlap of 75% percent.
        After the conversion to decibel, we obtain a standard spectrogram representation.'''
    
    w = essentia.standard.Windowing(type = 'hann') #'hamming'?
    power = essentia.standard.PowerSpectrum()
    spectrum = essentia.standard.Spectrum()
    
    pwrSpec=[]
    for frame in FrameGenerator(inAudio, frameSize = 1024, hopSize = 256):
        pwrSpec.append(spectrum(w(frame))) # NOTE: USING SPECTRUM. power returns: power spectrum of input
    
    pwrSpec = essentia.array(pwrSpec).T #transpose, then convert list to an essentia.array first (== numpy.array of floats)
    
    if showGraphics == True:
        plt.figure(figsize=(12,8))
        plt.subplot(2,3,1); plt.plot(inAudio)
        plt.subplot(2,3,2); plt.imshow(pwrSpec[:100,:], aspect = 'auto')

    pwrSpec = np.square(pwrSpec)
    pwrSpec = 10*np.log10(pwrSpec)
    pwrSpec[pwrSpec == -inf] = 0

    # NOTE: CBA DOES A NORMALIZATION STEP AFTER HANNING. GIVEN LOCAL NORMALIZATION BELOW, IT SEEMS REDUNDANT
    '''
        3. Emphasize local peaks within each frame of the STFT by subtracting from the power spectrum of each frame the running average using a window size of N = 21 frequency bins: x_emph = x_i - 1/N * Sigma_k=-N/2... Xmin(max(k,1),N Were Xi denotes the energy of the i-th frequency component of the current frame. This step is useful to emphasize very soft tones, belonging to background music. The perceivable horizontal bars in the spectogram are compositions of consecutive local maxima. Thus, we try to emphasize these soft bars by emphasizing all local maxima in the spectrum of a frame.'''

    wndw = 21
    for i in range(pwrSpec.shape[1]):
        if i < int(wndw/2): pwrSpec[:,i] = pwrSpec[:,0] # set left side to initial value
        elif i > (pwrSpec.shape[1] - int(wndw/2)): pwrSpec[:,i] = pwrSpec[:,-1] # right side to initial value
        else: pwrSpec[:,i] = pwrSpec[:,i] - np.mean(pwrSpec[:,(i-int(wndw/2)):(i+int(wndw/2))]) # de-mean center

    '''
        4. Binarize the frequency component to eliminate strength of activation (energy) in a given frame j,X_emph_ij by comparing to a fixed binarization threshold of 0.1 keeps even soft activations in the spectogram. But, inactive frequency bins are set to 0 using this low threshold.'''

    binThresh = 0.1

    pwrSpec[pwrSpec >= binThresh] = 1
    pwrSpec[pwrSpec < binThresh] = 0

    if showGraphics == True: plt.subplot(2,3,4); plt.imshow(pwrSpec[:100,:], aspect = 'auto')

    '''
        5. Compute frequency activtaion. Process the binarized power spectrum in terms of blocks. Each block consists of F = 100 frames and blocks overlap by 50%, which means that a block is an excerpt of the binarized spectrogram corresponding to 2.6 seconds of audio. For each block we compute the frequency activation function Activation(i). For each frequency bin i, the frequency activation function measures how often a frequency component is active in a block. We obtain the frequency activation function for a block by simply summing up the binarized values for each frequency bin i: Activation(i) = 1/F * Sigma_j=1^F Bij'''

    timeBlockSize = 30
    timeIncr = timeBlockSize #/2
    numTimeBlocks = (int(pwrSpec.shape[1] / timeIncr)) #-1
    pwrAct = np.zeros((pwrSpec.shape[0],numTimeBlocks))

    for freqRow in range(pwrSpec.shape[0]):
        timePosCtr = 0
        for timeCol in range(numTimeBlocks):
            pwrAct[freqRow,timeCol] = np.sum(pwrSpec[freqRow,timePosCtr:(timePosCtr+timeBlockSize)])
            timePosCtr = timePosCtr + timeIncr

    if showGraphics == True:
        plt.subplot(2,3,5); plt.imshow(pwrAct[0:100,:], aspect = 'auto')
        freqSum = np.sum(pwrAct,1)
        plt.subplot(2,3,6); plt.plot(freqSum[0:100])

    '''6. Detect strong peaks. Peaks in the frequency activation function of a given block indicate steady activations of narrow frequency bands. The spikier the activation function, the more likely horizontal bars, which are characteristic of sustained musical tones, are present. Even one large peak is quite a good indicator for the presence of a tone. The peakiness of the freuquency activation function is consequently a good indicator for the presence of music. To extract the peaks we use the following simple peak picking algorithm. a. Collect all local peaks, starting from the lowest frequency. Each local maximum of the activation function is a potential peak. b. For each peak x_p, compute its height-to-width index or peak value pv(xp) = h(xp)/w(xp), where the height h(xp) is defined as min[f(p) - f(xl),f(p) - f(xr)], with f (x) the value of the activation function at point (frequency bin) x and xl and xr are closest local minima of f to the left and right of xp, respec- tively. The width w(xp) of the peak is given by: w(x_p) = p - x_l, f(p) - f(xl) < f(p) - f(xr) ELSE x_r - p otherwise'''

    outPeaks = []
    
    for freqRow in range(pwrAct.shape[0]):
        
        tmpRow = pwrAct[freqRow,:]
        N = len(tmpRow)
        peaks = np.zeros((N))
        maxList = np.zeros((N))
        minList = np.zeros((N))
        
        for i in range(N):
            maxList[i] = -1
            minList[i] = -1
        lastMaxIndex = 0; lastMinIndex = 0
        direction = 0
        cf = 0
        
        # advance if steady initial value
        while ((cf < (N-1)) and (tmpRow[cf] == tmpRow[cf + 1])):
            cf = cf + 1
        
        # in some cases that takes you to end of input row. So, check.
        if(cf < (N-1)):
            # if start is a max or min, account for it
            if tmpRow[cf] > tmpRow[cf+1]:
                maxList[cf] = tmpRow[cf]
                lastMaxIndex = cf
                direction = 0
            
            elif tmpRow[cf] < tmpRow[cf+1]:
                minList[cf] = tmpRow[cf]
                lastMinIndex = cf
                direction = 1
        
        else:
            continue
        
        count = 1
        
        for i in range(1,N):
            
            if ((tmpRow[i] > tmpRow[i-1]) and (direction == 0)): # minimum detected
                
                # scan backards for earliest occurrence of that minimum
                cb = i-1
                while ((cb > 1) and (tmpRow[cb-1] == tmpRow[cb])):
                    cb = cb - 1
                
                # save minimum to list
                minList[cb] = tmpRow[cb] # save first minimuim
                minList[i-1] = tmpRow[i-1] # save second minimum
                count = count + 1
                direction = 1
                
                # calculate area of the peak
                if count < 3: # first value was a max
                    peaks[lastMaxIndex] = (maxList[lastMaxIndex] - minList[cb]) / (cb - lastMaxIndex)
                    # (hmax - hmin) / w
                
                else:
                    if minList[lastMinIndex] > minList[cb]: # hminL > hminR
                        peaks[lastMaxIndex] = (maxList[lastMaxIndex] - minList[lastMinIndex])/(lastMaxIndex - lastMinIndex)
                        # (hmax - hminL) / w
                    else:
                        peaks[lastMaxIndex] = (maxList[lastMaxIndex] - minList[cb])/(cb-lastMaxIndex)
                        # (hmsx - hminR) / w
        
                lastMinIndex = i - 1
                                
            elif ((tmpRow[i] < tmpRow[i-1]) and (direction == 1)):
                                    
                # save maximum to list
                maxList[i-1] = tmpRow[i-1]
                count = count + 1
                direction = 0
                lastMaxIndex = i - 1
                    
        '''
            7. Quantify the CFA of the activation function of a block, the pv values of all detected peaks are
            sorted in descending order, and the sum of the five largest peak values is taken to characterize the
            overall peakiness of the activation function.'''
                                                    
        # sort peaks
        for i in range(len(peaks)):
            outPeaks.append(peaks[i])

    outPeaks = sorted(outPeaks,reverse=True)
    outPeaks = outPeaks[0:numPeaks]
    #print(outPeaks)
    print(sum(outPeaks))
    if sourceType == "file":
        if sum(outPeaks) > 85:
            return("music")
        else:
            return("voice")
    elif sourceType == "mic":
        if sum(outPeaks) > 130:
            return("music")
        else:
            return("voice")
    else: print("error: invalid source type")
    
    '''Thus we obtain one numeric value for each block of frames, which quantifies the presence of steady frequency
        components within the current audio segment.'''


# initialize microphone, speech recognition
sourceType = "file"
fname = "stringTest.wav"
#fname = '../../data/maps/AkPnBcht/ISOL/CH/MAPS_ISOL_CH0.1_M_AkPnBcht.wav' #'../../data/wip/tmp/stringTest.wav'
signOff = False
clipDuration = 8 # Int (seconds) to segment file. None to listen to entire file.
offset = 0

r = sr.Recognizer()

# process user session
try:
    if sourceType == "mic":
        m = sr.Microphone()
        socketio.emit('my_response', {'data': 'Calibrating background noise...',
                      'who': 'AI'}, namespace='/chat')
        with m as source: r.adjust_for_ambient_noise(source)
        socketio.emit('my_response', {'data': 'Setting min energy to {}'.format(r.energy_threshold),
                      'who': 'AI'}, namespace='/chat')

    while signOff == False:
        socketio.emit('my_response', {'data': 'Listening...', 'who': 'AI'}, namespace='/chat')
        
        if sourceType == "mic":
            with m as source: audio = r.listen(source)
        
        elif sourceType == "file":
            AUDIO_FILE = sr.AudioFile(fname)
            
            if clipDuration == None: # you will listen to entire file
                
                signOff = True
            
                with AUDIO_FILE as source:
                    audio = r.record(source)
        
            else:
                with AUDIO_FILE as source:
                
                    if (source.DURATION - offset) <= clipDuration:
                        clipDuration = (source.DURATION - offset)
                        signOff = True

                    audio = r.record(source, duration=clipDuration, offset=offset)
                            
                offset = offset + clipDuration
        
        # write audio to a WAV file
        #wavCoord = baseDir + '/userData/user1/tmp.wav'
        wavCoord = 'tmp.wav'
        with open(wavCoord, "wb") as f:
            f.write(audio.get_wav_data())

        socketio.emit('my_response', {'data': 'Determining sound type...', 'who': 'AI'}, namespace='/chat')
        socketio.emit('my_response', {'data': '  ', 'who': 'AI'}, namespace='/chat')

        if cfa_classifier(inAudio=wavCoord, inType='wavDir', numPeaks=5, showGraphics=True) == "voice":
            try:
                #socketio.emit('my_response', {'data': 'Interpreting voice command...',
                #              'who': 'AI'}, namespace='/chat')
                              
                # recognize speech using Google Speech Recognition
                value = r.recognize_google(audio)
                
                socketio.emit('my_response', {'data': '{}'.format(value).encode("utf-8"), 'who': 'You'}, namespace='/chat')
                
                nextAction, actionText = command_interpreter(value)
                
                if nextAction == "end":
                    signOff = True
                else:
                    socketio.emit('my_response', {'data': '{}'.format(actionText).encode("utf-8"),
                                  'who': 'AI'}, namespace='/chat')
        
            except sr.UnknownValueError:
                socketio.emit('my_response', {'data': 'Say again?', 'who': 'AI'}, namespace='/chat')

            except sr.RequestError as e:
                socketio.emit('my_response', {'data': 'Uh oh! No results from Google service: {0}'.format(e), 'who': 'AI'}, namespace='/chat')
                    
        else:
            socketio.emit('my_response', {'data': 'See played below.', 'who': 'AI'}, namespace='/chat')
            # save wav
            # call recognition

    socketio.emit('my_response', {'data': 'Thanks for playing! Ending now.', 'who': 'AI'}, namespace='/chat')

except KeyboardInterrupt:
    pass

