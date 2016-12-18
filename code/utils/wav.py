import sys
import wave
import base64
import struct
import StringIO
import numpy as np

from scipy.io import wavfile
from scipy.signal import resample

from IPython.core.display import HTML
from IPython.core.display import display


def load_wav(filename, rate=16000):
    """
    Loading wav file
    :param filename: filename with full path
    :param rate: The sampling frequency (i.e. frame rate) of the data.
    :return: Sample rate and resampled audio
    """
    # load file
    rate, data = wavfile.read(filename)

    # convert stereo to mono
    if len(data.shape) > 1:
        data = data[:,0]/2 + data[:,1]/2

    # re-interpolate samplerate    
    ratio = float(rate) / float(rate)
    data = resample(data, len(data) * ratio)
    
    return rate, data.astype(np.int16)


def write_wav24(filename, data, rate):
    """
    Create a 24 bit wav file.
    :param filename: Name and full path
    :param data: Audio data
    :param rate: The sampling frequency (i.e. frame rate) of the data.
    """
    a32 = np.asarray(data, dtype=np.int32)
    if a32.ndim == 1:
        # Convert to a 2D array with a single column.
        a32.shape = a32.shape + (1,)
    # By shifting first 0 bits, then 8, then 16, the resulting output
    # is 24 bit little-endian.
    a8 = (a32.reshape(a32.shape + (1,)) >> np.array([0, 8, 16])) & 255
    wavdata = a8.astype(np.uint8).tostring()

    w = wave.open(filename, 'wb')
    w.setnchannels(a32.shape[1])
    w.setsampwidth(3)
    w.setframerate(rate)
    w.writeframes(wavdata)
    w.close()


def wavPlayer(data, rate):
    """
    Display html 5 player for compatible browser
    The browser need to know how to play wav through html5.
    there is no autoplay to prevent file playing when the browser opens
    Adapted from SciPy.io. and
    github.com/Carreau/posts/blob/master/07-the-sound-of-hydrogen.ipynb
    :param data: Audio data
    :param rate: The sampling frequency (i.e. frame rate) of the data.
    :return:
    """
    
    buffer_ = StringIO.StringIO()
    buffer_.write(b'RIFF')
    buffer_.write(b'\x00\x00\x00\x00')
    buffer_.write(b'WAVE')

    buffer_.write(b'fmt ')
    if data.ndim == 1:
        noc = 1
    else:
        noc = data.shape[1]
    bits = data.dtype.itemsize * 8
    sbytes = rate*(bits // 8)*noc
    ba = noc * (bits // 8)
    buffer_.write(struct.pack('<ihHIIHH', 16, 1, noc, rate, sbytes, ba, bits))

    # data chunk
    buffer_.write(b'data')
    buffer_.write(struct.pack('<i', data.nbytes))

    if data.dtype.byteorder == '>' or (data.dtype.byteorder == '=' and sys.byteorder == 'big'):
        data = data.byteswap()

    buffer_.write(data.tostring())
    # return buffer_.getvalue()
    # Determine file size and place it in correct
    # position at start of the file.
    size = buffer_.tell()
    buffer_.seek(4)
    buffer_.write(struct.pack('<i', size-8))
    
    val = buffer_.getvalue()
    
    src = """
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Simple Test</title>
    </head>
    
    <body>
    <audio controls="controls" style="width:600px" >
      <source controls src="data:audio/wav;base64,{base64}" type="audio/wav" />
      Your browser does not support the audio element.
    </audio>
    </body>
    """.format(base64=base64.encodestring(val))
    display(HTML(src))
