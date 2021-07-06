
from pydub import AudioSegment
from pydub.playback import play


def playsound():
    song = AudioSegment.from_wav('E:\downloads\origin\yolov5-5.0\song.wav')
    play(song)



