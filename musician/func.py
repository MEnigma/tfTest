"""
PROJECT : tfTest
AUTHER : MARK
DATE : 2019/11/21
IDE : Pycharm
"""
import os
import sys
import subprocess
import midi2audio


def convertMidiToMp3(inputPath:str,outputPath:str):
    """
    转换 midi to MP3
    :return:
    """
    input_file = inputPath#"output.midi"
    output_file = outputPath#"output.mp3"
    assert os.path.exists(input_file)

    print(" Converting %s to Mp3 " % input_file)
    midi2audio.FluidSynth().midi_to_audio(input_file,output_file)


if __name__ == "__main__":
    convertMidiToMp3("src/dong_fang_cui_meng_xiang.mid",'output/dong_fang_cui_meng_xiang.wav')