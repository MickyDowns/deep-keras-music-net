#!/usr/bin/env python3

"""JACK client that prints all received MIDI events."""

import jack
import binascii

monitorClient = jack.Client("MIDI-Monitor")
monitorPort = monitorClient.midi_inports.register("input")


@client.set_process_callback
def process(frames):
    for offset, data in monitorPort.incoming_midi_events():
        print("{0}: 0x{1}".format(monitorClient.last_frame_time + offset,
                                  binascii.hexlify(data).decode()))

with client:
    print("#" * 80)
    print("press Return to quit")
    print("#" * 80)
    input()