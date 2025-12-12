from pylsl import StreamInlet, resolve_byprop


def main():
    streams = resolve_byprop("name", "EEG_DATA")

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    while True:
        sample, timestamp = inlet.pull_sample()
        print("got %s at time %s" % (sample[0], timestamp))


if __name__ == "__main__":
    main()