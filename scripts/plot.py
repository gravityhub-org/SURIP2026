import pylab as plt
import scienceplots
plt.style.use(['science','ieee','bright'])

def plot_detector_data(data, waveforms, epoch, duration):
    fig = plt.figure(figsize=(12, 8))
    for idx, (name, d) in enumerate(data.items(), 1):
        plt.subplot(4, 1, idx)
        plt.plot(data[name].sample_times, data[name].data, label=name)
        plt.plot(waveforms[name].sample_times, waveforms[name].data)
        plt.xlim(epoch, epoch + duration)
        plt.ylabel('Strain')
        plt.title(name)
        if idx == 4:
            plt.xlabel('Time (s)')
    plt.tight_layout()
    return fig


