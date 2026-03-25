all: NRSur7dq4_v1.0.h5

plots/detector_data.pdf: create_data.py
	uv run create_data.py

NRSur7dq4_v1.0.h5:
	wget https://git.ligo.org/waveforms/software/lalsuite-waveform-data/-/raw/main/waveform_data/NRSur7dq4_v1.0.h5
