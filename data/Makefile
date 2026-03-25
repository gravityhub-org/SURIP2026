all: plots

plots: plots/detector_data.pdf

plots/detector_data.pdf: scripts/create_data.py
	uv run scripts/create_data.py

NRSur7dq4_v1.0.h5:
	wget https://git.ligo.org/waveforms/software/lalsuite-waveform-data/-/raw/main/waveform_data/NRSur7dq4_v1.0.h5
