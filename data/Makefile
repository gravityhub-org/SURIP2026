all: plots data

data: example_H1_data.txt example_L1_data.txt example_V1_data.txt example_K1_data.txt
plots: plots/detector_data.pdf

example_H1_data.txt example_L1_data.txt example_V1_data.txt example_K1_data.txt plots/detector_data.pdf: scripts/create_data.py
	uv run scripts/create_data.py


NRSur7dq4_v1.0.h5:
	wget https://git.ligo.org/waveforms/software/lalsuite-waveform-data/-/raw/main/waveform_data/NRSur7dq4_v1.0.h5
