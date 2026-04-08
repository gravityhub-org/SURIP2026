all: plots data

data: data/example_H1_data.txt data/example_L1_data.txt data/example_V1_data.txt data/example_K1_data.txt
plots: plots/detector_data.pdf

data/example_H1_data.txt data/example_L1_data.txt data/example_V1_data.txt data/example_K1_data.txt plots/detector_data.pdf: create_data.py
	uv run create_data.py

NRSur7dq4_v1.0.h5:
	wget https://git.ligo.org/waveforms/software/lalsuite-waveform-data/-/raw/main/waveform_data/NRSur7dq4_v1.0.h5
