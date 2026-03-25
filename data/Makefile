all: plots

data: H1_data.txt L1_data.txt V1_data.txt K1_data.txt
plots: plots/detector_data.pdf

H1_data.txt L1_data.txt V1_data.txt K1_data.txt plots/detector_data.pdf: scripts/create_data.py
	uv run scripts/create_data.py


NRSur7dq4_v1.0.h5:
	wget https://git.ligo.org/waveforms/software/lalsuite-waveform-data/-/raw/main/waveform_data/NRSur7dq4_v1.0.h5
