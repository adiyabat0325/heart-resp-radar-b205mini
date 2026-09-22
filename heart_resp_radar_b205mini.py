#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Heart Rate and Respiration CW Doppler Radar (B205mini)
# Author: Adiyabat
# Description: Contactless HR/RR vital-sign sensing using a USRP B205mini as
#              a CW Doppler radar.
#
# NOTE ON PROVENANCE
# -------------------
# This file is a hand-transcribed equivalent of heart_resp_radar_b205mini.grc,
# produced by reading the .grc flowgraph (blocks, parameters and connections)
# and reproducing the code GNU Radio Companion's code generator (grcc) would
# emit for it. It was NOT run through `grcc` itself, because a full GNU Radio
# + UHD install (~22 GB with its dependency tree) was not something the
# environment this was written in could take on. Every block, parameter and
# connection below is transcribed 1:1 from the .grc; the parts most likely to
# differ cosmetically from a real `grcc` run are the exact Qt grid layout
# positions of the widgets (the .grc leaves gui_hint blank for every sink, so
# GRC auto-places them -- the placement below is a reasonable explicit grid,
# not a byte-for-byte reproduction of GRC's auto-layout algorithm).
#
# Before relying on this for a live demo, regenerate it for real:
#   1. Open heart_resp_radar_b205mini.grc in GNU Radio Companion, or
#   2. Run:  grcc heart_resp_radar_b205mini.grc
# and diff the result against this file.
#
# GNU Radio version: 3.10 (qt_gui)

from gnuradio import analog
from gnuradio import blocks
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from PyQt5 import QtCore
from gnuradio import qtgui
from gnuradio import uhd
import time
from gnuradio.qtgui import Range, RangeWidget
import numpy as np
import sip

from gnuradio import eng_notation
from gnuradio.eng_arg import eng_float, intx
from argparse import ArgumentParser


# =============================================================================
# Embedded Python Blocks
#
# These correspond to the three `epy_block` blocks in the .grc
# (clutter_cancel, unwrap, rate_est_resp / rate_est_hr). GRC normally writes
# each epy_block's `_source_code` out to its own file and imports it; they are
# collected here as ordinary top-level classes for readability.
# =============================================================================

class clutter_canceller(gr.sync_block):
    """
    Adaptive static-clutter (TX->RX leakage) canceller.

    A monostatic CW radar's received signal is dominated by a large, nearly
    static complex vector: the direct TX->RX leakage plus reflections off
    everything in the room that isn't moving. The tiny phase modulation
    caused by a chest moving a few millimeters rides on top of this as a
    small perturbation. If the static vector is much larger than the motion
    term, atan2() becomes very sensitive to ordinary receiver noise (the
    "chaotic phase, no matter what" symptom), because the noise now competes
    with a weak wanted signal instead of the strong static one.

    dc_blocker_xx already removes the pure-0 Hz component, but any leakage
    that drifts slowly (temperature, cable flex, DC offset creep) is not
    exactly 0 Hz and survives a fixed-length DC blocker. This block tracks
    that drift with a slow exponential moving average (EMA) and subtracts
    it every sample:

        avg[n] = avg[n-1] + alpha * (x[n] - avg[n-1])
        y[n]   = x[n] - avg[n-1]

    This is a one-pole high-pass filter with cutoff f_c ~= alpha * samp_rate
    / (2*pi). At this block's sample rate (rate2, ~20 Sps) the default
    alpha=0.01 gives f_c ~= 0.03 Hz, well below resp_low (0.1 Hz) -- so real
    breathing/heartbeat modulation passes through essentially unaffected,
    while slow static/near-static clutter is tracked and cancelled.

    Tuning: if respiration amplitude still looks suppressed, lower alpha
    (slower tracking, e.g. 0.005). If leakage drift still isn't fully
    removed, raise alpha (faster tracking, e.g. 0.02) -- but too high will
    start eating into the respiration band itself.
    """

    def __init__(self, alpha=0.01):
        gr.sync_block.__init__(
            self,
            name='Clutter Canceller (slow EMA subtract)',
            in_sig=[np.complex64],
            out_sig=[np.complex64],
        )
        self.alpha = alpha
        self.avg = None

    def work(self, input_items, output_items):
        x = input_items[0]
        n = len(x)
        if self.avg is None:
            self.avg = x[0]
        avg = self.avg
        alpha = self.alpha
        out = np.empty(n, dtype=np.complex64)
        for i in range(n):
            out[i] = x[i] - avg
            avg = avg + alpha * (x[i] - avg)
        self.avg = avg
        output_items[0][:] = out
        return n


class phase_unwrap(gr.sync_block):
    """
    Continuous phase unwrap.

    np.unwrap() only unwraps within a single call; this block keeps state
    across GNU Radio work() calls so the phase stays continuous across the
    whole stream (needed since the radar runs indefinitely).
    """

    def __init__(self):
        gr.sync_block.__init__(
            self,
            name='Phase Unwrap (stateful)',
            in_sig=[np.float32],
            out_sig=[np.float32],
        )
        self.last_raw = None
        self.last_unwrapped = 0.0

    def work(self, input_items, output_items):
        x = input_items[0].astype(np.float64)
        n = len(x)
        if self.last_raw is None:
            self.last_raw = x[0]
        prepended = np.concatenate(([self.last_raw], x))
        diffs = np.diff(prepended)
        wrapped = (diffs + np.pi) % (2 * np.pi) - np.pi
        out = self.last_unwrapped + np.cumsum(wrapped)
        self.last_raw = x[-1]
        self.last_unwrapped = out[-1]
        output_items[0][:] = out.astype(np.float32)
        return n


class rate_estimator(gr.sync_block):
    """
    Peak-frequency rate estimator with a two-part signal-validity gate.

    Reports the strongest FFT peak inside [f_low, f_high] Hz (in events per
    minute) from a sliding window of the band-passed phase signal -- but
    ONLY when that peak looks like a real, sustained periodic signal, not
    colored noise left over from integrating (unwrapping) receiver/leakage
    noise. Two gates must both pass:

    1. Amplitude gate: windowed signal's std must exceed `min_amplitude`.
    2. In-band SNR gate + frequency-stability gate: the peak must stand
       out `snr_ratio`x above the median of the OTHER bins in the same
       band (a locally-fair noise-floor estimate, robust to the 1/f^2
       coloring that comes from phase-unwrap integrating noise -- a global
       full-spectrum median is not fair, since it compares against much
       quieter high-frequency bins). AND that peak's frequency must stay
       within `stability_tol` (relative) of itself across the last
       `stability_count` consecutive accepted windows -- a real
       breathing/heartbeat rate holds steady for several seconds in a row;
       a colored-noise peak wanders window to window.

    When any gate fails, the block reports 0.0 (not the last good value)
    and clears its stability history, so a live BPM readout of 0 clearly
    means "no valid, sustained rate detected."

    Tuning: watch this block's readout with no one in front of the radar.
    If it isn't reliably 0, raise `snr_ratio` and/or `stability_count`
    (fewer false positives, slower to lock on) or raise `min_amplitude` if
    the empty-room signal amplitude is high. If it's reliably 0 but real
    breathing/heartbeat also reads 0, lower these -- there's a real
    sensitivity/false-negative tradeoff; better TX/RX antenna isolation
    (reducing the underlying leakage-driven noise) helps both sides of it.
    """

    def __init__(self, samp_rate=20.0, f_low=0.1, f_high=0.5,
                 window_sec=15.0, update_sec=1.0,
                 min_amplitude=0.05, snr_ratio=8.0,
                 stability_count=5, stability_tol=0.15):
        gr.sync_block.__init__(
            self,
            name='Rate Estimator (BPM)',
            in_sig=[np.float32],
            out_sig=[np.float32],
        )
        self.samp_rate = samp_rate
        self.f_low = f_low
        self.f_high = f_high
        self.window_len = max(int(window_sec * samp_rate), 8)
        self.update_len = max(int(update_sec * samp_rate), 1)
        self.min_amplitude = min_amplitude
        self.snr_ratio = snr_ratio
        self.stability_count = max(int(stability_count), 1)
        self.stability_tol = stability_tol
        self.buf = np.zeros(self.window_len, dtype=np.float64)
        self.filled = 0
        self.since_update = 0
        self.bpm = 0.0
        self.recent_freqs = []

    def work(self, input_items, output_items):
        x = input_items[0]
        n = len(x)
        for i in range(n):
            self.buf[:-1] = self.buf[1:]
            self.buf[-1] = x[i]
            if self.filled < self.window_len:
                self.filled += 1
            self.since_update += 1
            if (self.since_update >= self.update_len
                    and self.filled >= self.window_len // 2):
                self._update_bpm()
                self.since_update = 0
            output_items[0][i] = self.bpm
        return n

    def _update_bpm(self):
        seg = self.buf[-self.filled:]
        seg = seg - np.mean(seg)

        # Gate 1: is anything actually moving?
        if np.std(seg) < self.min_amplitude:
            self.recent_freqs = []
            self.bpm = 0.0
            return

        win = np.hanning(len(seg))
        spec = np.abs(np.fft.rfft(seg * win))
        freqs = np.fft.rfftfreq(len(seg), d=1.0 / self.samp_rate)
        mask = (freqs >= self.f_low) & (freqs <= self.f_high)
        if not np.any(mask):
            self.recent_freqs = []
            self.bpm = 0.0
            return

        band_mag = spec[mask]
        band_freqs = freqs[mask]
        peak_idx = np.argmax(band_mag)
        peak_mag = band_mag[peak_idx]

        # Gate 2: local (in-band) SNR -- fair even if the overall
        # spectrum is colored (e.g. 1/f^2 from phase-unwrap noise).
        other_bins = np.delete(band_mag, peak_idx)
        noise_floor = (np.median(other_bins) if len(other_bins) else 0.0) + 1e-12
        if peak_mag < self.snr_ratio * noise_floor:
            self.recent_freqs = []
            self.bpm = 0.0
            return

        peak_freq = band_freqs[peak_idx]

        # Gate 3: frequency stability across consecutive windows -- a
        # real periodic signal holds its frequency; a wandering
        # colored-noise peak does not.
        self.recent_freqs.append(peak_freq)
        if len(self.recent_freqs) > self.stability_count:
            self.recent_freqs.pop(0)
        if len(self.recent_freqs) < self.stability_count:
            self.bpm = 0.0
            return
        lo, hi = min(self.recent_freqs), max(self.recent_freqs)
        center = float(np.mean(self.recent_freqs))
        if (hi - lo) > self.stability_tol * center:
            self.bpm = 0.0
            return

        self.bpm = center * 60.0


# =============================================================================
# Main flowgraph
# =============================================================================

class heart_resp_radar_b205mini(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "Heart Rate and Respiration CW Doppler Radar (B205mini)", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("Heart Rate and Respiration CW Doppler Radar (B205mini)")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}")
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "heart_resp_radar_b205mini")
        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 1e6
        self.decim1 = decim1 = 1000
        self.decim2 = decim2 = 50
        self.rate1 = rate1 = samp_rate / decim1
        self.rate2 = rate2 = rate1 / decim2
        self.hr_low = hr_low = 0.8
        self.hr_high = hr_high = 3.3
        self.resp_low = resp_low = 0.1
        self.resp_high = resp_high = 0.5
        self.snr_ratio = snr_ratio = 8.0
        self.min_amplitude = min_amplitude = 0.05

        ##################################################
        # Blocks
        ##################################################

        # ---- gated-parameter sliders ----
        self._min_amplitude_range = Range(0.0, 1.0, 0.005, 0.05, 200)
        self._min_amplitude_win = RangeWidget(self._min_amplitude_range, self.set_min_amplitude, "Min Amplitude (rad, gate)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._min_amplitude_win, 0, 0, 1, 2)

        self._snr_ratio_range = Range(1.0, 20.0, 0.5, 8.0, 200)
        self._snr_ratio_win = RangeWidget(self._snr_ratio_range, self.set_snr_ratio, "SNR Ratio (gate)", "counter_slider", float, QtCore.Qt.Horizontal)
        self.top_grid_layout.addWidget(self._snr_ratio_win, 0, 2, 1, 2)

        # ---- USRP B205mini: RX (CW radar receive path) ----
        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join(('', "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0, 1)),
            ),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_time_unknown_pps(uhd.time_spec(0))
        self.uhd_usrp_source_0.set_center_freq(5.53e9, 0)
        self.uhd_usrp_source_0.set_antenna("RX2", 0)
        self.uhd_usrp_source_0.set_gain(50, 0)

        # ---- USRP B205mini: TX (CW carrier / illuminator) ----
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join(('', "")),
            uhd.stream_args(
                cpu_format="fc32",
                args='',
                channels=list(range(0, 1)),
            ),
            '',
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_time_unknown_pps(uhd.time_spec(0))
        self.uhd_usrp_sink_0.set_center_freq(5.53e9, 0)
        self.uhd_usrp_sink_0.set_antenna("TX/RX", 0)
        self.uhd_usrp_sink_0.set_gain(40, 0)

        # ---- CW illuminator: constant complex tone fed straight to the TX sink ----
        self.analog_const_source_x_0 = analog.sig_source_c(0, analog.GR_CONST_WAVE, 0, 0, 1, 0)

        # ---- RX conditioning chain ----
        self.dc_block = filter.dc_blocker_cc(128, True)

        self.lpf1 = filter.fir_filter_ccf(
            decim1,
            firdes.low_pass(1, samp_rate, 5000, 2000, window.WIN_HAMMING, 6.76))

        self.low_pass_filter_0 = filter.fir_filter_ccf(
            1,
            firdes.low_pass(1, rate1, 5, 2, window.WIN_HAMMING, 6.76))

        self.clutter_cancel = clutter_canceller(alpha=0.03)

        self.phase = blocks.complex_to_arg(1)

        self.unwrap = phase_unwrap()

        # ---- rate-band filters ----
        self.bpf_resp = filter.fir_filter_fff(
            1,
            firdes.band_pass(1, rate2, resp_low, resp_high, 0.05, window.WIN_HAMMING, 6.76))

        self.bpf_hr = filter.fir_filter_fff(
            1,
            firdes.band_pass(1, rate2, hr_low, hr_high, 0.3, window.WIN_HAMMING, 6.76))

        # ---- rate estimators ----
        self.rate_est_resp = rate_estimator(
            samp_rate=rate2, f_low=resp_low, f_high=resp_high,
            window_sec=20.0, update_sec=1.0,
            min_amplitude=min_amplitude, snr_ratio=snr_ratio,
            stability_count=5, stability_tol=0.15)

        self.rate_est_hr = rate_estimator(
            samp_rate=rate2, f_low=hr_low, f_high=hr_high,
            window_sec=10.0, update_sec=1.0,
            min_amplitude=min_amplitude, snr_ratio=snr_ratio,
            stability_count=5, stability_tol=0.15)

        # ---- live BPM readouts ----
        self.num_resp_0 = qtgui.number_sink(
            gr.sizeof_float, 0, qtgui.NUM_GRAPH_HORIZ, 1, None)
        self.num_resp_0.set_update_time(0.10)
        self.num_resp_0.set_title("Respiration rate (breaths/min)")
        self.num_resp_0.set_min(0)
        self.num_resp_0.set_max(40)
        self.num_resp_0.set_color(0, "black", "black")
        self.num_resp_0.enable_autoscale(True)
        num_resp_0_win = sip.wrapinstance(self.num_resp_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(num_resp_0_win, 1, 2, 1, 2)

        self.num_hr_0 = qtgui.number_sink(
            gr.sizeof_float, 0, qtgui.NUM_GRAPH_HORIZ, 1, None)
        self.num_hr_0.set_update_time(0.10)
        self.num_hr_0.set_title("Heart rate (beats/min)")
        self.num_hr_0.set_min(0)
        self.num_hr_0.set_max(220)
        self.num_hr_0.set_color(0, "black", "black")
        self.num_hr_0.enable_autoscale(True)
        num_hr_0_win = sip.wrapinstance(self.num_hr_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(num_hr_0_win, 1, 0, 1, 2)

        # ---- scopes / spectra (diagnostic + presentation display) ----
        self.qtgui_sink_x_0 = qtgui.sink_c(
            1024, window.WIN_BLACKMAN_hARRIS, 0, 1e3,
            "", True, True, True, True)
        self.qtgui_sink_x_0.set_update_time(1.0 / 10)
        self.qtgui_sink_x_0.enable_rf_freq(False)
        qtgui_sink_x_0_win = sip.wrapinstance(self.qtgui_sink_x_0.qwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(qtgui_sink_x_0_win, 2, 0, 1, 4)

        self.qtgui_time_sink_x_0 = qtgui.time_sink_f(1024, rate2, '', 1, None)
        self.qtgui_time_sink_x_0.set_update_time(0.10)
        self.qtgui_time_sink_x_0.set_y_axis(-1, 1)
        self.qtgui_time_sink_x_0.set_y_label('Amplitude', "")
        self.qtgui_time_sink_x_0.enable_tags(True)
        self.qtgui_time_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_0.enable_autoscale(True)
        self.qtgui_time_sink_x_0.enable_grid(False)
        self.qtgui_time_sink_x_0.enable_legend(True)
        self.qtgui_time_sink_x_0.set_line_label(0, "Phase (rad)")
        qtgui_time_sink_x_0_win = sip.wrapinstance(self.qtgui_time_sink_x_0.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(qtgui_time_sink_x_0_win, 3, 0, 1, 2)

        self.qtgui_time_sink_x_1 = qtgui.time_sink_f(1024, 20, '', 1, None)
        self.qtgui_time_sink_x_1.set_update_time(0.10)
        self.qtgui_time_sink_x_1.set_y_axis(-1, 1)
        self.qtgui_time_sink_x_1.set_y_label('Amplitude', "")
        self.qtgui_time_sink_x_1.enable_tags(True)
        self.qtgui_time_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_1.enable_autoscale(True)
        self.qtgui_time_sink_x_1.enable_grid(False)
        self.qtgui_time_sink_x_1.enable_legend(True)
        self.qtgui_time_sink_x_1.set_line_label(0, "Unwrapped Phase")
        qtgui_time_sink_x_1_win = sip.wrapinstance(self.qtgui_time_sink_x_1.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(qtgui_time_sink_x_1_win, 3, 2, 1, 2)

        self.qtgui_time_sink_x_2 = qtgui.time_sink_f(1024, 20, '', 1, None)
        self.qtgui_time_sink_x_2.set_update_time(0.10)
        self.qtgui_time_sink_x_2.set_y_axis(-1, 1)
        self.qtgui_time_sink_x_2.set_y_label('Amplitude', "")
        self.qtgui_time_sink_x_2.enable_tags(True)
        self.qtgui_time_sink_x_2.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_2.enable_autoscale(True)
        self.qtgui_time_sink_x_2.enable_grid(False)
        self.qtgui_time_sink_x_2.enable_legend(True)
        self.qtgui_time_sink_x_2.set_line_label(0, "Respiration Waveform")
        qtgui_time_sink_x_2_win = sip.wrapinstance(self.qtgui_time_sink_x_2.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(qtgui_time_sink_x_2_win, 4, 0, 1, 2)

        self.qtgui_time_sink_x_3 = qtgui.time_sink_f(1024, 20, '', 1, None)
        self.qtgui_time_sink_x_3.set_update_time(0.10)
        self.qtgui_time_sink_x_3.set_y_axis(-1, 1)
        self.qtgui_time_sink_x_3.set_y_label('Amplitude', "")
        self.qtgui_time_sink_x_3.enable_tags(True)
        self.qtgui_time_sink_x_3.set_trigger_mode(qtgui.TRIG_MODE_FREE, qtgui.TRIG_SLOPE_POS, 0.0, 0, 0, "")
        self.qtgui_time_sink_x_3.enable_autoscale(True)
        self.qtgui_time_sink_x_3.enable_grid(False)
        self.qtgui_time_sink_x_3.enable_legend(True)
        self.qtgui_time_sink_x_3.set_line_label(0, "Heart Rate Waveform")
        qtgui_time_sink_x_3_win = sip.wrapinstance(self.qtgui_time_sink_x_3.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(qtgui_time_sink_x_3_win, 4, 2, 1, 2)

        self.qtgui_freq_sink_x_0 = qtgui.freq_sink_c(
            1024, window.WIN_BLACKMAN_hARRIS, 0, 20, "", 1, None)
        self.qtgui_freq_sink_x_0.set_update_time(0.10)
        self.qtgui_freq_sink_x_0.set_y_axis(-140, 10)
        self.qtgui_freq_sink_x_0.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_0.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_0.enable_autoscale(False)
        self.qtgui_freq_sink_x_0.enable_grid(False)
        self.qtgui_freq_sink_x_0.set_fft_average(1.0)
        self.qtgui_freq_sink_x_0.enable_axis_labels(True)
        self.qtgui_freq_sink_x_0.enable_control_panel(False)
        self.qtgui_freq_sink_x_0.set_fft_window_normalized(False)
        qtgui_freq_sink_x_0_win = sip.wrapinstance(self.qtgui_freq_sink_x_0.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(qtgui_freq_sink_x_0_win, 5, 0, 1, 2)

        self.qtgui_freq_sink_x_1 = qtgui.freq_sink_c(
            1024, window.WIN_BLACKMAN_hARRIS, 0, 20, "", 1, None)
        self.qtgui_freq_sink_x_1.set_update_time(0.10)
        self.qtgui_freq_sink_x_1.set_y_axis(-140, 10)
        self.qtgui_freq_sink_x_1.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_1.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_1.enable_autoscale(False)
        self.qtgui_freq_sink_x_1.enable_grid(False)
        self.qtgui_freq_sink_x_1.set_fft_average(1.0)
        self.qtgui_freq_sink_x_1.enable_axis_labels(True)
        self.qtgui_freq_sink_x_1.enable_control_panel(False)
        self.qtgui_freq_sink_x_1.set_fft_window_normalized(False)
        qtgui_freq_sink_x_1_win = sip.wrapinstance(self.qtgui_freq_sink_x_1.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(qtgui_freq_sink_x_1_win, 5, 2, 1, 2)

        self.qtgui_freq_sink_x_2 = qtgui.freq_sink_f(
            1024, window.WIN_BLACKMAN_hARRIS, 0, 20, "Resp Freq", 1, None)
        self.qtgui_freq_sink_x_2.set_update_time(0.10)
        self.qtgui_freq_sink_x_2.set_y_axis(-140, 10)
        self.qtgui_freq_sink_x_2.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_2.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_2.enable_autoscale(False)
        self.qtgui_freq_sink_x_2.enable_grid(False)
        self.qtgui_freq_sink_x_2.set_fft_average(1.0)
        self.qtgui_freq_sink_x_2.enable_axis_labels(True)
        self.qtgui_freq_sink_x_2.enable_control_panel(False)
        self.qtgui_freq_sink_x_2.set_fft_window_normalized(False)
        qtgui_freq_sink_x_2_win = sip.wrapinstance(self.qtgui_freq_sink_x_2.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(qtgui_freq_sink_x_2_win, 6, 0, 1, 2)

        self.qtgui_freq_sink_x_3 = qtgui.freq_sink_f(
            1024, window.WIN_BLACKMAN_hARRIS, 0, 20, "Heart rate in freq", 1, None)
        self.qtgui_freq_sink_x_3.set_update_time(0.10)
        self.qtgui_freq_sink_x_3.set_y_axis(-140, 10)
        self.qtgui_freq_sink_x_3.set_y_label('Relative Gain', 'dB')
        self.qtgui_freq_sink_x_3.set_trigger_mode(qtgui.TRIG_MODE_FREE, 0.0, 0, "")
        self.qtgui_freq_sink_x_3.enable_autoscale(False)
        self.qtgui_freq_sink_x_3.enable_grid(False)
        self.qtgui_freq_sink_x_3.set_fft_average(1.0)
        self.qtgui_freq_sink_x_3.enable_axis_labels(True)
        self.qtgui_freq_sink_x_3.enable_control_panel(False)
        self.qtgui_freq_sink_x_3.set_fft_window_normalized(False)
        qtgui_freq_sink_x_3_win = sip.wrapinstance(self.qtgui_freq_sink_x_3.pyqwidget(), Qt.QWidget)
        self.top_grid_layout.addWidget(qtgui_freq_sink_x_3_win, 6, 2, 1, 2)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_const_source_x_0, 0), (self.uhd_usrp_sink_0, 0))
        self.connect((self.bpf_hr, 0), (self.qtgui_freq_sink_x_3, 0))
        self.connect((self.bpf_hr, 0), (self.qtgui_time_sink_x_3, 0))
        self.connect((self.bpf_hr, 0), (self.rate_est_hr, 0))
        self.connect((self.bpf_resp, 0), (self.qtgui_freq_sink_x_2, 0))
        self.connect((self.bpf_resp, 0), (self.qtgui_time_sink_x_2, 0))
        self.connect((self.bpf_resp, 0), (self.rate_est_resp, 0))
        self.connect((self.clutter_cancel, 0), (self.phase, 0))
        self.connect((self.clutter_cancel, 0), (self.qtgui_freq_sink_x_1, 0))
        self.connect((self.dc_block, 0), (self.lpf1, 0))
        self.connect((self.low_pass_filter_0, 0), (self.clutter_cancel, 0))
        self.connect((self.low_pass_filter_0, 0), (self.qtgui_freq_sink_x_0, 0))
        self.connect((self.lpf1, 0), (self.low_pass_filter_0, 0))
        self.connect((self.lpf1, 0), (self.qtgui_sink_x_0, 0))
        self.connect((self.phase, 0), (self.qtgui_time_sink_x_0, 0))
        self.connect((self.phase, 0), (self.unwrap, 0))
        self.connect((self.rate_est_hr, 0), (self.num_hr_0, 0))
        self.connect((self.rate_est_resp, 0), (self.num_resp_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.dc_block, 0))
        self.connect((self.unwrap, 0), (self.bpf_hr, 0))
        self.connect((self.unwrap, 0), (self.bpf_resp, 0))
        self.connect((self.unwrap, 0), (self.qtgui_time_sink_x_1, 0))

    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "heart_resp_radar_b205mini")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()
        event.accept()

    # ---------------------------------------------------------------
    # Variable getters / setters (mirrors what GRC generates for each
    # variable block so downstream dependents stay consistent if you
    # change a value at runtime, e.g. from a script or the GUI).
    # ---------------------------------------------------------------

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_rate1(self.samp_rate / self.decim1)
        self.lpf1.set_taps(firdes.low_pass(1, self.samp_rate, 5000, 2000, window.WIN_HAMMING, 6.76))
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_decim1(self):
        return self.decim1

    def set_decim1(self, decim1):
        self.decim1 = decim1
        self.set_rate1(self.samp_rate / self.decim1)

    def get_decim2(self):
        return self.decim2

    def set_decim2(self, decim2):
        self.decim2 = decim2
        self.set_rate2(self.rate1 / self.decim2)

    def get_rate1(self):
        return self.rate1

    def set_rate1(self, rate1):
        self.rate1 = rate1
        self.set_rate2(self.rate1 / self.decim2)
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.rate1, 5, 2, window.WIN_HAMMING, 6.76))

    def get_rate2(self):
        return self.rate2

    def set_rate2(self, rate2):
        self.rate2 = rate2
        self.bpf_hr.set_taps(firdes.band_pass(1, self.rate2, self.hr_low, self.hr_high, 0.3, window.WIN_HAMMING, 6.76))
        self.bpf_resp.set_taps(firdes.band_pass(1, self.rate2, self.resp_low, self.resp_high, 0.05, window.WIN_HAMMING, 6.76))
        self.qtgui_time_sink_x_0.set_samp_rate(self.rate2)

    def get_hr_low(self):
        return self.hr_low

    def set_hr_low(self, hr_low):
        self.hr_low = hr_low
        self.bpf_hr.set_taps(firdes.band_pass(1, self.rate2, self.hr_low, self.hr_high, 0.3, window.WIN_HAMMING, 6.76))

    def get_hr_high(self):
        return self.hr_high

    def set_hr_high(self, hr_high):
        self.hr_high = hr_high
        self.bpf_hr.set_taps(firdes.band_pass(1, self.rate2, self.hr_low, self.hr_high, 0.3, window.WIN_HAMMING, 6.76))

    def get_resp_low(self):
        return self.resp_low

    def set_resp_low(self, resp_low):
        self.resp_low = resp_low
        self.bpf_resp.set_taps(firdes.band_pass(1, self.rate2, self.resp_low, self.resp_high, 0.05, window.WIN_HAMMING, 6.76))

    def get_resp_high(self):
        return self.resp_high

    def set_resp_high(self, resp_high):
        self.resp_high = resp_high
        self.bpf_resp.set_taps(firdes.band_pass(1, self.rate2, self.resp_low, self.resp_high, 0.05, window.WIN_HAMMING, 6.76))

    def get_snr_ratio(self):
        return self.snr_ratio

    def set_snr_ratio(self, snr_ratio):
        self.snr_ratio = snr_ratio
        self.rate_est_hr.snr_ratio = self.snr_ratio
        self.rate_est_resp.snr_ratio = self.snr_ratio

    def get_min_amplitude(self):
        return self.min_amplitude

    def set_min_amplitude(self, min_amplitude):
        self.min_amplitude = min_amplitude
        self.rate_est_hr.min_amplitude = self.min_amplitude
        self.rate_est_resp.min_amplitude = self.min_amplitude


def main(top_block_cls=heart_resp_radar_b205mini, options=None):
    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()
        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()


if __name__ == '__main__':
    main()
