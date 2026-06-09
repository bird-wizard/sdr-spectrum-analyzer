#!/usr/bin/env python3
import numpy as np

from PyQt6.QtCore import QSize, Qt, QThread, QTimer
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QGridLayout, QHBoxLayout, QVBoxLayout,
    QSlider, QLabel, QLineEdit, QPushButton, QComboBox,
    QDockWidget, QTextBrowser,
    QTableWidget, QTableWidgetItem, QFileDialog,
)
import pyqtgraph as pg

from Config import *
from Logger import ActivityLogger
from Worker import SDRWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RTL-SDR FM Spectrum Analyzer")
        self.setFixedSize(QSize(1600, 1000))

        # bounds
        self._spec_min = -80.0
        self._spec_max = 0.0

        # peaks tracker
        self._peak_text_items = []

        # sweep toggle
        self._sweep_active = False

        # QGrid layout: column 0 holds plots, column 1 holds controls.
        layout = QGridLayout()
        # stretch plots
        layout.setColumnStretch(0, 3)
        # controls small
        layout.setColumnStretch(1, 1)

        # time plot
        layout.setRowStretch(1, 1)
        # spectrum plot
        layout.setRowStretch(2, 1)
        # waterfall slightly taller
        layout.setRowStretch(3, 2)

        # Worker thread init
        self.sdr_thread = QThread()
        self.sdr_thread.setObjectName("SDR_Thread")
        self.worker = SDRWorker()
        self.worker.moveToThread(self.sdr_thread)

        # Logger thread init
        self.logger_thread = QThread()
        self.logger_thread.setObjectName("Logger_Thread")
        self.logger = ActivityLogger()
        self.logger.moveToThread(self.logger_thread)

        # Build plot and control widgets
        self._build_time_plot(layout)
        self._build_freq_plot(layout)
        self._build_waterfall(layout)
        self._build_freq_controls(layout)
        self._build_gain_controls(layout)
        self._build_sample_rate_combo(layout)
        self._build_detection_controls(layout)
        self._build_sweep_controls(layout)
        self._build_band_reference_panel()
        self._build_activity_panel()

        # widget layout holder
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # connect to update signals
        self.worker.time_plot_update.connect(self._on_time)
        self.worker.freq_plot_update.connect(self._on_freq)
        self.worker.waterfall_plot_update.connect(self._on_waterfall)
        self.worker.peaks_update.connect(self._on_peaks)
        self.worker.sweep_progress.connect(self._on_sweep_progress)
        self.worker.composite_update.connect(self._on_composite_update)
        self.worker.freq_axis_update.connect(self._on_freq_axis_update)

        # Wire peaks to the logger on its own thread.
        self.worker.peaks_update.connect(self.logger.log_peaks)

        self.worker.end_of_run.connect(lambda: QTimer.singleShot(0, self.worker.run))
        self.sdr_thread.started.connect(self.worker.run)
        self.sdr_thread.start()
        self.logger_thread.start()

    def _build_time_plot(self, layout):
        self.time_plot = pg.PlotWidget(labels={"left": "Amplitude", "bottom": "Sample"})
        self.time_plot.setMouseEnabled(x=False, y=True)
        self.time_plot.setYRange(-1.1, 1.1)
        self.time_curve_i = self.time_plot.plot([])
        self.time_curve_q = self.time_plot.plot([])
        layout.addWidget(self.time_plot, 1, 0)

        buttons = QVBoxLayout()
        layout.addLayout(buttons, 1, 1)

        auto_range_button = QPushButton("Auto Range")
        auto_range_button.setMaximumWidth(120)
        auto_range_button.clicked.connect(lambda: self.time_plot.autoRange())
        buttons.addWidget(auto_range_button)

        auto_range_button2 = QPushButton("-1 to +1\n(ADC limits)")
        auto_range_button2.setMaximumWidth(120)
        auto_range_button2.clicked.connect(lambda: self.time_plot.setYRange(-1.1, 1.1))
        buttons.addWidget(auto_range_button2)

    def _build_freq_plot(self, layout):
        self.freq_plot = pg.PlotWidget(labels={"left": "PSD [dB]", "bottom": "Frequency [MHz]"})
        self.freq_plot.setMouseEnabled(x=False, y=True)
        self.freq_plot.setXRange(DEFAULT_CENTER / 1e6 - 0.5, DEFAULT_CENTER / 1e6 + 0.5)
        self.freq_plot.setYRange(-80, 0)
        self.freq_curve = self.freq_plot.plot([])

        # scatter plot for detected peaks live
        # red triangles
        self.peak_scatter = pg.ScatterPlotItem(
            symbol="t", size=10, brush=pg.mkBrush(255, 80, 80, 200)
        )
        self.freq_plot.addItem(self.peak_scatter)

        # blue triangles during sweep
        self.composite_scatter = pg.ScatterPlotItem(
            symbol="t", size=8, brush=pg.mkBrush(80, 160, 255, 180)
        )
        self.freq_plot.addItem(self.composite_scatter)
        
        layout.addWidget(self.freq_plot, 2, 0)

        auto = QPushButton("Auto Range")
        auto.setMaximumWidth(120)
        auto.clicked.connect(lambda: self.freq_plot.autoRange())
        layout.addWidget(auto, 2, 1)

    def _build_waterfall(self, layout):
        wf_layout = QHBoxLayout()
        layout.addLayout(wf_layout, 3, 0)

        self.waterfall = pg.PlotWidget(labels={"left": "Time","bottom": "Frequency [MHz]"})
        self.imageitem = pg.ImageItem(axisOrder="col-major")
        self.waterfall.addItem(self.imageitem)
        self.waterfall.setMouseEnabled(x=False, y=False)
        wf_layout.addWidget(self.waterfall)

        self.colorbar = pg.HistogramLUTWidget()
        self.colorbar.setImageItem(self.imageitem)
        self.colorbar.item.gradient.loadPreset("viridis")
        self.imageitem.setLevels((-70, -20))
        wf_layout.addWidget(self.colorbar)

        auto = QPushButton("Auto Range\n(-2 to +2 sigma)")
        auto.setMaximumWidth(120)
        auto.clicked.connect(self._update_colormap)
        layout.addWidget(auto, 3, 1)

    def _build_freq_controls(self, layout):
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(FREQ_MIN_KHZ, FREQ_MAX_KHZ)
        self.freq_slider.setValue(int(DEFAULT_CENTER / 1e3))
        self.freq_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.freq_slider.setTickInterval(FREQ_TICK_KHZ)

        self.freq_input = QLineEdit()
        self.freq_input.setValidator(
            QDoubleValidator(FREQ_MIN_KHZ / 1e3, FREQ_MAX_KHZ / 1e3, 6)
        )
        self.freq_input.setText(f"{DEFAULT_CENTER / 1e6:.3f}")
        self.freq_input.setFixedWidth(110)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Center [MHz]:"))
        input_row.addWidget(self.freq_input)
        input_row.addStretch()

        # connect signal to worker
        self.freq_slider.valueChanged.connect(self.worker.update_freq)

        # connect text input to worker and sync slider
        def sync_input_from_slider(v):
            if not self.freq_input.hasFocus():
                self.freq_input.setText(f"{v / 1e3:.3f}")
        self.freq_slider.valueChanged.connect(sync_input_from_slider)

        def on_freq_text_committed():
            try:
                mhz = float(self.freq_input.text())
            except ValueError:
                self.freq_input.setText(f"{self.freq_slider.value() / 1e3:.3f}")
                return
            khz = max(FREQ_MIN_KHZ, min(FREQ_MAX_KHZ, int(round(mhz * 1e3))))
            self.freq_slider.setValue(khz)
            self.freq_input.setText(f"{khz / 1e3:.3f}")
        self.freq_input.editingFinished.connect(on_freq_text_committed)

        layout.addWidget(self.freq_slider, 4, 0)
        layout.addLayout(input_row, 4, 1)

    def _build_gain_controls(self, layout):

        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(0, int(max(VALID_GAINS)))
        self.gain_slider.setValue(int(DEFAULT_GAIN_DB))
        self.gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.gain_slider.setTickInterval(5)

        self.gain_input = QLineEdit()
        self.gain_input.setValidator(
            QDoubleValidator(0.0, float(max(VALID_GAINS)), 1)
        )
        self.gain_input.setText(f"{snap_gain(DEFAULT_GAIN_DB):.1f}")
        self.gain_input.setFixedWidth(110)

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("Gain [dB]:"))
        input_row.addWidget(self.gain_input)
        input_row.addStretch()

        self.gain_slider.valueChanged.connect(self.worker.update_gain)

        def sync_input_from_slider(v):
            if not self.gain_input.hasFocus():
                self.gain_input.setText(f"{snap_gain(v):.1f}")
        self.gain_slider.valueChanged.connect(sync_input_from_slider)

        def on_gain_text_committed():
            try:
                db = float(self.gain_input.text())
            except ValueError:
                self.gain_input.setText(f"{snap_gain(self.gain_slider.value()):.1f}")
                return
            db = max(0.0, min(float(max(VALID_GAINS)), db))
            self.gain_slider.setValue(int(round(db)))
            self.gain_input.setText(f"{snap_gain(self.gain_slider.value()):.1f}")
        self.gain_input.editingFinished.connect(on_gain_text_committed)

        layout.addWidget(self.gain_slider, 5, 0)
        layout.addLayout(input_row, 5, 1)

    def _build_sample_rate_combo(self, layout):
        self.sr_combo = QComboBox()
        self.sr_combo.addItems([f"{x} MHz" for x in SAMPLE_RATES_MHZ])
        self.sr_combo.setCurrentIndex(0)
        self.sr_combo.currentIndexChanged.connect(self.worker.update_sample_rate)

        self.sr_label = QLabel()
        def update_label(v):
            self.sr_label.setText(f"Sample Rate: {SAMPLE_RATES_MHZ[v]} MHz")
        self.sr_combo.currentIndexChanged.connect(update_label)
        update_label(self.sr_combo.currentIndex())

        layout.addWidget(self.sr_combo, 6, 0)
        layout.addWidget(self.sr_label, 6, 1)

    def _build_detection_controls(self, layout):
        # Helper that builds one labeled slider row and wires it to a worker slot.
        # Each call produces: slider (col 0) + live-updating label (col 1).
        def make_slider(row, display_name, lo, hi, default, slot, fmt_fn):
            """
            row          : grid row index
            display_name : label prefix shown next to the slider
            lo / hi      : integer range for QSlider
            default      : starting integer value
            slot         : worker slot connected to valueChanged
            fmt_fn       : callable(int) -> display string for the label
            """
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(lo, hi)
            slider.setValue(default)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(max(1, (hi - lo) // 10))

            lbl = QLabel(f"{display_name}: {fmt_fn(default)}")
            lbl.setMinimumWidth(160)

            # Wire to the worker slot AND update the label on every drag.
            slider.valueChanged.connect(slot)
            slider.valueChanged.connect(lambda v: lbl.setText(f"{display_name}: {fmt_fn(v)}"))

            layout.addWidget(slider, row, 0)
            layout.addWidget(lbl,   row, 1)
            return slider

        # Averageing alpha
        make_slider(
            row=7,
            display_name="Avg alpha",
            lo=1, hi=100,
            default=int(DEFAULT_AVG_ALPHA * 100),
            slot=self.worker.update_avg_alpha,
            fmt_fn=lambda v: f"{v / 100.0:.2f}",
        )

        # Gaussian Smooth sigma in FFT bins
        make_slider(
            row=8,
            display_name="Smooth sigma",
            lo=0, hi=40,
            default=int(DEFAULT_SMOOTH_SIGMA * 2),
            slot=self.worker.update_smooth_sigma,
            fmt_fn=lambda v: f"{v / 2.0:.1f} bins",
        )

        # prominence
        make_slider(
            row=9,
            display_name="Prominence",
            lo=1, hi=30,
            default=int(DEFAULT_PEAK_PROMINENCE_DB),
            slot=self.worker.update_peak_prominence,
            fmt_fn=lambda v: f"{v} dB",
        )

        # peak height
        make_slider(
            row=10,
            display_name="Min height",
            lo=-100, hi=-10,
            default=int(DEFAULT_PEAK_MIN_HEIGHT_DB),
            slot=self.worker.update_peak_min_height,
            fmt_fn=lambda v: f"{v} dBm",
        )

        # minimum peak width
        make_slider(
            row=11,
            display_name="Peak BW",
            lo=10, hi=200,
            default=int(DEFAULT_PEAK_BW_KHZ),
            slot=self.worker.update_peak_bw_khz,
            fmt_fn=lambda v: f"{v} kHz",
        )

    def _build_sweep_controls(self, layout):
        # Sweep extent for locking the x-axis when active
        self._sweep_lo_mhz = min(lo for lo, hi in BAND_SWEEP_RANGE)
        self._sweep_hi_mhz = max(hi for lo, hi in BAND_SWEEP_RANGE)

        # Start button
        self.sweep_btn = QPushButton("Start Sweep")
        self.sweep_btn.setCheckable(True)
        self.sweep_btn.setMaximumWidth(160)

        # progress
        self.sweep_progress_lbl = QLabel("Sweep: idle")
        self.sweep_progress_lbl.setMinimumWidth(160)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.sweep_btn)
        btn_row.addWidget(self.sweep_progress_lbl)
        btn_row.addStretch()
        layout.addLayout(btn_row, 12, 0, 1, 2)

        def on_sweep_toggled(checked: bool):
            self._sweep_active = checked
            self.worker.set_sweep_mode(checked)

            if checked:
                self.sweep_btn.setText("Stop Sweep")
                self.freq_slider.setEnabled(False)
                # wipe frequency and expand to sweep range
                self.freq_curve.setData([], [])
                self.freq_plot.setXRange(
                    self._sweep_lo_mhz, self._sweep_hi_mhz, padding=0.02
                )
            else:
                self.sweep_btn.setText("Start Sweep")
                self.freq_slider.setEnabled(True)
                self.sweep_progress_lbl.setText("Sweep: idle")
                self.composite_scatter.setData([], [])
                # Restore live x-range from the current slider value
                center_mhz = self.freq_slider.value() / 1e3
                half_bw    = (self.worker.sample_rate / 2) / 1e6
                self.freq_plot.setXRange(
                    center_mhz - half_bw, center_mhz + half_bw, padding=0
                )

        self.sweep_btn.toggled.connect(on_sweep_toggled)

        dwell_slider = QSlider(Qt.Orientation.Horizontal)
        dwell_slider.setRange(1, 20)
        dwell_slider.setValue(DEFAULT_SWEEP_DWELL)
        dwell_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        dwell_slider.setTickInterval(2)

        dwell_lbl = QLabel(f"Sweep dwell: {DEFAULT_SWEEP_DWELL} frames")
        dwell_lbl.setMinimumWidth(160)

        dwell_slider.valueChanged.connect(self.worker.update_sweep_dwell)
        dwell_slider.valueChanged.connect(
            lambda v: dwell_lbl.setText(f"Sweep dwell: {v} frames")
        )

        layout.addWidget(dwell_slider, 13, 0)
        layout.addWidget(dwell_lbl,   13, 1)

    def _build_band_reference_panel(self):
        dock = QDockWidget("RF Band Reference", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.LeftDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        browser = QTextBrowser()
        browser.setMinimumWidth(270)

        with open("Tutorial.html", "r", encoding="utf-8") as f:
            browser.setHtml(f.read())

        dock.setWidget(browser)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _on_time(self, samples):
        self.time_curve_i.setData(samples.real)
        self.time_curve_q.setData(samples.imag)

    def _on_freq(self, freqs_mhz, psd):
        if not self._sweep_active:
            self.freq_curve.setData(freqs_mhz, psd)
            self.freq_plot.setXRange(freqs_mhz[0], freqs_mhz[-1], padding=0)

        # Waterfall x-axis always follows the current center regardless of mode.
        self.waterfall.setXRange(freqs_mhz[0], freqs_mhz[-1], padding=0)
        self.imageitem.setRect(freqs_mhz[0], 0,
                               freqs_mhz[-1] - freqs_mhz[0], NUM_ROWS)

    def _on_sweep_progress(self, step: int, total: int):
        pct = int(100 * step / total) if total else 0
        self.sweep_progress_lbl.setText(f"Sweep: step {step}/{total}  ({pct}%)")

    def _on_composite_update(self, peaks: list):
        if not peaks:
            return
        xs = [p[0] for p in peaks]
        ys = [p[1] for p in peaks]
        self.composite_scatter.setData(xs, ys)
    
    def _on_freq_axis_update(self, lo_mhz: float, hi_mhz: float):
            axis = self.waterfall.getPlotItem().getAxis('bottom')
            axis.setRange(lo_mhz, hi_mhz)
        

    def _on_waterfall(self, spectrogram):
        self.imageitem.setImage(spectrogram, autoLevels=False)
        sigma = float(np.std(spectrogram))
        mean  = float(np.mean(spectrogram))
        self._spec_min = mean - 2 * sigma
        self._spec_max = mean + 2 * sigma

    def _on_peaks(self, peaks):
        self.peak_scatter.setData([p[0] for p in peaks], [p[1] for p in peaks])

        for item in self._peak_text_items:
            self.freq_plot.removeItem(item)
        self._peak_text_items.clear()

        for fm, pm, station in peaks:
            if station is None:
                continue
            t = pg.TextItem(f"{station:.1f}", color=(255, 80, 80), anchor=(0.5, 1.0))
            t.setPos(fm, pm + 2)
            self.freq_plot.addItem(t)
            self._peak_text_items.append(t)

    def _update_colormap(self):
        self.imageitem.setLevels((self._spec_min, self._spec_max))
        self.colorbar.setLevels(self._spec_min, self._spec_max)

    def _build_activity_panel(self):
        dock = QDockWidget("RF Activity Log", self)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.LeftDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        container = QWidget()
        vbox = QVBoxLayout(container)

        # Per-band summary table
        self.activity_table = QTableWidget(0, 5)
        self.activity_table.setHorizontalHeaderLabels(
            ["Frequency (MHz)", "Band", "Detections", "Last Seen", "Peak Power"]
        )
        self.activity_table.horizontalHeader().setStretchLastSection(True)
        self.activity_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.activity_table.setAlternatingRowColors(True)
        self.activity_table.setMinimumWidth(320)
        self.activity_table.setSortingEnabled(True)
        self.activity_table.horizontalHeader().setSectionsClickable(True)
        vbox.addWidget(self.activity_table)

        # Running total across all bands
        self.session_label = QLabel("Session total: 0 detections logged")
        vbox.addWidget(self.session_label)

        # Action buttons
        btn_row = QHBoxLayout()
        export_btn = QPushButton("Export CSV")
        clear_btn  = QPushButton("Clear History")
        btn_row.addWidget(export_btn)
        btn_row.addWidget(clear_btn)
        vbox.addLayout(btn_row)

        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        # stats_ready comes back from the logger thread; _on_stats_ready
        # runs on the GUI thread (queued connection) and updates the table.
        self.logger.stats_ready.connect(self._on_stats_ready)

        # Refresh the table every 5 seconds. The timer fires on the GUI thread
        self._stats_timer = QTimer()
        self._stats_timer.setInterval(5_000)
        self._stats_timer.timeout.connect(self.logger.fetch_band_stats)
        self._stats_timer.start()

        # First fetch fires 2 s after startup so the table isn't blank.
        QTimer.singleShot(2_000, self.logger.fetch_band_stats)

        def on_export():
            path, _ = QFileDialog.getSaveFileName(
                self, "Export Activity CSV", "sdr_activity.csv", "CSV Files (*.csv)"
            )
            if path:
                self.logger.export_csv(path)

        def on_clear():
            self.logger.clear_history()
            # Fetch immediately so the table empties rather than showing stale counts.
            self.logger.fetch_band_stats()

        export_btn.clicked.connect(on_export)
        clear_btn.clicked.connect(on_clear)

    def _on_stats_ready(self, rows: list):
        class NumericItem(QTableWidgetItem):
            def __init__(self, display: str, value: float):
                super().__init__(display)
                self._value = value
            def __lt__(self, other):
                if isinstance(other, NumericItem):
                    return self._value < other._value
                return super().__lt__(other)

        self.activity_table.setSortingEnabled(False)
        self.activity_table.setRowCount(len(rows))

        total = 0
        for r, (freq, band, count, last_seen, peak_power) in enumerate(rows):
            total += count
            self.activity_table.setItem(r, 0, NumericItem(f"{freq:.1f}", float(freq)))
            self.activity_table.setItem(r, 1, QTableWidgetItem(band or "Unknown"))
            self.activity_table.setItem(r, 2, NumericItem(f"{count:,}", float(count)))
            self.activity_table.setItem(r, 3, QTableWidgetItem(last_seen or "—"))
            self.activity_table.setItem(r, 4, NumericItem(f"{peak_power:.1f} dBm", float(peak_power)))

        self.activity_table.setSortingEnabled(True)
        self.activity_table.resizeColumnsToContents()
        self.session_label.setText(f"Session total: {total:,} detections logged")

    def closeEvent(self, event):
        self._stats_timer.stop()
        self.worker.stop()
        self.sdr_thread.quit()
        self.sdr_thread.wait(2000)
        self.logger_thread.quit()
        self.logger_thread.wait(2000)
        try:
            sdr.close()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
    

if __name__ == "__main__":
    main()
