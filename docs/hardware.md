# Hardware: 8-channel EMG acquisition rig

Built to capture facial/neck EMG the way Gaddy 2020 did (8 channels, ~1 kHz, monopolar,
shared reference behind the ear, 2 Hz high-pass + 60 Hz notch), so that the encoder in this
repo can eventually run on our own recordings instead of the public dataset.

## Build

- **ADC:** HiLetgo ADS1256 module, 8-channel, 24-bit, 5 V, onboard ADR03 2.5 V reference,
  7.68 MHz crystal. Module ties AINCOM to GND, so channels read single-ended vs ground.
- **MCU:** Arduino UNO, SPI at 2 MHz, direct-port chip-select. Streams a binary frame per
  sweep (0xAA 0x55 sync, counter, 8 × 24-bit samples) at 1 Mbaud.
- **Front end:** LM358 body-bias buffer (~1.67 V) plus anti-alias stage, validated in ngspice.
  Buffered input chosen: ~80 MΩ input impedance. The unbuffered path (150 kΩ / PGA) loaded
  the skin source in simulation.
- **Electrodes:** 3M Red Dot 2560 Ag/AgCl, alligator leads. Single-channel AD620 front end
  was the first build path and remains as a reference design.
- **Host:** Python (pyserial, scipy, pyqtgraph, PySide6): a health-check script (SPI/DRDY
  self-test, per-channel rate, dropped frames), a live 8-trace scope with per-muscle
  activation bars, record/pause/gain/window/notch controls.

## What was measured

- SPI round-trips and DRDY toggling verified; all 8 channels streaming with 0 dropped frames.
- **Rate ceiling.** One ADS1256 is a single muxed ADC. Cycling 8 channels in free-run at
  30 kSPS with a 5-cycle SINC settle discard caps at ~750 Hz/channel in silicon. Measured
  536 Hz/ch at 1 MHz SPI with digitalWrite; **625 Hz/ch** after moving to 2 MHz SPI and
  direct port writes. Reaching ~977 Hz/ch would need a 10 MHz external clock the UNO can't
  supply without a crystal swap.
- **Why 625 Hz is enough.** Nyquist at 312 Hz covers surface EMG, whose energy sits below
  ~150 Hz; Gaddy's own pipeline resamples 1000 → 800 Hz. True 1 kHz simultaneous sampling
  needs 8 independent ADCs (ADS1299 / OpenBCI Cyton class), which is the next hardware step.

## Status

Rig wired, flashed and verified alive. Next: attach electrodes with the body-bias stage and
record real facial EMG; channels rail when inputs float with nothing connected, as expected.

Firmware and host scripts lived in a separate sketches folder; they are being recovered and
will be added under `hardware/` in this repo.
