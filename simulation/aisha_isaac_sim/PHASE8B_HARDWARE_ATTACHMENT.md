# Phase 8B hardware attachment gate

This is the passive identity gate before the already prepared RS485 read-only
probe. It does not open a serial port, send a Modbus frame, energize a motor or
authorize wheel motion.

## Current result

The workstation has no `/dev/serial/by-id` directory and `lsusb` shows no
USB-RS485 adapter. The supplied outer archive contains
`ZLAC8015D V4.0.zip`. The passive audit also reads that nested ZIP's directory
without extracting it: the English RS485 document is `ZLAC8015D V4 Series
RS485 Communication Version 1.06-20251111.pdf`, and no entry is named V4.2.
The archive has no photo of the received driver. Therefore the physical
telemetry gate remains blocked.

The software serial path was exercised separately against a Linux pseudo-
terminal. The adapter issued exactly three Modbus function `0x03` requests for
position/speed, status and fault registers and decoded the returned signed
values. No function `0x06` request was emitted.

## Evidence to capture at the robot

Provide one sharp photo of the driver label and one context photo of the
connector side. The label photo must show the complete model/hardware revision,
serial number and input-voltage rating. Also photograph both sides of the
USB-RS485 adapter so its chipset/model and serial identity can be recorded.

Obtain the exact RS485 communication manual for the received revision from the
supplier. A V4 Series or V4.0 document is not automatically treated as a V4.2
manual even if its register table looks similar.

Connect only the USB-RS485 adapter to the workstation first, then capture:

```bash
ls -l /dev/serial/by-id
udevadm info --query=property --name=/dev/ttyUSBX
```

Do not use `/dev/ttyUSB0` as the permanent configuration; its index may change.
Use the stable `/dev/serial/by-id/...` path and record the adapter serial.

## Re-run the passive audit

```bash
simulation/aisha_isaac_sim/tools/audit_phase8b_hardware_attachment.py \
  --supplier-archive /absolute/path/to/supplier.zip \
  --driver-label-photo /absolute/path/to/driver-label.jpg \
  --confirmed-driver-label "ZLAC8015D V4.2" \
  --matching-rs485-manual /absolute/path/to/exact-manual.pdf \
  --confirm-manual-matches-label \
  --expected-usb-serial SERIAL_FROM_UDEV
```

A passing attachment audit authorizes only human review before the guarded
function-`0x03` telemetry probe. It does not authorize register writes or motor
power. Motor leads must remain physically isolated during that later probe.
