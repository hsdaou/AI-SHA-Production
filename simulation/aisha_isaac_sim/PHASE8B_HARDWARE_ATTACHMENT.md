# Phase 8B hardware attachment gate

This is the passive identity gate before the already prepared RS485 read-only
probe. It does not open a serial port, send a Modbus frame, energize a motor or
authorize wheel motion.

## Current result

The workstation has no `/dev/serial/by-id` directory and `lsusb` shows no
USB-RS485 adapter. The supplied outer archive is named V4.0, but Phase 8C now
binds the procurement and shipping evidence for `ZLAC8015D V4.2` to the exact
hash of the V4 Series RS485 manual. The supplier explicitly attests that this
manual applies to V4.2. Documentary compatibility therefore passes even though
no manual file is named V4.2. The archive still has no photo of the actual
received driver, so the physical telemetry gate remains blocked on the label
and USB adapter identities.

The software serial path was exercised separately against a Linux pseudo-
terminal. The adapter issued exactly three Modbus function `0x03` requests for
position/speed, status and fault registers and decoded the returned signed
values. No function `0x06` request was emitted.

## Evidence to capture at the robot

Provide one sharp photo of the driver label and one context photo of the
connector side. The label photo must show the complete model/hardware revision,
serial number and input-voltage rating. Also photograph both sides of the
USB-RS485 adapter so its chipset/model and serial identity can be recorded.

The applicable RS485 manual is already hash-registered through Phase 8C. Do not
substitute another revision without adding a new supplier attestation and hash
verification.

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
  --expected-usb-serial SERIAL_FROM_UDEV
```

A passing attachment audit authorizes only human review before the guarded
function-`0x03` telemetry probe. It does not authorize register writes or motor
power. Motor leads must remain physically isolated during that later probe.
