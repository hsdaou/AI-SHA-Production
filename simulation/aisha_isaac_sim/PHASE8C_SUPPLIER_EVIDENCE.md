# Phase 8C supplier documentary evidence gate

Phase 8C resolves the document-level V4.2/manual ambiguity without extending
that result to physical hardware. The procurement and shipping records identify
two `ZLAC8015D V4.2` drivers and two `ZLLG80ASM250-4096 V2.18` motors. A supplier
representative explicitly states that the V4 Series RS485 manual is compatible
with ZLAC8015D V4.2.

Only a sanitized derivative is committed. It contains logical evidence IDs,
dates, hardware identifiers, the supplier's compatibility conclusion and
SHA-256 hashes. It contains no invoice files, email body, contact details or
local source paths.

## Result

The 17/17 documentary contract and strict local hash verification pass for the
procurement record, shipping record and the 25-page V4 Series RS485 manual
version `1.06-20251111`. The passive Phase 8B audit accepts this documentary
chain and now reports exactly two physical blockers:

- `received_driver_label_not_provided`
- `no_stable_usb_rs485_device`

This result does not verify the delivered unit, read a physical driver, validate
the candidate 16384-count encoder scale, authorize wheel motion or release the
robot.

## Revalidate private source files locally

Keep the three source documents outside Git and run:

```bash
simulation/aisha_isaac_sim/tools/validate_phase8c_supplier_evidence.py \
  --procurement-record /absolute/path/to/procurement-record.pdf \
  --shipping-record /absolute/path/to/shipping-record.pdf \
  --rs485-manual /absolute/path/to/rs485-manual.pdf \
  --require-local-hash-verification
```

The next executable phase starts after delivery: photograph the complete V4.2
label and connector context, connect only the USB-RS485 adapter, record its
stable `/dev/serial/by-id/...` identity, and rerun the passive Phase 8B audit.
Only a passing human-reviewed attachment gate may proceed to the separately
guarded, motor-leads-isolated, function-`0x03` telemetry probe.
