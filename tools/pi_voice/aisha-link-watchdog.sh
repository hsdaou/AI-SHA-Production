#!/bin/bash
# AI-SHA direct-link watchdog.
#
# On 2026-08-21 the Pi's eth0 transmit queue wedged: the link stayed up and the
# interface kept RECEIVING (30k packets), but its TX counter froze at 104 and it
# stopped answering ARP entirely. The Jetson could no longer reach the speech
# bridge, so the robot went silent with no error anywhere -- every service was
# "active", the speaker was fine, and nothing logged a fault.
#
# The symptom is specific and worth matching precisely: CARRIER PRESENT but the
# peer unreachable. If the carrier is down the cable is simply unplugged or the
# Jetson is off, and bouncing would just thrash the interface, so we do nothing.
#
# Recovery is a link bounce, which is what fixed it by hand.
set -u

IFACE="${AISHA_LINK_IFACE:-eth0}"
PEER="${AISHA_LINK_PEER:-192.168.77.1}"
PINGS="${AISHA_LINK_PINGS:-3}"
MIN_INTERVAL="${AISHA_LINK_MIN_INTERVAL:-300}"   # seconds between bounces
STAMP="/run/aisha-link-watchdog.last"

log() { logger -t aisha-link-watchdog "$*"; echo "$*"; }

carrier="$(cat "/sys/class/net/$IFACE/carrier" 2>/dev/null || echo 0)"
if [ "$carrier" != "1" ]; then
    # No cable / peer powered down. Not our failure mode.
    exit 0
fi

if ping -c "$PINGS" -W 2 -I "$IFACE" "$PEER" >/dev/null 2>&1; then
    exit 0
fi

# Carrier is up but the peer will not answer -- the wedge signature.
now="$(date +%s)"
last="$(cat "$STAMP" 2>/dev/null || echo 0)"
if [ $((now - last)) -lt "$MIN_INTERVAL" ]; then
    log "peer $PEER unreachable on $IFACE, but bounced $((now - last))s ago -- holding off"
    exit 0
fi

tx_before="$(cat "/sys/class/net/$IFACE/statistics/tx_packets" 2>/dev/null || echo ?)"
log "peer $PEER unreachable while carrier is UP on $IFACE (tx_packets=$tx_before) -- bouncing link"
echo "$now" > "$STAMP"

ip link set "$IFACE" down
sleep 3
ip link set "$IFACE" up
sleep 10

if ping -c 2 -W 2 -I "$IFACE" "$PEER" >/dev/null 2>&1; then
    tx_after="$(cat "/sys/class/net/$IFACE/statistics/tx_packets" 2>/dev/null || echo ?)"
    log "recovered: $PEER reachable again (tx_packets $tx_before -> $tx_after)"
else
    log "bounce did NOT recover $PEER -- link may be unplugged or the peer is down"
fi
