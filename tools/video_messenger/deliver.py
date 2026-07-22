#!/usr/bin/env python3
"""AI-SHA skill: GATED delivery of queued video messages to the outside world.

The robot operates fully locally; this is the ONE deliberate gate to the
internet, and it is CLOSED by default. It only sends when:
  1. the config file exists AND has  "enabled": true    (the gate)
  2. a transport is configured with the admin's own credentials
  3. the network is actually reachable

Transports:
  whatsapp  — official WhatsApp Business Cloud API (Meta). Requires the admin
              to create a Meta developer app and supply: access token,
              phone_number_id, and the destination number. We do NOT use
              unofficial WhatsApp libraries (ToS violation, number-ban risk).
  email     — SMTP fallback (e.g. Gmail + app password).

Credentials live ONLY in ~/video_messages/config.json (0600, git-ignored,
outside the repo). Nothing is ever committed or logged.

Usage:
  deliver.py init            write a disabled template config
  deliver.py status          show gate state + queue
  deliver.py send            deliver everything in the outbox (if gate open)
  deliver.py send --dry-run  show what WOULD be sent, send nothing
"""
import argparse, json, os, shutil, socket, sys, time

HOME      = os.path.expanduser("~")
DATA_HOME = os.environ.get("VIDEO_MSG_HOME", f"{HOME}/video_messages")
OUTBOX    = f"{DATA_HOME}/outbox"
SENT      = f"{DATA_HOME}/sent"
CONFIG    = f"{DATA_HOME}/config.json"

TEMPLATE = {
    "enabled": False,          # <— THE GATE. Set true only when delivery is wanted.
    "transport": "whatsapp",   # "whatsapp" or "email"
    "whatsapp": {
        "access_token": "",     # Meta developer app token (admin supplies)
        "phone_number_id": "",  # from the WhatsApp Cloud API dashboard
        "to": ""                # destination, international format e.g. 9715XXXXXXXX
    },
    "email": {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "user": "",             # sending account
        "app_password": "",     # app-specific password, NOT the real password
        "to": ""                # recipient address
    },
    "max_mb": 15                # refuse to send anything bigger
}


def _load():
    if not os.path.exists(CONFIG):
        return None
    return json.load(open(CONFIG))


def _online(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except OSError:
        return False


def cmd_init(_a):
    os.makedirs(DATA_HOME, exist_ok=True)
    if os.path.exists(CONFIG):
        print(f"[init] config already exists: {CONFIG} (not overwriting)"); return
    with open(CONFIG, "w") as f:
        json.dump(TEMPLATE, f, indent=2)
    os.chmod(CONFIG, 0o600)
    print(f"[init] wrote DISABLED template: {CONFIG}")
    print("[init] fill in credentials and set \"enabled\": true to open the gate.")


def _queue():
    if not os.path.isdir(OUTBOX):
        return []
    return sorted(f for f in os.listdir(OUTBOX) if f.endswith(".mp4"))


def cmd_status(_a):
    cfg = _load()
    gate = "NO CONFIG (closed)" if cfg is None else ("OPEN" if cfg.get("enabled") else "CLOSED")
    print(f"[status] delivery gate : {gate}")
    if cfg:
        print(f"[status] transport     : {cfg.get('transport')}")
    print(f"[status] network       : {'online' if _online() else 'offline'}")
    q = _queue()
    print(f"[status] outbox        : {len(q)} message(s)")
    for f in q:
        print(f"           {f}")


# ── transports ───────────────────────────────────────────────────────────────
def _send_whatsapp(cfg, path):
    """Official WhatsApp Business Cloud API: upload media, then send as video."""
    import requests
    wa = cfg["whatsapp"]
    tok, pnid, to = wa["access_token"], wa["phone_number_id"], wa["to"]
    if not (tok and pnid and to):
        return False, "whatsapp credentials incomplete in config.json"
    base = f"https://graph.facebook.com/v21.0/{pnid}"
    hdr = {"Authorization": f"Bearer {tok}"}
    with open(path, "rb") as f:
        r = requests.post(f"{base}/media", headers=hdr,
                          data={"messaging_product": "whatsapp", "type": "video/mp4"},
                          files={"file": (os.path.basename(path), f, "video/mp4")},
                          timeout=120)
    if r.status_code != 200:
        return False, f"media upload failed: {r.status_code} {r.text[:200]}"
    media_id = r.json().get("id")
    body = {"messaging_product": "whatsapp", "to": to, "type": "video",
            "video": {"id": media_id,
                      "caption": "Video message from the AI-SHA robot (administration)"}}
    r = requests.post(f"{base}/messages", headers={**hdr, "Content-Type": "application/json"},
                      json=body, timeout=60)
    if r.status_code != 200:
        return False, f"send failed: {r.status_code} {r.text[:200]}"
    return True, r.json().get("messages", [{}])[0].get("id", "sent")


def _send_email(cfg, path):
    import smtplib
    from email.message import EmailMessage
    em = cfg["email"]
    if not (em["user"] and em["app_password"] and em["to"]):
        return False, "email credentials incomplete in config.json"
    msg = EmailMessage()
    msg["Subject"] = f"AI-SHA video message: {os.path.basename(path)}"
    msg["From"] = em["user"]; msg["To"] = em["to"]
    msg.set_content("A visitor recorded this video message on the AI-SHA robot "
                    "in the administration.")
    with open(path, "rb") as f:
        msg.add_attachment(f.read(), maintype="video", subtype="mp4",
                           filename=os.path.basename(path))
    with smtplib.SMTP(em["smtp_host"], em["smtp_port"], timeout=60) as s:
        s.starttls(); s.login(em["user"], em["app_password"]); s.send_message(msg)
    return True, f"emailed to {em['to']}"


def cmd_send(a):
    cfg = _load()
    if cfg is None:
        print("[send] gate CLOSED: no config. Run 'deliver.py init' first."); sys.exit(1)
    if not cfg.get("enabled"):
        print("[send] gate CLOSED (\"enabled\": false). The robot stays local."); sys.exit(1)
    q = _queue()
    if not q:
        print("[send] outbox empty"); return
    if not _online():
        print("[send] network offline — leaving messages queued."); sys.exit(1)
    os.makedirs(SENT, exist_ok=True)
    transport = cfg.get("transport", "whatsapp")
    fn = _send_whatsapp if transport == "whatsapp" else _send_email
    for f in q:
        path = os.path.join(OUTBOX, f)
        mb = os.path.getsize(path) / 1e6
        if mb > cfg.get("max_mb", 15):
            print(f"[send] SKIP {f}: {mb:.1f} MB > max_mb"); continue
        if a.dry_run:
            print(f"[send] DRY-RUN would send {f} ({mb:.1f} MB) via {transport}"); continue
        ok, info = fn(cfg, path)
        if ok:
            print(f"[send] SENT {f} via {transport}: {info}")
            for ext in (".mp4", ".json"):
                p = path.replace(".mp4", ext)
                if os.path.exists(p):
                    shutil.move(p, os.path.join(SENT, os.path.basename(p)))
        else:
            print(f"[send] FAILED {f}: {info} — kept in outbox.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init").set_defaults(fn=cmd_init)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    s = sub.add_parser("send"); s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_send)
    args = ap.parse_args(); args.fn(args)
