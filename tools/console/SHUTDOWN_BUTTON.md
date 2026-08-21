# Console "Shut down" button — deploy requirements

The console's ⏻ Shut down button powers off the Pi (best-effort) then the Jetson.
Two things must exist ON THE JETSON (not in the repo — they are host/secret state):

1. **Passwordless poweroff** so the console (systemd service, user hsdaou, no TTY)
   can power the board off:
   ```
   # write via a temp file — piping the rule THROUGH `sudo -S tee` feeds the
   # password line to tee instead of the rule (that shipped an EMPTY file once).
   printf 'hsdaou ALL=(root) NOPASSWD: /usr/bin/systemctl poweroff, /sbin/poweroff, /usr/sbin/poweroff\n' > /tmp/pw
   sudo cp /tmp/pw /etc/sudoers.d/aisha-poweroff && rm /tmp/pw
   sudo chmod 440 /etc/sudoers.d/aisha-poweroff
   sudo visudo -cf /etc/sudoers.d/aisha-poweroff      # must say "parsed OK"
   sudo -k && sudo -n -l /usr/bin/systemctl poweroff   # clear cache FIRST, then must be ALLOWED
   # (the credential cache will make -n look allowed right after any sudo -S)
   ```

2. **Pi shutdown config** (0600, outside the repo), holding the Pi's SSH login:
   ```
   ~/robot_console/shutdown.json
   { "pi": { "host": "<Pi address reachable FROM the Jetson>",
             "user": "pi5", "password": "..." } }
   ```
   `host` must be routable from the Jetson. In the USB-link dev setup the Jetson
   cannot reach the Pi, so the Pi step reports "unreachable" and only the Jetson
   powers off — expected. It works fully once both boards share a network.

Also needs `sshpass` on the Jetson (`sudo apt-get install -y sshpass`).

Tested 2026-08-13 with the poweroff stubbed: button -> POST /shutdown 200 ->
Pi step -> Jetson poweroff fired. `sudo -n /sbin/poweroff` verified authorized.
