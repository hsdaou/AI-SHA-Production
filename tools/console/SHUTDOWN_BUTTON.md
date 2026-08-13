# Console "Shut down" button — deploy requirements

The console's ⏻ Shut down button powers off the Pi (best-effort) then the Jetson.
Two things must exist ON THE JETSON (not in the repo — they are host/secret state):

1. **Passwordless poweroff** so the console (systemd service, user hsdaou, no TTY)
   can power the board off:
   ```
   echo 'hsdaou ALL=(root) NOPASSWD: /sbin/poweroff, /usr/sbin/poweroff, \
     /usr/bin/systemctl poweroff, /bin/systemctl poweroff' \
     | sudo tee /etc/sudoers.d/aisha-poweroff
   sudo chmod 440 /etc/sudoers.d/aisha-poweroff
   sudo visudo -cf /etc/sudoers.d/aisha-poweroff      # must say "parsed OK"
   sudo -n -l /sbin/poweroff                          # must be ALLOWED, no prompt
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
