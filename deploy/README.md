# Jetson deployment artifacts

Files the running robot depends on that live **outside** the ROS packages. The boot script itself
is NOT here - it is versioned at `tools/console/aisha_boot.sh`. They were previously
only on the Jetson's disk, unversioned, so a reflash or an accidental overwrite would have lost
them. They are copies — deploying still means putting them back in place.

| File | Belongs at | Notes |
|---|---|---|
| `jetson/aisha-console.service` | `/etc/systemd/system/` | Runs `aisha_boot.sh` on boot. |
| `jetson/ollama-override.conf` | `/etc/systemd/system/ollama.service.d/override.conf` | `OLLAMA_IGPU_ENABLE=1` is what makes Ollama use the Orin's integrated GPU at all. |
| `models/Modelfile.aisha-3b` | — | `ollama create aisha:3b -f Modelfile.aisha-3b` |
| `models/Modelfile.aisha-1b` | — | The smaller fallback. |

## Why the Modelfiles matter
`llama3.2` advertises a 131072-token context. Sized for that, the KV cache alone asks for ~2 GiB of
the Orin's shared 8 GB, `cudaMalloc` fails and `llama-server` dies at startup — the robot then
answers only I encountered an error processing your question. The `num_ctx 2048` cap is what
prevents that. Never point `llm_model:=` at a bare `llama3.2:*` tag; use an `aisha:*` model built
from these files.

Note that `admin_node` must ALSO pass `context_window`, or llama_index overrides the cap anyway.

## Switching model size
Edit the `llm_model:=` argument in `tools/console/aisha_boot.sh`, copy it to `~/aisha_boot.sh`, then
`sudo systemctl restart aisha-console.service`. 3b is ~2.3 GB resident and 5-10 s per answer;
1b is ~1.4 GB and 3-5 s but does not scope its refusals correctly.
