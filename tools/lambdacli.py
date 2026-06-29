#------------------- LAMBDA BOT ------------------
# Make sure to export your API-KEY & APP-PASSWORD
# Author: Srinjay

#!/usr/bin/env python3
import os
import sys
import time
import uuid
import requests
import subprocess
import platform
import smtplib
import imaplib
from pathlib import Path
from email.message import EmailMessage
from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text
from prompt_toolkit.shortcuts import radiolist_dialog
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter

console = Console()
session = PromptSession()

NOTIFIER_EMAIL = "lambdabotai@gmail.com"
RECIPIENT_EMAILS = [
    "srinjaycode@gmail.com",
    "yousef.saleh456@gmail.com",
    "pixelsauras@gmail.com",
    "ahmed.rizsoft@gmail.com",
    "hamzahusseini2008@gmail.com",
    "jameelon8@gmail.com",
    "nour.eldien265@gmail.com"
]

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "lambdabotai@gmail.com"
SMTP_PASS = os.environ.get("APP_PASSWORD")

IMAP_SERVER = "imap.gmail.com"
IMAP_USER = "lambdabotai@gmail.com"
IMAP_PASS = os.environ.get("APP_PASSWORD")

API_BASE = os.environ.get("LAMBDA_CLOUD_API_BASE", "https://cloud.lambdalabs.com/api/v1")
API_KEY = os.environ.get("LAMBDA_CLOUD_API_KEY")
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def api_get(path):
    r = requests.get(API_BASE + path, headers=HEADERS)
    r.raise_for_status()
    return r.json().get("data", [])

def api_post(endpoint, payload):
    r = requests.post(API_BASE + endpoint, headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()

def list_instance_types():
    return api_get("/instance-types")

def list_instances():
    return api_get("/instances")

def list_ssh_keys():
    return api_get("/ssh-keys")

def list_filesystems():
    return api_get("/file-systems")

def get_instance(inst_id):
    r = requests.get(f"{API_BASE}/instances/{inst_id}", headers=HEADERS)
    r.raise_for_status()
    return r.json().get("data")

def terminate_instance(inst_id):
    r = requests.post(f"{API_BASE}/instance-operations/terminate", headers=HEADERS, json={"instance_ids": [inst_id]})
    r.raise_for_status()
    return r.json()

def restart_instance(inst_id):
    r = requests.post(f"{API_BASE}/instance-operations/restart", headers=HEADERS, json={"instance_ids": [inst_id]})
    r.raise_for_status()
    return r.json()

def pretty_gpus(gpu_data):
    t = Table(title="GPU Inventory", box=box.HEAVY)
    t.add_column("Index")
    t.add_column("Type", style="cyan")
    t.add_column("GPU", style="green")
    t.add_column("#")
    t.add_column("$/hr")
    t.add_column("Available")
    t.add_column("Regions", style="yellow")
    
    keys = list(gpu_data.keys())
    for i, k in enumerate(keys):
        it = gpu_data[k]["instance_type"]
        regs = gpu_data[k].get("regions_with_capacity_available", [])
        price = f"${it['price_cents_per_hour']/100:.2f}"
        avail = "YES" if regs else "NO"
        col = "green" if regs else "red"
        region_str = ", ".join(r["description"] for r in regs) if regs else "—"
        t.add_row(
            str(i), 
            it["name"], 
            it["gpu_description"], 
            str(it["specs"]["gpus"]), 
            price, 
            Text(avail, style=col), 
            region_str
        )
    console.print(t)

def pretty_instances(insts):
    t = Table(title="Instances", box=box.ROUNDED)
    t.add_column("ID")
    t.add_column("GPU")
    t.add_column("Region")
    t.add_column("Status")
    t.add_column("IP")
    t.add_column("$/hr")
    
    for inst in insts:
        it = inst.get("instance_type", {})
        price = f"${it.get('price_cents_per_hour', 0)/100:.2f}"
        status = inst.get("status", "?")
        color = "green" if status == "active" else "yellow"
        t.add_row(
            inst.get("id", "?"),
            it.get("name", "?"),
            inst.get("region", {}).get("description", "?"),
            Text(status, style=color),
            inst.get("ip", "—"),
            price
        )
    console.print(t)

def choose_radio(title, choices):
    return radiolist_dialog(title=title, text="Use ↑/↓ + Enter", values=choices).run()

def interactive_terminate():
    insts = list_instances()
    if not insts:
        console.print("[yellow]No instances found[/yellow]")
        return
    
    inst_choices = [
        (inst["id"], f"{inst['id']} - {inst.get('instance_type', {}).get('name', '?')} ({inst.get('status', '?')})")
        for inst in insts
    ]
    
    sel_inst = choose_radio("Select instance to terminate", inst_choices)
    if not sel_inst:
        return
    
    console.print(f"[yellow]Terminating instance {sel_inst}...[/yellow]")
    try:
        terminate_instance(sel_inst)
        console.print(f"[green]Instance {sel_inst} terminated successfully[/green]")
    except Exception as e:
        console.print(f"[red]Failed to terminate: {e}[/red]")

def interactive_restart():
    insts = list_instances()
    active_insts = [inst for inst in insts if inst.get("status") == "active"]
    
    if not active_insts:
        console.print("[yellow]No active instances found[/yellow]")
        return
    
    inst_choices = [
        (inst["id"], f"{inst['id']} - {inst.get('instance_type', {}).get('name', '?')}")
        for inst in active_insts
    ]
    
    sel_inst = choose_radio("Select instance to restart", inst_choices)
    if not sel_inst:
        return
    
    console.print(f"[yellow]Restarting instance {sel_inst}...[/yellow]")
    try:
        restart_instance(sel_inst)
        console.print(f"[green]Instance {sel_inst} restarted successfully[/green]")
    except Exception as e:
        console.print(f"[red]Failed to restart: {e}[/red]")

def interactive_gpu_selector():
    gpus = list_instance_types()
    avail = {k: v for k, v in gpus.items() if v.get("regions_with_capacity_available")}
    
    if not avail:
        console.print("[red]No GPUs available[/red]")
        return
    
    gpu_choices = [
        (k, f"{v['instance_type']['name']} ({v['instance_type']['gpu_description']})") 
        for k, v in avail.items()
    ]
    sel_gpu = choose_radio("Select GPU", gpu_choices)
    if not sel_gpu:
        return
    
    gpu = avail[sel_gpu]
    inst_type = gpu["instance_type"]["name"]
    regs = gpu.get("regions_with_capacity_available", [])
    region_choices = [(r["name"], r["description"]) for r in regs]
    sel_region = choose_radio("Select Region", region_choices)
    if not sel_region:
        return

    ssh_keys = list_ssh_keys()
    if not ssh_keys:
        console.print("[red]No SSH keys found[/red]")
        return
    
    ssh_choices = [(k["name"], k["name"]) for k in ssh_keys]
    sel_ssh = choose_radio("Select SSH key", ssh_choices)
    if not sel_ssh:
        return

    fs_list = list_filesystems()
    region_filesystems = [
        f for f in fs_list 
        if f.get("region", {}).get("name") == sel_region
    ]
    
    fs_choices = [("none", "No filesystem")]
    if region_filesystems:
        fs_choices.extend([
            (f["name"], f"{f['name']} ({f.get('region', {}).get('description', '?')})")
            for f in region_filesystems
        ])
    else:
        console.print(f"[yellow]No filesystems available in {sel_region}[/yellow]")
    
    sel_fs = choose_radio("Select filesystem", fs_choices)
    
    console.print("[cyan]Launching instance...[/cyan]")
    res = launch_and_wait(inst_type, sel_region, sel_ssh, sel_fs)
    if res:
        inst_id, ip = res
        ssh_into(sel_ssh, ip)

def launch_and_wait(instance_type, region, ssh_key, filesystem):
    payload = {
        "instance_type_name": instance_type,
        "region_name": region,
        "ssh_key_names": [ssh_key]
    }
    
    if filesystem and filesystem != "none":
        payload["file_system_names"] = [filesystem]
    
    try:
        data = api_post("/instance-operations/launch", payload)
        inst_data = data.get("data", {})
        inst_ids = inst_data.get("instance_ids", [])
        
        if not inst_ids:
            console.print("[red]Failed to launch instance: No instance ID returned[/red]")
            return None
        
        inst_id = inst_ids[0]
        console.print(f"[green]Instance launched with ID: {inst_id}[/green]")
        console.print("[cyan]Waiting for instance to become active...[/cyan]")
        
        max_wait = 600
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                inst = get_instance(inst_id)
                status = inst.get("status")
                ip = inst.get("ip")
                
                console.print(f"[cyan]Status: {status} | IP: {ip or 'pending...'}[/cyan]")
                
                if status == "active" and ip:
                    console.print(f"[green]Instance is ready! IP: {ip}[/green]")
                    time.sleep(5)
                    return inst_id, ip
                elif status in ["unhealthy", "terminated"]:
                    console.print(f"[red]Instance failed with status: {status}[/red]")
                    return None
                    
            except Exception as e:
                console.print(f"[yellow]Error checking instance: {e}[/yellow]")
            
            time.sleep(10)
        
        console.print("[red]Timeout waiting for instance to become active[/red]")
        return None
        
    except Exception as e:
        console.print(f"[red]Launch failed: {e}[/red]")
        return None

def ssh_into(ssh_key, ip):
    is_windows = platform.system() == "Windows"
    ssh_dir = Path.home() / ".ssh"
    
    possible_keys = [
        ssh_dir / f"{ssh_key}.pem",
        ssh_dir / ssh_key
    ]
    
    keyfile = None
    for key_path in possible_keys:
        if key_path.exists():
            keyfile = str(key_path)
            break
    
    if not keyfile:
        console.print(f"[yellow]SSH key not found in {ssh_dir}[/yellow]")
        console.print(f"[yellow]Looked for: {ssh_key}.pem, {ssh_key}[/yellow]")
        console.print(f"[cyan]Manual SSH command: ssh -i ~/.ssh/{ssh_key} ubuntu@{ip}[/cyan]")
        return
    
    console.print(f"[cyan]Connecting via SSH to ubuntu@{ip}[/cyan]")
    
    try:
        subprocess.run([
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-i", keyfile,
            f"ubuntu@{ip}"
        ])
    except FileNotFoundError:
        console.print("[yellow]SSH client not found.[/yellow]")
        console.print(f"[cyan]Manual: ssh -i {keyfile} ubuntu@{ip}[/cyan]")

def send_alert(subject, body):
    if not SMTP_PASS:
        console.print("[red]APP_PASSWORD not set, cannot send email[/red]")
        return
    
    msg = EmailMessage()
    msg["From"] = NOTIFIER_EMAIL
    msg["To"] = ", ".join(RECIPIENT_EMAILS)
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        console.print("[green]Alert email sent successfully[/green]")
    except Exception as e:
        console.print(f"[red]Failed to send email: {e}[/red]")

def wait_for_reply(token):
    if not IMAP_PASS:
        console.print("[red]APP_PASSWORD not set, cannot check email[/red]")
        return False
    
    try:
        M = imaplib.IMAP4_SSL(IMAP_SERVER)
        M.login(IMAP_USER, IMAP_PASS)
        
        console.print(f"[cyan]Waiting for email reply with token: {token}[/cyan]")
        
        while True:
            M.select("INBOX")
            _, data = M.search(None, f'(BODY "{token}")')
            if data[0].split():
                M.logout()
                return True
            time.sleep(15)
    except Exception as e:
        console.print(f"[red]Error checking email: {e}[/red]")
        return False

def watch_rtx():
    token = str(uuid.uuid4())[:8]
    models = ["RTX 6000", "A6000"]

    console.print("[bold cyan]Watching RTX6000 / A6000 availability...[/bold cyan]")
    console.print(f"[yellow]Token for this session: {token}[/yellow]")

    while True:
        try:
            gpus = list_instance_types()

            for k, v in gpus.items():
                it = v["instance_type"]
                regions = v.get("regions_with_capacity_available", [])

                if any(m.lower() in it["gpu_description"].lower() for m in models) and regions:
                    region = regions[0]["name"]
                    region_desc = regions[0]["description"]

                    subject = f"RTX/A6000 AVAILABLE [{token}]"
                    body = f"""GPU Available!

GPU Type: {it['name']}
Description: {it['gpu_description']}
Region: {region_desc}
Price: ${it['price_cents_per_hour']/100:.2f}/hr

Reply to this email with the token to auto-launch: {token}
"""
                    send_alert(subject, body)

                    console.print(f"[bold green]Availability detected! {it['name']} in {region_desc}[/bold green]")
                    console.print("[cyan]Email sent to recipients. Waiting for reply...[/cyan]")

                    if wait_for_reply(token):
                        console.print("[bold yellow]Reply detected! Launching instance...[/bold yellow]")
                        
                        ssh_keys = list_ssh_keys()
                        if ssh_keys:
                            ssh_key = ssh_keys[0]["name"]
                            launch_and_wait(it["name"], region, ssh_key, None)
                        else:
                            console.print("[red]No SSH keys found, cannot launch[/red]")
                        return
                    
            time.sleep(10)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping watch...[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error in watch loop: {e}[/red]")
            time.sleep(10)

def repl():
    if not API_KEY:
        console.print("[red]Error: LAMBDA_CLOUD_API_KEY environment variable not set[/red]")
        sys.exit(1)
    
    comp = WordCompleter(
        ["list-gpus", "list-instances", "launch", "terminate", "restart", "watch-rtx", "exit"],
        ignore_case=True
    )
    
    console.print("[cyan]Lambda Cloud CLI[/cyan]")
    console.print("Commands: list-gpus | list-instances | launch | terminate | restart | watch-rtx | exit")
    
    while True:
        try:
            cmd = session.prompt("lambda> ", completer=comp).strip()
            
            if cmd == "list-gpus":
                pretty_gpus(list_instance_types())
            elif cmd == "list-instances":
                pretty_instances(list_instances())
            elif cmd == "launch":
                interactive_gpu_selector()
            elif cmd == "terminate":
                interactive_terminate()
            elif cmd == "restart":
                interactive_restart()
            elif cmd == "watch-rtx":
                watch_rtx()
            elif cmd == "exit":
                console.print("[cyan]Goodbye![/cyan]")
                return
            elif cmd:
                console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit[/yellow]")
        except EOFError:
            console.print("\n[cyan]Goodbye![/cyan]")
            return

if __name__ == "__main__":
    repl()
