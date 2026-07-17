# Deploying to Azure

This deploys the exact `docker-compose.yml` stack (api, mlflow, prometheus,
grafana) to a single Azure VM — see `docs/architecture.md` §5.2 for why a
VM + Docker Compose was chosen over Azure Container Apps / AKS: the course's
own Decision Framework says start monolithic, and running the identical
compose file locally and on Azure means there is nothing to translate or
keep in sync between the two.

**Prerequisites**
- An Azure subscription with available credit (Azure for Students)
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) installed locally
- This repo pushed to GitHub (already done: `https://github.com/PhamLeTanThinh/CDC_Classification`)

## 1. Log in and set variables

```powershell
az login

$RG       = "rg-cdc-cci"
$LOCATION = "southeastasia"   # Singapore — low latency from Vietnam; change if your student subscription restricts regions
$VM_NAME  = "vm-cdc-cci"
$VM_SIZE  = "Standard_B2s"    # 2 vCPU / 4 GB RAM — enough for api+mlflow+prometheus+grafana; comfortably inside student credit
```

## 2. Create a resource group and VM

```powershell
az group create --name $RG --location $LOCATION

az vm create `
  --resource-group $RG `
  --name $VM_NAME `
  --image Ubuntu2204 `
  --size $VM_SIZE `
  --admin-username azureuser `
  --generate-ssh-keys
```

`--generate-ssh-keys` creates (or reuses) `~/.ssh/id_rsa[.pub]` and wires it
up for you — no password auth, no separate key step.

## 3. Open the ports the stack needs

```powershell
az vm open-port --resource-group $RG --name $VM_NAME --port 22   --priority 900  # SSH
az vm open-port --resource-group $RG --name $VM_NAME --port 8000 --priority 910  # API + web UI
az vm open-port --resource-group $RG --name $VM_NAME --port 5000 --priority 920  # MLflow UI
az vm open-port --resource-group $RG --name $VM_NAME --port 9090 --priority 930  # Prometheus UI
az vm open-port --resource-group $RG --name $VM_NAME --port 3000 --priority 940  # Grafana UI
```

> These ports are open to the whole internet with no TLS and no auth in
> front of them — acceptable for a course demo, not for anything handling
> real patient data. See `docs/architecture.md` §8 (Known limitations).

## 4. Get the public IP and SSH in

```powershell
az vm show --resource-group $RG --name $VM_NAME -d --query publicIps -o tsv
ssh azureuser@<PUBLIC_IP>
```

## 5. Install Docker on the VM (run once, inside the SSH session)

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo \"$VERSION_CODENAME\") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

sudo usermod -aG docker $USER
# log out and back in (or `newgrp docker`) so `docker` works without sudo
```

## 6. Clone the repo and start the stack

```bash
git clone https://github.com/PhamLeTanThinh/CDC_Classification.git
cd CDC_Classification
docker compose up --build -d
docker compose ps
```

The first build trains the model as part of `docker build` (see
`Dockerfile`), so this step takes a few minutes; subsequent
`docker compose up` calls reuse the cached image unless the source changed.

## 7. Verify

From your own machine (not the VM):

| URL | What |
|---|---|
| `http://<PUBLIC_IP>:8000/` | Web UI |
| `http://<PUBLIC_IP>:8000/docs` | Swagger UI |
| `http://<PUBLIC_IP>:8000/health` | Health check |
| `http://<PUBLIC_IP>:5000/` | MLflow UI |
| `http://<PUBLIC_IP>:9090/targets` | Prometheus — confirm the `cdc_cci_api` target is `UP` |
| `http://<PUBLIC_IP>:9090/alerts` | Prometheus — the 4 alert rules, state `inactive` until triggered |
| `http://<PUBLIC_IP>:3000/` | Grafana (anonymous viewer access enabled; admin/admin for edit) — "CDC/CCI API Overview" dashboard should already be provisioned |

## 8. Updating after a code change

```bash
cd CDC_Classification
git pull
docker compose up --build -d
```

## 9. Cost control

Stop (deallocate) the VM whenever you're not actively demoing it — you are
billed for compute while it's running, not while stopped:

```powershell
az vm deallocate --resource-group $RG --name $VM_NAME
```

Start it again before a demo:

```powershell
az vm start --resource-group $RG --name $VM_NAME
```

> Stopping/starting changes the public IP unless you allocate a static
> one. For a short-lived course demo this is usually fine — re-run the
> "get public IP" command in step 4 after each start. If you need a fixed
> URL across restarts, allocate a static public IP:
> `az network public-ip update --resource-group $RG --name <ip-resource-name> --allocation-method Static`
> (find `<ip-resource-name>` with `az network public-ip list --resource-group $RG -o table`).

## 10. Tear down

When the project is graded and you no longer need the VM:

```powershell
az group delete --name $RG --yes --no-wait
```

This deletes the VM, its disk, NSG, and public IP together (everything in
the resource group) in one step.
