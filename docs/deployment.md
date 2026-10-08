# Model Deployment

## Overview

After compilation, models need to be deployed to the Modalix device for
execution. Model Compiler provides the `llima-deploy` utility to streamline this
process:

``` console
sima-user@docker-image-id:/home/docker$ llima-deploy <source_directory> <destination_directory>
```

Where:

- `source_directory` - Path to the compiled model directory (contains `sima_files/` with `devkit/`, `mpk/`, and optionally `npy_files/` subdirectories)
- `destination_directory` - Target directory on the Modalix device as `[user@]host:/path`, or a local path

When you run this command, the deployment tool performs these steps:

1.  **Validates** that the source directory contains required files (`sima_files/devkit/` and `sima_files/mpk/`)
2.  **Checks** a remote destination before any local work (see [Remote destination check](#remote-destination-check))
3.  **Extracts** ELF files from MPK archives (`*.tar.gz`)
4.  **Syncs** the following to the destination using `rsync`:
    - `devkit/` - Runtime orchestration files
    - `elf_files/` - Extracted binary files
    - `npy_files/` - LoRA adapter weights (automatically included if present)

The tool uses `rsync` internally for efficient file transfer and will skip files that are already up-to-date.

## Deployment Workflow

After compiling your model with `llima-compile`, you'll have a directory structure like:

``` text
Llama-3.2-3B-Instruct_out/
├── onnx_files/
└── sima_files/
    ├── devkit/
    └── mpk/
```

To deploy this to the Modalix device, you have two options:

**Option A: Direct deployment to Modalix device**

If your host machine has network access to the Modalix device:

``` console
sima-user@docker-image-id:/home/docker$ llima-deploy Llama-3.2-3B-Instruct_out sima@192.168.1.20:/media/nvme/llima/llama3_2
```

**Option B: Deploy to local directory for manual transfer**

``` console
sima-user@docker-image-id:/home/docker$ llima-deploy Llama-3.2-3B-Instruct_out llama3_2
sima-user@docker-image-id:/home/docker$ scp -r llama3_2 sima@192.168.1.20:/media/nvme/llima/
```

:::note
`192.168.1.20` is an example Modalix IP address. Use the IP address of your device.
:::

Once deployed, SSH into the Modalix device and run the model:

``` console
modalix:~$ ssh sima@192.168.1.20
```

Then run the model using the `llima` CLI. See [LLiMa CLI](runtime.md) for details.

``` console
modalix:~$ llima run <model_name>
```

## Remote destination check

For a remote destination, `llima-deploy` connects to the device before it
extracts ELF files. It creates the destination directory, checks that it is
writable, and reports the free space:

``` console
sima-user@docker-image-id:/home/docker$ llima-deploy Llama-3.2-3B-Instruct_out sima@192.168.1.20:/media/nvme/llima/llama3_2
Checking sima@192.168.1.20:/media/nvme/llima/llama3_2 ...
sima@192.168.1.20's password:
Destination sima@192.168.1.20:/media/nvme/llima/llama3_2 OK (402.3 GB free).
```

Any password or host-key prompt appears at this point, and `rsync` reuses the
same SSH connection, so the rest of the deployment runs unattended. If the
device cannot be reached or the destination is not writable, the command stops
before extracting anything. If the free space looks smaller than the model, it
prints a warning and continues.

To avoid password prompts entirely, install an SSH key on the device once:

``` console
sima-user@docker-image-id:/home/docker$ ssh-copy-id sima@192.168.1.20
```

Options:

- `--rsh "<command>"` - Remote shell command for both the check and `rsync`,
  for example `--rsh "ssh -p 2222 -i ~/.ssh/modalix"`. It defaults to the
  `RSYNC_RSH` environment variable. A custom remote shell is used as given,
  without connection sharing.
- `--no-preflight` - Skip the check and contact the device only during `rsync`.

`llima-deploy-lora` runs the same check and also verifies that the destination
model directory already contains `elf_files/`.

## Speculative decoding models

When `llima-compile` is given `--draft_model_path`, its output contains target and
draft compiler outputs under one parent. Deploy the parent in one command:

``` console
llima-deploy compiled-eagle3 spec-decoding-output
```

The deployed package contains two ordinary runtime model directories:

``` text
spec-decoding-output/
├── <target-model>/
│   ├── devkit/
│   └── elf_files/
└── <draft-model>/
    ├── devkit/
    └── elf_files/
```

Run the parent directory so LLiMa can identify and load both models from their
serialized speculative-decoding configuration:

``` console
llima run spec-decoding-output
```

## Troubleshooting

**Error: "Cannot reach ..."**

Check the device IP address, the network connection, and your SSH credentials.

**Error: "Destination ... is not writable"**

Check that the storage is mounted (for example `/media/nvme`) and that the SSH
user can write to the destination.

**Error: "devkit directory cannot be found"**

Ensure the source directory is the output directory from `llima-compile`, which should contain `sima_files` subdirectory.

**Error: "mpk directory cannot be found"**

Verify that compilation completed successfully. The `sima_files/mpk/` directory should contain `.tar.gz` files.

**Slow deployment**

- Use `rsync` with compression: The tool uses `rsync -aP` by default
- Deploy to NVMe storage on Modalix for faster model loading
- Consider deploying only changed files using `--resume` during compilation
