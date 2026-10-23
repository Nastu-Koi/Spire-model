"""Imported by the feeder server alone (`loader._feeders` preloads it): the server
starts the one shared-memory manager its feeders and their packers register with.

Under torch's `file_system` sharing, a process that shares a tensor while it knows of
no manager starts one (`torch_shm_manager`); a process forked later knows of its
parent's and registers with that. A process that receives a shared tensor connects to
the manager the tensor names and stays connected for as long as it lives: a manager
the trainer received from runs until the trainer exits. Forked from a server that
knew of none, every packer of every round started a manager of its own, and the
trainer kept them all (some thirty a round, about 60 MB each)."""

import torch

from main.learn.loader import _share_by_name

_share_by_name()
# Held for the server's life, so that its manager serves every feeder it forks.
MANAGED = torch.zeros(1).share_memory_()
