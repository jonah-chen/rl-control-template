import os
import json
import hydra
from omegaconf import DictConfig
from filelock import FileLock
import time

# --- The Allocator Class ---
class CpuBinder:
    def __init__(self, lock_file="cpu_registry.json", cpus_per_job=2):
        self.lock_file = lock_file
        self.lock = FileLock(f"{self.lock_file}.lock")
        self.cpus_per_job = cpus_per_job
        self.my_cores = []

    def _init_registry(self):
        """Creates the registry file if it doesn't exist."""
        if not os.path.exists(self.lock_file):
            total_cpus = os.cpu_count()
            # Create a dictionary of slots: { "0,1": "FREE", "2,3": "FREE", ... }
            registry = {}
            for i in range(0, total_cpus, self.cpus_per_job):
                # Ensure we don't go over the total CPU count
                if i + self.cpus_per_job <= total_cpus:
                    core_range = list(range(i, i + self.cpus_per_job))
                    # Key is a string representation of the list, e.g., "0,1"
                    key = ",".join(map(str, core_range))
                    registry[key] = "FREE"
            
            with open(self.lock_file, 'w') as f:
                json.dump(registry, f)

    def __enter__(self):
        """Acquire a slot when entering the 'with' block"""
        with self.lock:
            self._init_registry()
            
            # Read current state
            with open(self.lock_file, 'r') as f:
                registry = json.load(f)

            # Find a FREE slot
            found_slot = None
            for cores, status in registry.items():
                if status == "FREE":
                    found_slot = cores
                    break
            
            if found_slot:
                # Mark as BUSY with current PID (useful for debugging)
                registry[found_slot] = f"BUSY_PID_{os.getpid()}"
                self.my_cores = [int(c) for c in found_slot.split(",")]
                
                # Write back to file
                with open(self.lock_file, 'w') as f:
                    json.dump(registry, f)
                
                # ACTUAL PINNING HAPPENS HERE
                os.sched_setaffinity(0, self.my_cores)
                print(f"[PID {os.getpid()}] Acquired CPUs: {self.my_cores}")
            else:
                # Fallback: No cores available (shouldn't happen if n_jobs < max_cpus)
                print(f"[PID {os.getpid()}] WARNING: No CPU slots free. Running unpinned.")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the slot when exiting the 'with' block"""
        if self.my_cores:
            with self.lock:
                # Read registry again (it might have changed by other jobs)
                with open(self.lock_file, 'r') as f:
                    registry = json.load(f)
                
                # Mark my slot as FREE
                key = ",".join(map(str, self.my_cores))
                registry[key] = "FREE"
                
                # Save
                with open(self.lock_file, 'w') as f:
                    json.dump(registry, f)
                
                print(f"[PID {os.getpid()}] Released CPUs: {self.my_cores}")

