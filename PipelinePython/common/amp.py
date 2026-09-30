import torch


class AMPContext:
    """
    Unified AMP context manager for PyTorch 2.0+.
    Replaces deprecated torch.cuda.amp.autocast and torch.cpu.amp.autocast.
    """

    def __init__(
        self,
        device_type: str = "cuda",
        dtype: torch.dtype = torch.float16,
        enabled: bool = True,
    ):
        """
        Args:
            device_type: "cuda", "cpu", or other supported device
            dtype: torch.float16 or torch.bfloat16
            enabled: whether AMP is enabled
        """
        self.device_type = device_type
        self.dtype = dtype
        self.enabled = enabled

        # CRITICAL: RTX 2060 does NOT support bfloat16 — use float16
        if device_type == "cuda" and dtype == torch.bfloat16:
            raise ValueError(
                "RTX 2060 (Turing architecture) does not support bfloat16. "
                "Use torch.float16 instead."
            )

    def autocast(self) -> torch.amp.autocast:
        """
        Return autocast context manager.
        Usage: with amp_context.autocast():
        """
        return torch.amp.autocast(
            device_type=self.device_type, dtype=self.dtype, enabled=self.enabled
        )

    def get_scaler(self) -> torch.amp.GradScaler:
        """
        Return GradScaler for the current device.
        Usage: scaler = amp_context.get_scaler()
        """
        return torch.amp.GradScaler(self.device_type, enabled=self.enabled)
