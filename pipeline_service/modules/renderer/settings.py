from enum import Enum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field

# rrggbb hex, optionally '#'-prefixed in config; normalized to the bare form
HexColor = Annotated[str, Field(pattern=r"^#?[0-9a-fA-F]{6}$"), AfterValidator(lambda v: v.removeprefix("#"))]


class LightingMode(str, Enum):
    FOLLOW = "follow"
    STUDIO = "studio"
    NEUTRAL = "neutral"


class RendererConfig(BaseModel):
    sidecar_host: str = "127.0.0.1"
    sidecar_port: int = 8003
    startup_timeout_s: float = 60.0
    request_timeout_s: float = 90.0
    node_binary: str = "node"
    img_size: int = 518
    grid_gap: int = 5
    bg_color: HexColor | None = None
    lighting: LightingMode = LightingMode.NEUTRAL
    sidecar_count: int = 4
    pool_size: int = 2
    judge_multiview: bool = True
    judge_white_bg: HexColor = "ffffff"
    judge_gray_bg: HexColor = "808080"
    render_timeout_ms: int = 60000
    protocol_timeout_ms: int = 60000
    static_port_base: int = 3000
