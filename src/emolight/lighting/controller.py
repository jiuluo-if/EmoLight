from emolight.lighting.policy import LightCommand


class SimulatorLightController:
    def __init__(self) -> None:
        self.current: LightCommand | None = None

    def apply(self, command: LightCommand) -> None:
        self.current = command


class MockLightController(SimulatorLightController):
    """In-memory sink used by headless runs and tests."""
