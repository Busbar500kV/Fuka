# core/organism.py
def advance_boundary(offset: float, motor_mass: float, speed: float, X_env: int) -> float:
    # Offset advances proportionally to how much “motor” we generated in the band.
    return (offset + speed * motor_mass) % float(X_env)