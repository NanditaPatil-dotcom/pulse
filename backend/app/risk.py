def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(value, high))


def heart_rate_risk(hr: int) -> float:
    """
    Heart rate contribution: [0, 0.3]
    """
    if hr < 50:
        return 0.3
    elif 50 <= hr < 60:
        return 0.15
    elif 60 <= hr <= 100:
        return 0.0
    elif 100 < hr <= 120:
        return 0.2
    else:  # hr > 120
        return 0.3


def spo2_risk(spo2: int) -> float:
    """
    SpO2 contribution: [0, 0.5]
    """
    if spo2 >= 95:
        return 0.0
    elif 92 <= spo2 < 95:
        return 0.15
    elif 88 <= spo2 < 92:
        return 0.35
    else:  # spo2 < 88
        return 0.5


def temperature_risk(temp_c: float | None) -> float:
    """
    Temperature contribution: [0, 0.15]
    """
    if temp_c is None:
        return 0.0

    if temp_c <= 37.5:
        return 0.0
    elif 37.5 < temp_c <= 38.5:
        return 0.08
    else:  # temp > 38.5
        return 0.15


def steps_modifier(steps: int | None) -> float:
    """
    Protective modifier: [-0.1, 0]
    """
    if steps is None:
        return 0.0

    if steps > 3000:
        return -0.1
    elif steps > 1000:
        return -0.05
    else:
        return 0.0


def compute_risk(
    heart_rate: int,
    spo2: int,
    temp_c: float | None = None,
    steps: int | None = None,
) -> float:
    """
    Final physiological risk score ∈ [0, 1]
    """
    risk = 0.0

    risk += heart_rate_risk(heart_rate)
    risk += spo2_risk(spo2)
    risk += temperature_risk(temp_c)
    risk += steps_modifier(steps)

    return clamp(risk)
